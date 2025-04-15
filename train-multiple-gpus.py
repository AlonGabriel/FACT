import pathlib
import re
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.utils.data as data
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.optim import lr_scheduler
from ignite.engine import Events
from ignite.handlers import EarlyStopping, ModelCheckpoint, ProgressBar
from ignite.metrics import RunningAverage
from sacred import Experiment
from readable_number import ReadableNumber

import evaluators
import losses
import trainers
import transforms
from augmentation import construct_augmenter
from datasets import from_npz
from models import construct_model
from utils import register_configs_files, restore_best
from models import CLAPBasedModel
from ignite.contrib.handlers.wandb_logger import WandBLogger

ex = Experiment(save_git_info=False)
register_configs_files(ex)


### 🔹 **Distributed Training Setup**
def setup_distributed():
    """Initialize distributed training"""
    dist.init_process_group(backend="nccl", init_method="env://")


def cleanup():
    """Cleanup distributed training"""
    dist.destroy_process_group()


### 🔹 **Model Creation**
@ex.capture
def make_model(base_model, projection_head, prediction_head, random_init, checkpoint, _log, supervised, reinit_proj_head=False):
    model = construct_model(base_model, projection_head, random_init, prediction_head, reinit_proj_head)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = ReadableNumber(num_params, use_shortform=True)
    _log.info(f'Trainable parameters: {num_params}')
    
    if checkpoint:
        checkpoint = torch.load(checkpoint, map_location="cpu")
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict, strict=False)
    
    if supervised:
        model.proj_head = None
        model.pred_head = nn.Sequential(
            nn.Linear(768, 128),
            nn.ReLU(),
            nn.Linear(128, 2)
        )
    
    return model


### 🔹 **Loss Function**
@ex.capture
def make_criterion(loss_fn, loss_params):
    constructor = getattr(losses, loss_fn) if hasattr(losses, loss_fn) else getattr(nn, loss_fn)
    return constructor(**loss_params)


### 🔹 **Optimizer**
@ex.capture
def make_optimizer(model, optimizer, learning_rate, weight_decay):
    constructor = getattr(optim, optimizer)
    return constructor(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


### 🔹 **Transform Function**
@ex.capture
def make_transform(transform, transform_params, checkpoints_dir):
    if transform is None:
        return None

    if transform == "MinMaxNormalize":
        scaler_path = pathlib.Path(checkpoints_dir) / "scaler.joblib"
        normalizer = transforms.MinMaxNormalize(scaler_path)
        if scaler_path.exists():
            normalizer.load_scaler(scaler_path)
        return normalizer
    if transform == "SampleNormalize":
        return transforms.SampleNormalize()

    constructor = getattr(transforms, transform)
    return constructor(**transform_params)


### 🔹 **Data Loaders with Distributed Sampler**
@ex.capture
def make_loaders(dataset, weighted_sampling, batch_size, num_workers, unlabeled_data, 
                 unlabeled_ratio, checkpoints_dir, transform, transform_params):
    dataset = np.load(dataset, allow_pickle=True)
    transform = make_transform(transform, transform_params, checkpoints_dir)
    scaler_path = pathlib.Path(checkpoints_dir) / "scaler.joblib"
    train_set = from_npz(dataset, 'train', transform=transform, scaler_path=scaler_path)
    valid_set = from_npz(dataset, 'val', transform=transform, scaler_path=scaler_path)
    test_set = from_npz(dataset, 'test', transform=transform, scaler_path=scaler_path)

    rank = dist.get_rank()  # Get GPU rank
    world_size = dist.get_world_size()  # Number of GPUs

    train_sampler = DistributedSampler(train_set, num_replicas=world_size, rank=rank, shuffle=True)
    valid_sampler = DistributedSampler(valid_set, num_replicas=world_size, rank=rank, shuffle=False)
    test_sampler = DistributedSampler(test_set, num_replicas=world_size, rank=rank, shuffle=False)

    train_loader = data.DataLoader(train_set, batch_size=batch_size // world_size, sampler=train_sampler, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size // world_size, sampler=valid_sampler, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=batch_size // world_size, sampler=test_sampler, num_workers=num_workers)

    return train_loader, valid_loader, test_loader


### 🔹 **Trainer and Evaluator**
@ex.capture
def make_trainer(model, criterion, optimizer, device, trainer, base_model, _config):
    constructor = getattr(trainers, trainer)
    augmenter = construct_augmenter(base_model)
    return constructor(model, criterion, optimizer, augmenter, device, _config)


@ex.capture
def make_evaluator(model, criterion, device, evaluator, base_model):
    factory = getattr(evaluators, evaluator)
    augmenter = construct_augmenter(base_model)
    factory = factory(model, criterion, augmenter, device)
    return factory.create_engine()


@ex.capture
def make_checkpointer(trainer, validator, checkpoint_interval, checkpoints_dir, monitor, objective, num_saved, resume, _log):
    assert objective in ('minimize', 'maximize')

    state_dict = trainer.state_dict()

    interval_checkpointer = ModelCheckpoint(
        checkpoints_dir,
        n_saved=None,
        create_dir=True,
        require_empty=False,
        global_step_transform=lambda engine, event: trainer.engine.state.epoch,
    )
    trainer.on(Events.EPOCH_COMPLETED(every=checkpoint_interval), interval_checkpointer, validator, state_dict)

    def score_fn(engine):
        score = engine.state.metrics[monitor]
        return -score if objective == 'minimize' else score

    checkpointer = ModelCheckpoint(
        checkpoints_dir,
        n_saved=num_saved,
        create_dir=True,
        require_empty=False,
        score_name=monitor,
        score_function=score_fn,
        global_step_transform=lambda engine, event: trainer.engine.state.epoch,
    )
    validator.add_event_handler(Events.COMPLETED, checkpointer, validator, state_dict)
    
    final_checkpointer = ModelCheckpoint(
        checkpoints_dir,
        n_saved=1,  # Only save the most recent final checkpoint
        create_dir=True,
        require_empty=False,
        global_step_transform=lambda engine, event: trainer.engine.state.epoch,
    )
    trainer.on(Events.COMPLETED, final_checkpointer, validator, state_dict)
        
    if resume:  # From the most recent checkpoint before preemption
        checkpoints_dir = pathlib.Path(checkpoints_dir)
        checkpoints = {filepath: int(re.search(r'checkpoint_(\d+)', str(filepath)).group(1)) for filepath in checkpoints_dir.glob('*.pt')}
        if checkpoints:
            checkpoint = max(checkpoints, key=checkpoints.get)
            checkpointer.load_objects(state_dict, checkpoint)
            _log.info(f'Resuming training from checkpoint: {checkpoint}')
        
        
@ex.capture
def make_logger(trainer, validator, tester, best_bester, name, project, _config, checkpoint):
    wandb = WandBLogger(
        name=name,
        project=project,
        job_type='train',
        config=_config,
    )
    wandb.attach_output_handler(
        trainer.engine,
        tag='train',
        metric_names='all',
        event_name=Events.ITERATION_COMPLETED,
        global_step_transform=lambda engine, event: trainer.engine.state.iteration,
    )
    wandb.attach_output_handler(
        validator,
        tag='valid',
        metric_names='all',
        event_name=Events.COMPLETED,
        global_step_transform=lambda engine, event: trainer.engine.state.iteration,
    )
    wandb.attach_output_handler(
        tester,
        tag='test',
        metric_names='all',
        event_name=Events.COMPLETED,
        global_step_transform=lambda engine, event: trainer.engine.state.iteration,
    )
    wandb.attach_output_handler(
        best_bester,
        tag='test/best',
        metric_names='all',
        event_name=Events.COMPLETED,
        global_step_transform=lambda engine, event: trainer.engine.state.iteration,
    )
    
    wandb.log({"test/checkpoint_used": checkpoint})

    return wandb


### 🔹 **Early Stopping**
@ex.capture
def make_early_stopper(trainer, patience):
    def neg_loss(engine):
        return -engine.state.metrics['loss']
    return EarlyStopping(patience, score_function=neg_loss, trainer=trainer.engine)


### 🔹 **Scheduler Updater**
@ex.capture
def make_scheduler_updater(scheduler):
    def update_scheduler(engine):
        scheduler.step()
        print(f"Epoch {engine.state.epoch}: Learning Rate is {scheduler.get_last_lr()[0]}")
    return update_scheduler


### 🔹 **Main Function (With DDP)**
@ex.automain
def main(device, num_epochs, early_stopping, eval_only, test_eval_freq, eval_on_best_checkpoint, 
         checkpoints_dir, use_scheduler, learning_rate, save_model, checkpoint_path, save_path, 
         patience, name, project, _log):
    torch.autograd.set_detect_anomaly(True)
    setup_distributed()  # Initialize multi-GPU training
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        _log.info(f"Running on {world_size} GPUs...")

    # ✅ Model setup (DDP applied)
    model = make_model().to(device)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(model, device_ids=[rank], output_device=rank, find_unused_parameters=True)

    # ✅ Criterion, Optimizer, Data Loaders
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    train_set, valid_set, test_set = make_loaders()
    
    # ✅ Trainer & Evaluators
    trainer = make_trainer(model, criterion, optimizer, device)
    validator = make_evaluator(model, criterion, device)
    tester = make_evaluator(model, criterion, device)
    best_bester = make_evaluator(model, criterion, device)

    # ✅ Learning Rate Scheduler
    if use_scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
        update_scheduler = make_scheduler_updater(scheduler)
        trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, update_scheduler)

    # ✅ Early Stopping
    if early_stopping:
        handler = make_early_stopper(trainer, patience)
        validator.add_event_handler(Events.COMPLETED, handler)

    # ✅ Attach Validators & Testers
    trainer.on(Events.EPOCH_COMPLETED, validator.run, valid_set)
    test_event = Events.EPOCH_COMPLETED(every=test_eval_freq) if test_eval_freq > 0 else Events.COMPLETED
    trainer.on(test_event, tester.run, test_set)
    
    # ✅ Attach Progress Bar & Loss Tracking
    RunningAverage(output_transform=lambda x: x).attach(trainer.engine, 'loss')
    ProgressBar(persist=True).attach(trainer.engine, ['loss'])

    if rank == 0:
        # ✅ Checkpointing
        make_checkpointer(trainer, validator)
        
        # ✅ WandB Logging
        wandb = make_logger(trainer, validator, tester, best_bester, name, project, checkpoint=checkpoint_path)

    # ✅ Training & Evaluation Loop
    if not eval_only:
        trainer.run(train_set, num_epochs)
    else:
        tester.run(test_set)

    # ✅ Evaluate with Best Checkpoint (if enabled)
    if eval_on_best_checkpoint:
        restore_best(checkpoints_dir, model)
        best_bester.run(test_set)

    # ✅ Cleanup & Exit
    if rank == 0:
        wandb.close()

    cleanup()
