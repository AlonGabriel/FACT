import pathlib
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.utils.data as data
from ignite.engine import Events
from ignite.handlers import (
    EarlyStopping,
    ModelCheckpoint,
    ProgressBar,
    WandBLogger,
)
from ignite.metrics import RunningAverage
from readable_number import ReadableNumber
from sacred import Experiment

import evaluators
import losses
import trainers
import transforms
from augmentation import construct_augmenter
from datasets import (
    NumpyDataset,
    ZippedLoader,
    from_npz,
)
from models import construct_model
from utils import (
    register_configs_files,
    restore_best,
)
from models import CLAPBasedModel

ex = Experiment(save_git_info=False)
register_configs_files(ex)


@ex.capture
def make_model(base_model, projection_head, prediction_head, random_init, checkpoint, _log, supervised, reinit_proj_head=False):
    model = construct_model(base_model, projection_head, random_init, prediction_head, reinit_proj_head)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_params = ReadableNumber(num_params, use_shortform=True)
    _log.info(f'Trainable parameters: {num_params}')
    if checkpoint:
        checkpoint = torch.load(checkpoint)
        state_dict = checkpoint['model']
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            _log.warning(f'Following keys are missing from model checkpoint: {missing}')
        if unexpected:
            _log.warning(f'Model checkpoint has unexpected keys: {unexpected}')
    if supervised:
        model.proj_head = None
        model.pred_head = torch.nn.Sequential(
            torch.nn.Linear(768, 128),  # Input size matches encoder output
            torch.nn.ReLU(),
            torch.nn.Linear(128, 2)    # Number of classes
        )
    return model


@ex.capture
def make_criterion(loss_fn, loss_params):
    constructor = getattr(losses, loss_fn) if hasattr(losses, loss_fn) else getattr(nn, loss_fn)
    return constructor(**loss_params)


@ex.capture
def make_optimizer(model, optimizer, learning_rate, weight_decay):
    constructor = getattr(optim, optimizer)
    return constructor(model.parameters(), lr=learning_rate, weight_decay=weight_decay)


@ex.capture
def make_transform(transform, transform_params, checkpoints_dir):
    if transform is None:
        return None

    if transform == "MinMaxNormalize":
        scaler_path = pathlib.Path(checkpoints_dir) / "scaler.joblib"
        normalizer = transforms.MinMaxNormalize(scaler_path)

        # Load the scaler if it exists
        if scaler_path.exists():
            normalizer.load_scaler(scaler_path)
        return normalizer
    if transform == "SampleNormalize":
        return transforms.SampleNormalize()
    
    # Default case for other transforms
    constructor = getattr(transforms, transform)
    return constructor(**transform_params)


@ex.capture
def make_loaders(dataset, weighted_sampling, batch_size, num_workers, unlabeled_data, unlabeled_ratio, checkpoints_dir, transform, transform_params):
    dataset = np.load(dataset, allow_pickle=True)
    transform = make_transform(transform, transform_params, checkpoints_dir)
    scaler_path = pathlib.Path(checkpoints_dir) / "scaler.joblib"
    train_set = from_npz(dataset, 'train', transform=transform, scaler_path=scaler_path)
    valid_set = from_npz(dataset, 'val', transform=transform, scaler_path=scaler_path)
    test_set = from_npz(dataset, 'test', transform=transform, scaler_path=scaler_path)
    if weighted_sampling:
        print("Using Weighted Sampling in DataLoader")
        train_loader = data.DataLoader(train_set, batch_size=batch_size, sampler=train_set.weighted_sampler(), num_workers=num_workers)
    else:
        print("Using Random Shuffling in DataLoader")
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    return train_loader, valid_loader, test_loader


@ex.capture
def make_trainer(model, criterion, optimizer, device, trainer, base_model, _config):
    constructor = getattr(trainers, trainer)
#     if base_model == "pluskal-lab/DreaMS":
#         augmenter = None
#     else:
    augmenter = construct_augmenter(base_model)
    return constructor(model, criterion, optimizer, augmenter, device, _config)


@ex.capture
def make_evaluator(model, criterion, device, evaluator, base_model):
    factory = getattr(evaluators, evaluator)
#     if base_model == "pluskal-lab/DreaMS":
#         augmenter = None
#     else:
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


@ex.capture
def make_early_stopper(trainer, patience):
    def neg_loss(engine):
        return -engine.state.metrics['loss']
    return EarlyStopping(patience, score_function=neg_loss, trainer=trainer.engine, cumulative_delta=True)


@ex.capture
def make_scheduler_updater(scheduler):
    def update_scheduler(engine):
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
        print(f"Epoch {engine.state.epoch}: Learning Rate is {current_lr}")
    return update_scheduler


@ex.automain
def main(device, num_epochs, early_stopping, eval_only, test_eval_freq, eval_on_best_checkpoint, checkpoints_dir, use_scheduler, learning_rate, save_model, checkpoint_path, save_path, _log):
    device = torch.device(device)
    model = make_model()
    model = model.to(device)
    criterion = make_criterion()
    optimizer = make_optimizer(model)
    train_set, valid_set, test_set = make_loaders()
    trainer = make_trainer(model, criterion, optimizer, device)
    validator = make_evaluator(model, criterion, device)
    tester = make_evaluator(model, criterion, device)
    best_bester = make_evaluator(model, criterion, device)
    if use_scheduler:
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=0.00001)
        #scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=3, T_mult=2, eta_min=0.00001)
        update_scheduler = make_scheduler_updater(scheduler)
        trainer.engine.add_event_handler(Events.EPOCH_COMPLETED, update_scheduler)
    # Callbacks
    if early_stopping:
        handler = make_early_stopper(trainer)
        validator.add_event_handler(Events.COMPLETED, handler)
    trainer.on(Events.EPOCH_COMPLETED, validator.run, valid_set)
    test_event = Events.EPOCH_COMPLETED(every=test_eval_freq) if test_eval_freq > 0 else Events.COMPLETED
    trainer.on(test_event, tester.run, test_set)
    # Loss and Progress Bar
    RunningAverage(output_transform=lambda x: x).attach(trainer.engine, 'loss')
    ProgressBar(persist=True).attach(trainer.engine, ['loss'])
    # Checkpointing
    make_checkpointer(trainer, validator)
    # Logging
    wandb = make_logger(trainer, validator, tester, best_bester)
    # Training and Evaluation
    if not eval_only:
        trainer.run(train_set, num_epochs)
    else:
        tester.run(test_set)
    # Evaluation using the best checkpoint
    if eval_on_best_checkpoint:
        restore_best(checkpoints_dir, model)
        best_bester.run(test_set)

    # Cleanup
    wandb.close()
