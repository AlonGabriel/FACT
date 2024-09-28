import pathlib
import re

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
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

ex = Experiment(save_git_info=False)
register_configs_files(ex)


@ex.capture
def make_model(base_model, projection_head, prediction_head, random_init, checkpoint, _log):
    model = construct_model(base_model, projection_head, random_init, prediction_head)
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
    return model


@ex.capture
def make_criterion(loss_fn, loss_params):
    constructor = getattr(losses, loss_fn) if hasattr(losses, loss_fn) else getattr(nn, loss_fn)
    return constructor(**loss_params)


@ex.capture
def make_optimizer(model, optimizer, learning_rate):
    constructor = getattr(optim, optimizer)
    return constructor(model.parameters(), lr=learning_rate)


@ex.capture
def make_loaders(dataset, batch_size, weighted_sampling, num_workers, unlabeled_data, unlabeled_ratio):
    dataset = np.load(dataset)
    train_set = from_npz(dataset, 'train')
    valid_set = from_npz(dataset, 'val')
    test_set = from_npz(dataset, 'test')
    if weighted_sampling:
        train_loader = data.DataLoader(train_set, batch_size=batch_size, sampler=train_set.weighted_sampler(), num_workers=num_workers)
    else:
        train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, num_workers=num_workers)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, num_workers=num_workers)
    if unlabeled_data is not None:
        unlabeled_data = np.load(unlabeled_data)
        unlabeled_batch_size = int(batch_size * unlabeled_ratio)
        unlabeled_dataset = NumpyDataset(**unlabeled_data, output_labels=False)
        unlabeled_loader = data.DataLoader(unlabeled_dataset, unlabeled_batch_size, num_workers=num_workers)
        return ZippedLoader(train_loader, unlabeled_loader), valid_loader, test_loader
    return train_loader, valid_loader, test_loader


@ex.capture
def make_trainer(model, criterion, optimizer, device, trainer, _config):
    constructor = getattr(trainers, trainer)
    return constructor(model, criterion, optimizer, device, _config)


@ex.capture
def make_evaluator(model, criterion, device, evaluator):
    factory = getattr(evaluators, evaluator)
    factory = factory(model, criterion, device)
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

    if resume:  # From the most recent checkpoint before preemption
        checkpoints_dir = pathlib.Path(checkpoints_dir)
        checkpoints = {filepath: int(re.search(r'checkpoint_(\d+)', str(filepath)).group(1)) for filepath in checkpoints_dir.glob('*.pt')}
        if checkpoints:
            checkpoint = max(checkpoints, key=checkpoints.get)
            checkpointer.load_objects(state_dict, checkpoint)
            _log.info(f'Resuming training from checkpoint: {checkpoint}')


@ex.capture
def make_logger(trainer, validator, tester, best_bester, name, project, _config):
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
    return wandb


@ex.capture
def make_early_stopper(trainer, patience):
    def neg_loss(engine):
        return -engine.state.metrics['loss']
    return EarlyStopping(patience, score_function=neg_loss, trainer=trainer.engine, cumulative_delta=True)


@ex.automain
def main(device, num_epochs, early_stopping, eval_only, test_eval_freq, eval_on_best_checkpoint, checkpoints_dir):
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
