import os
import re
from pathlib import Path

import torch
from ignite.utils import convert_tensor


def register_configs_files(ex, base_dir=Path('configs')):
    ex.add_config(str(base_dir / 'defaults.yaml'))
    for filepath in base_dir.iterdir():
        if not filepath.name.endswith('.yaml') or filepath.name == 'defaults.yaml':
            continue
        name, ext = os.path.splitext(filepath.name)
        ex.add_named_config(name, str(filepath))


def restore_best(checkpoints_dir, model):
    checkpoints_dir = Path(checkpoints_dir)
    pattern = re.compile(r'checkpoint_(\d+)_([a-z_]+)=(-?[0-9]+\.[0-9]+)\.pt')
    checkpoints = {filepath: re.search(pattern, str(filepath)) for filepath in checkpoints_dir.glob('*.pt')}
    checkpoints = {filepath: int(match.group(1)) for filepath, match in checkpoints.items() if match}
    checkpoint = max(checkpoints, key=checkpoints.get)
    state_dict = torch.load(checkpoint)
    model.load_state_dict(state_dict['model'])
    return model


def prepare_batch(batch, device):
    return [convert_tensor(el, device=device) for el in batch]
