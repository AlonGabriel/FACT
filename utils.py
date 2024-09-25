import os
from pathlib import Path


def register_configs_files(ex, base_dir=Path('configs')):
    ex.add_config(str(base_dir / 'defaults.yaml'))
    for filepath in base_dir.iterdir():
        if not filepath.name.endswith('.yaml') or filepath.name == 'defaults.yaml':
            continue
        name, ext = os.path.splitext(filepath.name)
        ex.add_named_config(name, str(filepath))
