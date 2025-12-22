# ============================================================================
# config/__init__.py
# ============================================================================
"""Configuration loading utilities."""

import yaml
from pathlib import Path
from typing import Dict


def load_config(config_path: Path) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict, save_path: Path):
    """Save configuration to YAML file."""
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


__all__ = ['load_config', 'save_config']