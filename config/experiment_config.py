"""Experiment configuration with grid search support."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import List, Dict, Any
import itertools


@dataclass
class ExperimentConfig:
    """Configuration for a single experiment run."""
    
    # Model architecture
    d_model: int = 512
    nhead: int = 8
    num_encoder_layers: int = 4
    num_decoder_layers: int = 4
    num_kv_heads: int = 4
    ffn_mult: float = 2.7
    dropout: float = 0.1
    use_rope: bool = True
    
    # Training
    batch_size: int = 32
    grad_accum: int = 32
    epochs: int = 50
    warmup_steps: int = 4000
    lr_factor: float = 1.0
    label_smoothing: float = 0.1
    
    # Data
    max_decode_len: int = 100
    
    # System
    seed: int = 42
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def get_run_name(self) -> str:
        """Generate unique run name from config."""
        return (f"gsl_d{self.d_model}_h{self.nhead}"
                f"_enc{self.num_encoder_layers}_dec{self.num_decoder_layers}"
                f"_kv{self.num_kv_heads}_ffn{self.ffn_mult}"
                f"_drop{self.dropout}_bs{self.batch_size}"
                f"_rope{int(self.use_rope)}_seed{self.seed}")


class ExperimentGrid:
    """Generate experiment configurations from parameter grid."""
    
    def __init__(self, base_config: ExperimentConfig = None):
        self.base_config = base_config or ExperimentConfig()
        self.param_grid: Dict[str, List[Any]] = {}
    
    def add_param(self, param_name: str, values: List[Any]):
        """Add a parameter to sweep over."""
        self.param_grid[param_name] = values
        return self
    
    def generate_configs(self) -> List[ExperimentConfig]:
        """Generate all combinations of parameters."""
        if not self.param_grid:
            return [self.base_config]
        
        configs = []
        param_names = list(self.param_grid.keys())
        param_values = [self.param_grid[k] for k in param_names]
        
        for combination in itertools.product(*param_values):
            config_dict = self.base_config.to_dict()
            for param_name, value in zip(param_names, combination):
                config_dict[param_name] = value
            configs.append(ExperimentConfig(**config_dict))
        
        return configs


def create_default_grid() -> ExperimentGrid:
    """Create a comprehensive parameter grid for GSL experiments."""
    grid = ExperimentGrid()
    
    # Architecture variations
    grid.add_param('d_model', [256, 512])
    grid.add_param('nhead', [4, 8])
    grid.add_param('num_encoder_layers', [2, 4, 6])
    grid.add_param('num_decoder_layers', [2, 4, 6])
    grid.add_param('num_kv_heads', [2, 4])
    grid.add_param('ffn_mult', [2.0, 2.7, 4.0])
    grid.add_param('dropout', [0.1, 0.2, 0.3])
    grid.add_param('use_rope', [True, False])
    
    return grid


def create_quick_grid() -> ExperimentGrid:
    """Create a smaller grid for quick testing."""
    grid = ExperimentGrid()
    
    grid.add_param('d_model', [256, 512])
    grid.add_param('num_encoder_layers', [2, 4])
    grid.add_param('num_decoder_layers', [2, 4])
    grid.add_param('dropout', [0.1, 0.2])
    
    return grid


def create_focused_grid() -> ExperimentGrid:
    """Create a focused grid based on promising parameters."""
    grid = ExperimentGrid()
    
    # Keep d_model and nhead fixed
    base = ExperimentConfig(d_model=512, nhead=8)
    grid.base_config = base
    
    # Focus on depth and regularization
    grid.add_param('num_encoder_layers', [3, 4, 5])
    grid.add_param('num_decoder_layers', [3, 4, 5])
    grid.add_param('dropout', [0.1, 0.15, 0.2])
    grid.add_param('ffn_mult', [2.0, 2.7, 3.5])
    
    return grid
