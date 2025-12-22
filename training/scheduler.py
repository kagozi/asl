# ============================================================================
# training/scheduler.py
# ============================================================================
"""Learning rate schedulers for transformer training."""

import torch
from torch.optim import Optimizer
from typing import Dict


class NoamLRScheduler:
    """
    Noam learning rate scheduler from "Attention is All You Need".
    Implements warmup followed by inverse square root decay.
    """
    
    def __init__(
        self, 
        optimizer: Optimizer, 
        d_model: int, 
        warmup_steps: int = 4000, 
        factor: float = 1.0
    ):
        """
        Args:
            optimizer: PyTorch optimizer
            d_model: Model dimension (embedding size)
            warmup_steps: Number of warmup steps
            factor: Scaling factor for learning rate
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.factor = factor
        self.d_model = d_model
        self._step = 0
        self._rate = 0
    
    def state_dict(self) -> Dict:
        """Returns the state of the scheduler as a dict."""
        return {
            'step': self._step,
            'rate': self._rate,
            'warmup_steps': self.warmup_steps,
            'factor': self.factor,
            'd_model': self.d_model
        }
    
    def load_state_dict(self, state_dict: Dict):
        """Loads the scheduler's state."""
        self._step = state_dict['step']
        self._rate = state_dict['rate']
        self.warmup_steps = state_dict['warmup_steps']
        self.factor = state_dict['factor']
        self.d_model = state_dict['d_model']
    
    def step(self):
        """Update learning rate."""
        self._step += 1
        rate = self.get_rate()
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = rate
        self._rate = rate
    
    def get_rate(self, step: int = None) -> float:
        """Calculate learning rate for given step."""
        if step is None:
            step = self._step
        
        if step == 0:
            step = 1
        
        return self.factor * (
            self.d_model ** (-0.5) * 
            min(step ** (-0.5), step * self.warmup_steps ** (-1.5))
        )
    
    def get_last_lr(self):
        """Return last computed learning rate."""
        return [self._rate]
