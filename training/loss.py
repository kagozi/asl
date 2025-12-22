# ============================================================================
# training/loss.py
# ============================================================================
"""Loss functions for transformer training."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LabelSmoothing(nn.Module):
    """
    Label smoothing regularization.
    Prevents the model from becoming overconfident.
    """
    
    def __init__(self, smoothing: float = 0.1, ignore_index: int = 0):
        """
        Args:
            smoothing: Smoothing factor (0.0 = no smoothing, 0.1 = 10% smoothing)
            ignore_index: Index to ignore (padding token)
        """
        super().__init__()
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.ignore_index = ignore_index
    
    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred: Predictions (batch_size * seq_len, vocab_size)
            target: Target labels (batch_size * seq_len)
        
        Returns:
            Smoothed cross-entropy loss
        """
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            # Create the true distribution
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (pred.size(-1) - 2))  # -2 for pad and target
            true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
            
            # Create mask for padding tokens
            mask = (target != self.ignore_index).float()
            true_dist = true_dist * mask.unsqueeze(-1)
        
        loss = -torch.sum(true_dist * pred) / torch.sum(mask)
        return loss