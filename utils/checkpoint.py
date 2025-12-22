# ============================================================================
# utils/checkpoint.py
# ============================================================================
"""Checkpoint management utilities."""

import torch
from pathlib import Path
from typing import Dict, Optional
import shutil


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    metrics: Dict,
    config: Dict,
    path: Path,
    is_best: bool = False
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Learning rate scheduler
        epoch: Current epoch
        metrics: Training metrics
        config: Model configuration
        path: Path to save checkpoint
        is_best: Whether this is the best model
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'metrics': metrics,
        'config': config
    }
    
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)
    
    if is_best:
        best_path = path.parent / 'best_model.pt'
        shutil.copy(path, best_path)


def load_checkpoint(
    path: Path,
    model,
    optimizer=None,
    scheduler=None,
    device=None
) -> Dict:
    """
    Load model checkpoint.
    
    Args:
        path: Path to checkpoint
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        scheduler: Optional scheduler to load state
        device: Device to load checkpoint on
    
    Returns:
        Checkpoint dictionary
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    checkpoint = torch.load(path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint