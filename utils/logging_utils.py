# ============================================================================
# utils/logging_utils.py
# ============================================================================
"""Logging utilities."""

import logging
from pathlib import Path
from datetime import datetime
import sys


def setup_logger(name: str, log_dir: Path, level=logging.INFO) -> logging.Logger:
    """
    Set up logger with file and console handlers.
    
    Args:
        name: Logger name
        log_dir: Directory to save log files
        level: Logging level
    
    Returns:
        Configured logger
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers
    logger.handlers = []
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = log_dir / f'{name}_{timestamp}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(level)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger


# ============================================================================
# utils/visualization.py
# ============================================================================
"""Visualization utilities for training metrics."""

import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np

sns.set_style('whitegrid')


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    learning_rates: List[float],
    save_path: Path
):
    """
    Plot training curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        learning_rates: Learning rates per epoch
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    epochs = range(1, len(train_losses) + 1)
    axes[0].plot(epochs, train_losses, label='Train Loss', marker='o', markersize=3)
    axes[0].plot(epochs, val_losses, label='Val Loss', marker='s', markersize=3)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Plot learning rate
    axes[1].plot(epochs, learning_rates, marker='o', markersize=3, color='green')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Learning Rate')
    axes[1].set_title('Learning Rate Schedule')
    axes[1].set_yscale('log')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {save_path}")


def plot_metric_comparison(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Path,
    title: str = "Model Comparison"
):
    """
    Plot comparison of multiple models across metrics.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        save_path: Path to save plot
        title: Plot title
    """
    # Convert to DataFrame
    df = pd.DataFrame(results_dict).T
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(df.columns))
    width = 0.8 / len(df)
    
    for i, (model_name, row) in enumerate(df.iterrows()):
        offset = (i - len(df)/2) * width + width/2
        ax.bar(x + offset, row.values, width, label=model_name, alpha=0.8)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(df.columns, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved metric comparison to {save_path}")


def save_metrics_table(
    results_dict: Dict[str, Dict[str, float]],
    save_path: Path
):
    """
    Save metrics as a formatted table.
    
    Args:
        results_dict: Dictionary mapping model names to their metrics
        save_path: Path to save table (CSV or Markdown)
    """
    df = pd.DataFrame(results_dict).T
    
    # Round to 2 decimal places
    df = df.round(2)
    
    # Save based on extension
    if save_path.suffix == '.csv':
        df.to_csv(save_path)
    elif save_path.suffix == '.md':
        with open(save_path, 'w') as f:
            f.write(df.to_markdown())
    else:
        # Default to CSV
        df.to_csv(save_path.with_suffix('.csv'))
    
    print(f"Saved metrics table to {save_path}")


def plot_length_analysis(
    lengths: List[int],
    scores: List[float],
    save_path: Path,
    metric_name: str = "BLEU"
):
    """
    Plot performance vs sequence length.
    
    Args:
        lengths: Sequence lengths
        scores: Corresponding scores
        save_path: Path to save plot
        metric_name: Name of metric being plotted
    """
    plt.figure(figsize=(10, 6))
    
    # Bin by length
    length_bins = pd.cut(lengths, bins=10)
    df = pd.DataFrame({'length': lengths, 'score': scores, 'bin': length_bins})
    binned = df.groupby('bin')['score'].mean()
    
    plt.plot(range(len(binned)), binned.values, marker='o')
    plt.xlabel('Sequence Length Bin')
    plt.ylabel(f'{metric_name} Score')
    plt.title(f'{metric_name} vs Sequence Length')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved length analysis to {save_path}")


# ============================================================================
# utils/__init__.py
# ============================================================================
"""Utility functions."""

from .checkpoint import save_checkpoint, load_checkpoint
from .logging_utils import setup_logger
from .visualization import (
    plot_training_curves,
    plot_metric_comparison,
    save_metrics_table,
    plot_length_analysis
)

__all__ = [
    'save_checkpoint',
    'load_checkpoint',
    'setup_logger',
    'plot_training_curves',
    'plot_metric_comparison',
    'save_metrics_table',
    'plot_length_analysis'
]

