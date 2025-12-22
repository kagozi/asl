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