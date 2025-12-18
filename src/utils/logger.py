"""
Simple logging setup for the application.
Makes it easier to track what's happening when things go wrong.
"""

import logging
import sys
from pathlib import Path
from datetime import datetime
from src.utils.config import Config


def setup_logger(name: str = __name__) -> logging.Logger:
    """
    Create a logger that writes to both console and file.
    
    Args:
        name: Logger name (usually __name__ from the calling module)
    
    Returns:
        Configured logger instance
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, Config.LOG_LEVEL))
    
    # Don't add handlers if they already exist (prevents duplicates)
    if logger.handlers:
        return logger
    
    # Console handler - prints to terminal
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    logger.addHandler(console_handler)
    
    # File handler - saves to log file
    if not Path('logs').exists():
        Path('logs').mkdir(parents=True)
    
    log_file = Path('logs') / f'app_{datetime.now().strftime("%Y%m%d")}.log'
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    file_handler.setFormatter(file_format)
    logger.addHandler(file_handler)
    
    return logger
