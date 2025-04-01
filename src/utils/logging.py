"""Logging utilities for the BigQuery Cost Intelligence Engine."""

import logging
import os
import sys
from datetime import datetime


def setup_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Set up a logger with the given name and level.
    
    Args:
        name: Logger name, typically __name__ from the calling module
        level: Logging level, defaults to INFO
        
    Returns:
        Configured logger instance
    """
    # Get or create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # If the logger already has handlers, return it (to avoid duplicate handlers)
    if logger.handlers:
        return logger
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    
    # Create formatter
    formatter = logging.Formatter(
        "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(console_handler)
    
    # Create file handler if LOG_FILE environment variable is set
    log_file = os.environ.get("LOG_FILE")
    if log_file:
        # Create directory if it doesn't exist
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
            
        # Add timestamp to filename to prevent overwriting
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename, ext = os.path.splitext(log_file)
        timestamped_log_file = f"{filename}_{timestamp}{ext}"
        
        file_handler = logging.FileHandler(timestamped_log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger
