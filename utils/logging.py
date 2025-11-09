"""Logging utilities for live-audio-classifier."""
import logging
import sys
from pathlib import Path


def setup_logging(level=logging.INFO, log_file=None):
    """
    Set up logging configuration for the application.
    
    Args:
        level: Logging level (default: INFO)
        log_file: Optional path to log file. If None, logs only to console.
    
    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger("live_audio_classifier")
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_path)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_logger(name=None):
    """
    Get a logger instance.
    
    Args:
        name: Logger name. If None, returns the default 'live_audio_classifier' logger.
    
    Returns:
        Logger instance.
    """
    if name:
        return logging.getLogger(f"live_audio_classifier.{name}")
    return logging.getLogger("live_audio_classifier")

