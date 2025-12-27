"""
Centralized logging configuration for STA410 Stock Prediction System.

This module provides structured logging with proper levels (DEBUG, INFO, WARNING, ERROR, CRITICAL),
formatted output, and file-based persistence for production monitoring.
"""

import logging
import logging.handlers
import os
from datetime import datetime
from pathlib import Path


def setup_logger(name: str, log_level=None) -> logging.Logger:
    """
    Set up a logger with both console and file handlers.
    
    Args:
        name: Logger name (typically __name__)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
                  If None, uses LOG_LEVEL environment variable or INFO default.
    
    Returns:
        Configured logger instance
    
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info("Application started")
        >>> logger.error("Error occurred", exc_info=True)
    """
    # Determine log level from environment or parameter
    if log_level is None:
        log_level_str = os.getenv("LOG_LEVEL", "INFO").upper()
        log_level = getattr(logging, log_level_str, logging.INFO)
    elif isinstance(log_level, str):
        log_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logger = logging.getLogger(name)
    
    # Avoid duplicate handlers if logger already configured
    if logger.hasHandlers():
        return logger
    
    logger.setLevel(log_level)
    
    # Create logs directory if it doesn't exist
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    
    # Standard formatter with timestamp, level, module, and message
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler (for immediate feedback)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (for persistence and debugging)
    log_file = log_dir / f"sta410_{datetime.now().strftime('%Y%m%d')}.log"
    file_handler = logging.handlers.RotatingFileHandler(
        log_file,
        maxBytes=10 * 1024 * 1024,  # 10 MB
        backupCount=10,  # Keep 10 backup files
        encoding='utf-8'
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Separate error log file
    error_log_file = log_dir / f"sta410_errors_{datetime.now().strftime('%Y%m%d')}.log"
    error_handler = logging.handlers.RotatingFileHandler(
        error_log_file,
        maxBytes=10 * 1024 * 1024,
        backupCount=10,
        encoding='utf-8'
    )
    error_handler.setLevel(logging.ERROR)
    error_handler.setFormatter(formatter)
    logger.addHandler(error_handler)
    
    return logger


def log_function_call(logger: logging.Logger, func_name: str, args: dict = None, result: str = ""):
    """
    Log function entry/exit for debugging.
    
    Args:
        logger: Logger instance
        func_name: Function name being logged
        args: Dictionary of arguments
        result: Result message (optional)
    """
    if args:
        logger.debug(f"Entering {func_name} with args: {args}")
    else:
        logger.debug(f"Entering {func_name}")
    
    if result:
        logger.debug(f"Exiting {func_name}: {result}")


def log_execution_time(logger: logging.Logger, operation: str, duration: float, threshold: float = 1.0):
    """
    Log execution time with warning if exceeds threshold.
    
    Args:
        logger: Logger instance
        operation: Operation description
        duration: Execution time in seconds
        threshold: Warning threshold in seconds (default: 1.0 second)
    """
    if duration > threshold:
        logger.warning(f"{operation} took {duration:.2f}s (exceeds threshold of {threshold}s)")
    else:
        logger.debug(f"{operation} completed in {duration:.2f}s")
