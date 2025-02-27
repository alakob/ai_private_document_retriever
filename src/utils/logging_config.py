"""
Centralized logging configuration for the application.
Provides consistent logging setup with proper file paths and rotation.
"""

import os
import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

# Get project root directory (2 levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent.absolute()
LOGS_DIR = os.path.join(PROJECT_ROOT, "logs")

# Create logs directory if it doesn't exist
os.makedirs(LOGS_DIR, exist_ok=True)

def setup_logger(module_name: str) -> logging.Logger:
    """
    Configure a logger with console and rotating file handlers.
    
    Args:
        module_name: Name of the module (used for the logger name and log file)
        
    Returns:
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(module_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers to avoid duplicates when reconfiguring
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Configure rotating file handler
    log_file = os.path.join(LOGS_DIR, f"{module_name.replace('.', '_')}.log")
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10*1024*1024,  # 10MB
        backupCount=5
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    
    # Configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    logger.info(f"Logger initialized. Log file: {log_file}")
    
    return logger 