"""
Custom error types and error handling utilities.
"""

import logging
from typing import Type, Optional, Any
from functools import wraps
import traceback

logger = logging.getLogger(__name__)

class DocumentProcessingError(Exception):
    """Base exception for document processing errors."""
    pass

class FileLoadError(DocumentProcessingError):
    """Raised when a file cannot be loaded."""
    pass

class EmbeddingError(DocumentProcessingError):
    """Raised when embeddings cannot be generated."""
    pass

class DatabaseError(DocumentProcessingError):
    """Raised when database operations fail."""
    pass

class VectorStoreError(Exception):
    """Base exception for vector store operations."""
    pass

class DataAcquisitionError(VectorStoreError):
    """Exception raised for errors during data acquisition."""
    pass

class VisualizationError(VectorStoreError):
    """Exception raised for errors during visualization."""
    pass

def handle_exceptions(
    error_type: Type[Exception],
    default_message: str,
    reraise: bool = True,
    log_level: str = "error"
) -> callable:
    """Decorator for handling exceptions with proper logging."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                log_func = getattr(logger, log_level)
                error_msg = f"{default_message}: {str(e)}"
                log_func(error_msg)
                log_func(f"Traceback:\n{traceback.format_exc()}")
                
                if reraise:
                    raise error_type(error_msg) from e
                return None
        return wrapper
    return decorator 