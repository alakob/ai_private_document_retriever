"""
Text processing utilities.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

def sanitize_text(text: str) -> str:
    """Clean text for PostgreSQL storage with validation."""
    if not text or not isinstance(text, str):
        return ""
        
    # Remove null bytes
    text = text.replace('\x00', '')
    
    # Replace problematic characters
    text = ''.join(
        char for char in text 
        if ord(char) >= 32 or char in '\n\r\t'
    )
    
    # Validate content
    text = text.strip()
    if not text:
        return ""
    
    # Limit length if needed
    max_length = 1_000_000  # Adjust based on your PostgreSQL config
    if len(text) > max_length:
        text = text[:max_length]
        logger.warning(f"Text truncated to {max_length} characters")
    
    return text

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length with suffix."""
    if not text:
        return ""
        
    text = str(text).strip()
    if len(text) <= max_length:
        return text
        
    truncated = text[:max_length - len(suffix)]
    return truncated.rstrip() + suffix

def clean_filename(filename: str) -> str:
    """Clean filename by removing problematic characters."""
    # Remove invalid characters
    valid_chars = '-_.() abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
    cleaned = ''.join(c for c in filename if c in valid_chars)
    
    # Remove leading/trailing spaces and dots
    cleaned = cleaned.strip('. ')
    
    # Ensure filename is not empty
    if not cleaned:
        cleaned = 'unnamed_file'
        
    return cleaned 