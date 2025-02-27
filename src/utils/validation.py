"""
Validation utilities for input checking.
"""

import logging
from typing import Any, Dict, List, Optional, Type
from pydantic import BaseModel, ValidationError

logger = logging.getLogger(__name__)

def validate_input_type(value: Any, expected_type: Type, param_name: str) -> None:
    """Validate input parameter type."""
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Parameter '{param_name}' must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )

def validate_text_length(
    text: str,
    max_length: int,
    param_name: str
) -> None:
    """Validate text length."""
    if len(text) > max_length:
        raise ValueError(
            f"Parameter '{param_name}' exceeds maximum length of {max_length} characters"
        )

def validate_numeric_range(
    value: float,
    min_value: float,
    max_value: float,
    param_name: str
) -> None:
    """Validate numeric value range."""
    if not min_value <= value <= max_value:
        raise ValueError(
            f"Parameter '{param_name}' must be between {min_value} and {max_value}"
        )

class InputValidator:
    """Utility class for input validation."""
    
    @staticmethod
    def validate_search_params(
        query: str,
        k: int,
        threshold: float
    ) -> None:
        """Validate search parameters."""
        if not query.strip():
            raise ValueError("Search query cannot be empty")
        
        if k < 1:
            raise ValueError("Number of results (k) must be positive")
        
        if not 0 <= threshold <= 1:
            raise ValueError("Threshold must be between 0 and 1")
    
    @staticmethod
    def validate_document_metadata(metadata: Dict[str, Any]) -> None:
        """Validate document metadata."""
        required_fields = ['source', 'page']
        for field in required_fields:
            if field not in metadata:
                raise ValueError(f"Required metadata field '{field}' is missing") 