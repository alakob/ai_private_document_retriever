"""
Resource monitoring configuration settings.
"""

from dataclasses import dataclass

@dataclass
class ResourceConfig:
    """Configuration for resource monitoring."""
    memory_threshold_mb: int = 1000
    cpu_threshold_percent: float = 80.0
    min_free_memory_mb: int = 500 