"""
Rate limiter configuration settings.
"""

from dataclasses import dataclass

@dataclass
class RateLimitConfig:
    """Enhanced configuration for rate limiting."""
    max_requests_per_minute: int = 150
    max_concurrent_requests: int = 20
    batch_size: int = 50
    min_delay: float = 0.1
    max_delay: float = 30.0
    initial_delay: float = 1.0
    backoff_factor: float = 2.0
    jitter: float = 0.1
    max_retries: int = 3
    retry_delay: float = 5.0 