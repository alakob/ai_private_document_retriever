"""
Rate limiting utilities for API calls.
"""

import asyncio
import logging
import random
import time
from typing import Optional
from ..config.rate_limiter import RateLimitConfig

logger = logging.getLogger(__name__)

class RateLimiter:
    """Enhanced rate limiter with token bucket and concurrent request limiting."""
    
    def __init__(self, config: RateLimitConfig):
        self.config = config
        self.requests = []
        self.concurrent_requests = 0
        self.lock = asyncio.Lock()
        self.last_request_time = time.time()
    
    async def wait_if_needed(self):
        """Implement enhanced rate limiting with backoff."""
        async with self.lock:
            now = time.time()
            
            # Remove requests older than 1 minute
            self.requests = [t for t in self.requests if now - t < 60]
            
            # Check rate limit
            if len(self.requests) >= self.config.max_requests_per_minute:
                wait_time = 60 - (now - self.requests[0])
                if wait_time > 0:
                    logger.info(f"Rate limit reached, waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)
            
            # Check concurrent requests
            while self.concurrent_requests >= self.config.max_concurrent_requests:
                await asyncio.sleep(self.config.min_delay)
            
            # Add jitter to prevent thundering herd
            jitter = random.uniform(0, self.config.jitter)
            
            # Ensure minimum delay between requests
            time_since_last = now - self.last_request_time
            if time_since_last < self.config.min_delay:
                await asyncio.sleep(self.config.min_delay - time_since_last + jitter)
            
            self.requests.append(now)
            self.concurrent_requests += 1
            self.last_request_time = time.time()
    
    async def release(self):
        """Release a concurrent request slot."""
        async with self.lock:
            self.concurrent_requests = max(0, self.concurrent_requests - 1)

    async def backoff_delay(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        delay = min(
            self.config.max_delay,
            self.config.initial_delay * (self.config.backoff_factor ** attempt)
        )
        return delay + random.uniform(0, self.config.jitter) 