"""
Async testing utilities.
"""

import asyncio
import logging
from typing import Any, Callable, Coroutine, TypeVar
from functools import wraps
import pytest
from unittest.mock import AsyncMock, patch

logger = logging.getLogger(__name__)

T = TypeVar('T')

def async_test(func: Callable[..., Coroutine[Any, Any, T]]) -> Callable[..., T]:
    """Decorator to run async test functions."""
    @wraps(func)
    def wrapper(*args, **kwargs) -> T:
        return asyncio.run(func(*args, **kwargs))
    return wrapper

class AsyncTestContext:
    """Context manager for async tests."""
    
    def __init__(self):
        self._mocks = {}
        self.loop = None
    
    async def __aenter__(self):
        """Set up async test context."""
        self.loop = asyncio.get_event_loop()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Clean up async test context."""
        for mock in self._mocks.values():
            mock.reset_mock()
    
    def mock_coroutine(self, target: str) -> AsyncMock:
        """Create and register a mock coroutine."""
        mock = AsyncMock()
        self._mocks[target] = mock
        return mock

class AsyncTestCase:
    """Base class for async test cases."""
    
    @classmethod
    def setup_class(cls):
        """Set up test class."""
        cls.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(cls.loop)
    
    @classmethod
    def teardown_class(cls):
        """Clean up test class."""
        cls.loop.close()
    
    def setup_method(self):
        """Set up test method."""
        self.mocks = {}
    
    def teardown_method(self):
        """Clean up test method."""
        for mock in self.mocks.values():
            mock.reset_mock()
    
    async def run_coroutine(self, coroutine: Coroutine) -> Any:
        """Run a coroutine in the test loop."""
        return await self.loop.run_until_complete(coroutine)
    
    def mock_coroutine(self, target: str) -> AsyncMock:
        """Create and register a mock coroutine."""
        mock = AsyncMock()
        self.mocks[target] = mock
        return mock 