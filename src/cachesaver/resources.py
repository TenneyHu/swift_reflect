import logging
from abc import ABC, abstractmethod
from typing import TypeVar, AsyncContextManager

logger = logging.getLogger(__name__)

T = TypeVar('T', bound='AsyncResource')


class AsyncResource(AsyncContextManager[T], ABC):
    """Base class for resources that need async cleanup."""

    async def __aenter__(self: T) -> T:
        """Enter async context."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit async context and cleanup resources."""
        await self.cleanup()

    @abstractmethod
    async def cleanup(self):
        """Cleanup resources. Must be implemented by subclasses."""
        pass
