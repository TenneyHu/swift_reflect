from typing import Optional
from diskcache import Cache

from .typedefs import SingleRequestModel, BatchRequestModel
from .batching import AsyncBatcher
from .deduplicator import AsyncDeduplicator
from .caching import AsyncCacher
from .reordering import RequestReorderer
from .resources import AsyncResource


class LocalAPI(AsyncResource, SingleRequestModel, BatchRequestModel):
    """
    Pipeline optimized for local model inference.

    Flow: Cache -> Batcher -> Model
    Optimizes for full batch utilization while avoiding unnecessary model calls.
    """

    def __init__(
        self,
        model: BatchRequestModel,
        cache: Cache,
        batch_size: int,
        timeout: int = 30,
        allow_batch_overflow: bool = False
    ):
        # Create pipeline from inside out
        self.batcher = AsyncBatcher(
            model=model,
            batch_size=batch_size,
            timeout=timeout,
            name="local_batcher",
            allow_batch_overflow=allow_batch_overflow
        )
        self.cacher = AsyncCacher(model=self.batcher, cache=cache)

    async def request(self, request):
        """Single request API, delegates to cache."""
        return await self.cacher.request(request)

    async def batch_request(self, batch):
        """Batch request API, delegates to cache."""
        return await self.cacher.batch_request(batch)

    async def cleanup(self):
        """Cleanup all pipeline components."""
        await self.batcher.cleanup()
        if hasattr(self.cacher, 'cleanup'):
            await self.cacher.cleanup()


class OnlineAPI(AsyncResource, SingleRequestModel, BatchRequestModel):
    """
    Pipeline optimized for cloud API cost reduction.

    Flow: Batcher -> Reorderer -> Deduplicator -> Cache -> Model
    """

    def __init__(
        self,
        model: BatchRequestModel,
        cache: Cache,
        batch_size: int,
        timeout: int = 30,
        allow_batch_overflow: bool = False,
        correctness: bool = False # Happy to remove this if not needed
    ):
        # Create pipeline from inside out
        self.cached_model = AsyncCacher(model=model, cache=cache)
        self.deduplicator = AsyncDeduplicator(model=self.cached_model, correctness=correctness)
        self.reorderer = RequestReorderer(model=self.deduplicator)
        self.batcher = AsyncBatcher(
            model=self.reorderer,
            batch_size=batch_size,
            timeout=timeout,
            name="online_batcher",
            allow_batch_overflow=allow_batch_overflow
        )

    async def request(self, request):
        """Single request API, delegates to batcher."""
        return await self.batcher.request(request)

    async def batch_request(self, batch):
        """Batch request API, delegates to batcher."""
        return await self.batcher.batch_request(batch)

    async def cleanup(self):
        """Cleanup all pipeline components."""
        await self.batcher.cleanup()
        if hasattr(self.cached_model, 'cleanup'):
            await self.cached_model.cleanup()


class OrderedLocalAPI(AsyncResource, SingleRequestModel, BatchRequestModel):
    """
    Pipeline optimized for local model inference with guaranteed ordering.

    Flow: Collect Batch -> Reorder -> Cache -> Hardware Batch -> Model

    This pipeline ensures:
    1. Responses are ordered by request_id (reproducible results)
    2. Cache is checked for existing responses
    3. Remaining requests are batched for efficient hardware utilization
    """

    def __init__(
        self,
        model: BatchRequestModel,
        cache: Cache,
        collection_batch_size: int,
        hardware_batch_size: int,
        timeout: int = 30,
        allow_batch_overflow: bool = False
    ):
        # Create pipeline from inside out (model -> user)
        self.hardware_batcher = AsyncBatcher(
            model=model,
            batch_size=hardware_batch_size,
            timeout=timeout,
            name="hardware_batcher"
        )
        self.cacher = AsyncCacher(
            model=self.hardware_batcher,
            cache=cache
        )
        self.deduplicator = AsyncDeduplicator(
            model=self.cacher
        )
        self.reorderer = RequestReorderer(
            model=self.deduplicator
        )
        self.collection_batcher = AsyncBatcher(
            model=self.reorderer,
            batch_size=collection_batch_size,
            timeout=timeout,
            name="collection_batcher",
            allow_batch_overflow=allow_batch_overflow
        )

    async def request(self, request):
        """Single request API, delegates to collection batcher."""
        return await self.collection_batcher.request(request)

    async def batch_request(self, batch):
        """Batch request API, delegates to collection batcher."""
        return await self.collection_batcher.batch_request(batch)

    async def cleanup(self):
        """Cleanup all pipeline components."""
        await self.collection_batcher.cleanup()
        await self.hardware_batcher.cleanup()
        if hasattr(self.cacher, 'cleanup'):
            await self.cacher.cleanup()
