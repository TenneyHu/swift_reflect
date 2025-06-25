import asyncio
import logging
from typing import List, Any, Dict
from diskcache import Cache
from .resources import AsyncResource

from .typedefs import Request, Batch, Response, SingleRequestModel, BatchRequestModel

logger = logging.getLogger(__name__)


class AsyncCacher(AsyncResource, SingleRequestModel, BatchRequestModel):
    """
    Caches responses from a model, with namespace-aware caching.

    While this class implements both SingleRequestModel and BatchRequestModel protocols,
    it requires a SingleRequestModel as input and handles batches by splitting them
    into individual requests.

    Within a namespace, each request gets fresh responses.
    Across namespaces, responses can be reused.
    """

    def __init__(self, model: SingleRequestModel, cache: Cache):
        assert cache is not None, "cache must be provided"
        assert model is not None, "model must be provided"
        assert isinstance(
            model, SingleRequestModel), "model must implement SingleRequestModel"

        self.model = model
        self.cache = cache
        self.key2mutex: Dict[str, asyncio.Lock] = {}
        self.namespace2used_counts: Dict[str, Dict[str, int]] = {}
        self.used_counts: Dict[str, int] = {}

    async def request(self, request: Request) -> Response:
        """Handle a single request, potentially reusing cached responses."""
        key = request.hash()

        # Get or create mutex for this prompt
        mutex = self.key2mutex.get(key, asyncio.Lock())
        self.key2mutex[key] = mutex

        async with mutex:
            entries_in_cache = self.cache.get(key, [])

            # Track usage counts per namespace
            if request.namespace is not None:
                used_counts = self.namespace2used_counts.get(
                    request.namespace, {})
                self.namespace2used_counts[request.namespace] = used_counts
            else:
                used_counts = self.used_counts

            used = used_counts.get(key, 0)

            # Check if we need more responses
            num_needed = max(request.n - len(entries_in_cache[used:]), 0)
            if num_needed > 0:
                # Get fresh responses from model
                response = await self.model.request(request)
                new_entries = response.data
                entries_in_cache.extend(new_entries)
                self.cache.set(key, entries_in_cache)

            # Update usage count
            used_counts[key] = used + request.n

            response = Response(
                data=entries_in_cache[used:used + request.n],
                cached=[True] * (request.n-num_needed) + [False] * num_needed,
                duplicated=[False] * request.n,
            )
            return response

    async def batch_request(self, batch: Batch) -> List[Response]:
        """Handle a batch of requests by processing each individually."""
        coroutines = [self.request(request) for request in batch.requests]
        return await asyncio.gather(*coroutines)

    async def cleanup(self):
        """Cleanup cache resources."""
        self.cache.close()
