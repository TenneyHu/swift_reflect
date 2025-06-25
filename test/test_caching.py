import pytest
import tempfile
from typing import List, Any
from diskcache import Cache
import asyncio
from src.cachesaver.typedefs import Request, Response, Batch, SingleRequestModel
from src.cachesaver.caching import AsyncCacher


class CountingMockModel(SingleRequestModel):
    """Mock model that returns incrementing responses to track call order."""

    def __init__(self):
        self.counter = 0
        self.calls = []

    async def request(self, request: Request) -> List[Any]:
        self.calls.append(request)
        responses = []
        for _ in range(request.n):
            responses.append(f"response_{self.counter}")
            self.counter += 1
        return Response(data=responses)


class TestAsyncCacher:
    @pytest.fixture
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio
    async def test_basic_caching(self, cache):
        """Test that responses are cached and reused."""
        model = CountingMockModel()
        cacher = AsyncCacher(model=model, cache=cache)

        # First request should hit the model
        request = Request(prompt="test", n=2,
                          request_id="1", namespace="ns1")
        result1 = await cacher.request(request)
        assert result1.data == ["response_0", "response_1"]
        assert len(model.calls) == 1

        # Same request in different namespace should reuse cache
        request = Request(prompt="test", n=2,
                          request_id="2", namespace="ns2")
        result2 = await cacher.request(request)
        assert result2.data == ["response_0", "response_1"]
        assert len(model.calls) == 1  # No new model calls

    @pytest.mark.asyncio
    async def test_namespace_isolation(self, cache):
        """Test that requests within same namespace get fresh responses."""
        model = CountingMockModel()
        cacher = AsyncCacher(model=model, cache=cache)

        # First request
        request1 = Request(prompt="test", n=1,
                           request_id="1", namespace="same_ns")
        result1 = await cacher.request(request1)
        assert result1.data == ["response_0"]

        # Same prompt, same namespace should get fresh response
        request2 = Request(prompt="test", n=1,
                           request_id="2", namespace="same_ns")
        result2 = await cacher.request(request2)
        assert result2.data == ["response_1"]  # New response
        assert len(model.calls) == 2  # Required new model call

    @pytest.mark.asyncio
    async def test_batch_request_caching(self, cache):
        """Test that batch requests properly use cache."""
        model = CountingMockModel()
        cacher = AsyncCacher(model=model, cache=cache)

        batch = Batch(requests=[
            Request(prompt="p1", n=2,
                    request_id="1", namespace="ns1"),
            Request(prompt="p1", n=1, request_id="2",
                    namespace="ns2"),  # Should reuse
            Request(prompt="p2", n=1,
                    request_id="3", namespace="ns1")
        ])

        results = await cacher.batch_request(batch)
        assert len(results) == 3
        assert results[0].data == ["response_0", "response_1"]  # First p1 request
        assert results[1].data == ["response_0"]  # Reused from cache
        assert results[2].data == ["response_2"]  # New p2 request

        assert len(model.calls) == 2  # Only needed calls for p1/ns1 and p2/ns1

    @pytest.mark.asyncio
    async def test_partial_cache_reuse(self, cache):
        """Test requesting more responses than cached."""
        model = CountingMockModel()
        cacher = AsyncCacher(model=model, cache=cache)

        # Initial request caches 2 responses
        request1 = Request(prompt="test", n=2,
                           request_id="1", namespace="ns1")
        result1 = await cacher.request(request1)
        assert result1.data == ["response_0", "response_1"]

        # Request more responses in different namespace
        request2 = Request(prompt="test", n=3,
                           request_id="2", namespace="ns2")
        result2 = await cacher.request(request2)
        # Should reuse first 2 responses and get 1 new
        assert result2.data == ["response_0", "response_1", "response_2"]
        assert len(model.calls) == 2

    @pytest.mark.asyncio
    async def test_concurrent_access(self, cache):
        """Test that concurrent requests are properly synchronized."""
        model = CountingMockModel()
        cacher = AsyncCacher(model=model, cache=cache)

        # Create multiple concurrent requests for same prompt
        requests = [
            Request(prompt="test", n=1,
                    request_id=str(i), namespace=f"ns{i}")
            for i in range(3)
        ]

        results = await asyncio.gather(*(cacher.request(r) for r in requests))

        # First response should be used for all requests
        assert all(r.data == ["response_0"] for r in results)
        assert len(model.calls) == 1  # Should only call model once
