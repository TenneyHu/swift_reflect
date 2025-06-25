import pytest
import tempfile
from typing import List, Any
from diskcache import Cache
import asyncio

from src.cachesaver.typedefs import Request, Response, Batch, BatchRequestModel
from src.cachesaver.pipelines import LocalAPI


class BatchTrackingModel(BatchRequestModel):
    """Mock model that tracks batch sizes and returns deterministic responses."""

    def __init__(self):
        self.batch_calls: List[Batch] = []
        self.counter = 0

    async def batch_request(self, batch: Batch) -> List[Response]:
        self.batch_calls.append(batch)
        responses = []
        for request in batch.requests:
            request_responses = []
            for _ in range(request.n):
                response = f"{request.prompt}_{request.namespace}_{self.counter}"
                self.counter += 1
                request_responses.append(response)
            responses.append(Response(data=request_responses))
        return responses


class TestLocalAPI:
    TEST_TIMEOUT = 0.1  # Short timeout for tests

    @pytest.fixture
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio
    async def test_batch_size_optimization(self, cache):
        """Test that requests are properly batched before reaching model."""
        model = BatchTrackingModel()
        batch_size = 2

        async with LocalAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Create 3 requests (should result in two batches: size 2 and 1)
            requests = [
                Request(prompt=f"test{i}", n=1,
                        request_id=str(i), namespace="ns1")
                for i in range(3)
            ]

            # Send requests concurrently
            tasks = [api.request(req) for req in requests]
            await asyncio.gather(*tasks)

            # Verify batching
            assert len(model.batch_calls) == 2
            assert len(model.batch_calls[0].requests) == batch_size
            assert len(model.batch_calls[1].requests) == 1

    @pytest.mark.asyncio
    async def test_cache_reuse_across_namespaces(self, cache):
        """Test that responses are reused across namespaces while maintaining batch size."""
        model = BatchTrackingModel()
        batch_size = 2

        async with LocalAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # First namespace requests
            ns1_requests = [
                Request(prompt="test1", n=1,
                        request_id="1", namespace="ns1"),
                Request(prompt="test2", n=1,
                        request_id="2", namespace="ns1")
            ]
            ns1_results = await asyncio.gather(*[api.request(req) for req in ns1_requests])

            # Same prompts, different namespace
            ns2_requests = [
                Request(prompt="test1", n=1,
                        request_id="3", namespace="ns2"),
                Request(prompt="test2", n=1,
                        request_id="4", namespace="ns2")
            ]
            ns2_results = await asyncio.gather(*[api.request(req) for req in ns2_requests])

            # Verify cache reuse
            assert [r.data for r in ns1_results] == [r.data for r in ns2_results]  # Same responses reused
            assert len(model.batch_calls) == 1  # Only one batch call to model

    @pytest.mark.asyncio
    async def test_mixed_cache_and_fresh_requests(self, cache):
        """Test handling mix of cached and fresh requests while maintaining batch size."""
        model = BatchTrackingModel()
        batch_size = 2

        async with LocalAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Initial request to populate cache
            request1 = Request(prompt="cached", n=1,
                               request_id="1", namespace="ns1")
            await api.request(request1)

            # Mix of cached and fresh requests
            mixed_requests = [
                Request(prompt="cached", n=1, request_id="2",
                        namespace="ns2"),  # Should use cache
                Request(prompt="fresh1", n=1, request_id="3",
                        namespace="ns1"),  # New request
                Request(prompt="fresh2", n=1, request_id="4",
                        namespace="ns1")   # New request
            ]

            results = await asyncio.gather(*[api.request(req) for req in mixed_requests])

            # Verify batching of fresh requests
            assert len(model.batch_calls) == 2
            assert len(model.batch_calls[1].requests) == 2
            assert all(req.prompt.startswith("fresh")
                       for req in model.batch_calls[1].requests)

    @pytest.mark.asyncio
    async def test_batch_request_interface(self, cache):
        """Test the batch_request interface directly."""
        model = BatchTrackingModel()
        batch_size = 2

        async with LocalAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=self.TEST_TIMEOUT
        ) as api:
            batch = Batch(requests=[
                Request(prompt="test1", n=1,
                        request_id="1", namespace="ns1"),
                Request(prompt="test2", n=1,
                        request_id="2", namespace="ns1"),
                Request(prompt="test3", n=1,
                        request_id="3", namespace="ns1")
            ])

            results = await api.batch_request(batch)

            # Verify results structure
            assert len(results) == 3
            assert all(isinstance(r, Response) for r in results)

            # Verify batching
            assert len(model.batch_calls) == 2
            assert len(model.batch_calls[0].requests) == batch_size
            assert len(model.batch_calls[1].requests) == 1


class TestLocalAPIBatching:
    TEST_TIMEOUT = 0.1  # Short timeout for tests

    @pytest.fixture
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio
    async def test_allow_batch_overflow_single_large_request(self, cache):
        """Test overflow with a single request larger than batch size."""
        model = BatchTrackingModel()
        batch_size = 5

        async with LocalAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=self.TEST_TIMEOUT,
            allow_batch_overflow=True
        ) as api:
            # Single request larger than batch size
            request1 = Request(prompt="large", n=7, request_id="r1", namespace="test")
            result1 = await api.request(request1)

            # Verify: Should process as one batch despite size
            assert len(result1.data) == 7
            assert len(model.batch_calls) == 1
            assert len(model.batch_calls[0].requests) == 7
    
    @pytest.mark.asyncio
    async def test_allow_batch_overflow_fills_existing(self, cache):
        """Test overflow where a request fills and exceeds existing items."""
        model = BatchTrackingModel()
        batch_size = 5

        async with LocalAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=self.TEST_TIMEOUT,
            allow_batch_overflow=True
        ) as batcher:
            # Send 3 initial requests (batch size 5)
            initial_requests = [
                Request(prompt=f"init{i}", n=1, request_id=f"i{i}", namespace="test")
                for i in range(3)
            ]
            replies = [batcher.request(req) for req in initial_requests]

            # Create extra request that leads to overflow
            overflow_request = Request(prompt="overflow", n=7, request_id="of1", namespace="test")
            overflow_reply = batcher.request(overflow_request)

            # Await the results
            replies.append(overflow_reply)
            all_results = await asyncio.gather(*replies)

            # Extract results for verification
            initial_results = all_results[:3]
            overflow_result = all_results[3]

            # Verify: Should still be one batch with all 10 requests
            assert len([r.data[0] for r in initial_results]) == 3
            assert len(overflow_result.data) == 7
            assert len(model.batch_calls) == 1
            assert len(model.batch_calls[0].requests) == 10 # 3 initial + 7 overflow
