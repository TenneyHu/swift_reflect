import pytest
import logging
import tempfile
import asyncio
from typing import List, Any
from collections import defaultdict
from diskcache import Cache

from src.cachesaver.typedefs import Request, Response, Batch, BatchRequestModel
from src.cachesaver.pipelines import OrderedLocalAPI


class BatchCounterModel(BatchRequestModel):
    """Mock model that maintains per-prompt counters and tracks batch sizes."""

    def __init__(self):
        self.batch_calls: List[Batch] = []
        self.prompt_counters = defaultdict(int)

    async def batch_request(self, batch: Batch) -> List[List[Any]]:
        self.batch_calls.append(batch)
        responses = []
        for request in batch.requests:
            request_responses = []
            for _ in range(request.n):
                counter = self.prompt_counters[request.prompt]
                request_responses.append(f"{request.prompt}_count{counter}")
                self.prompt_counters[request.prompt] += 1
            responses.append(Response(data=request_responses))
        return responses


class TestOrderedLocalAPI:
    TEST_TIMEOUT = 0.1

    @pytest.fixture
    def cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.fixture(autouse=True)
    def setup_logging(self):
        """Configure logging for all tests in this module."""
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    @pytest.mark.asyncio
    async def test_two_stage_batching(self, cache):
        """Test that requests go through both collection and hardware batching."""
        model = BatchCounterModel()

        async with OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=4,
            hardware_batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            requests = [
                Request(prompt=f"p{i}", n=1,
                        request_id=str(i), namespace="ns1")
                for i in range(4)
            ]

            await asyncio.gather(*[api.request(req) for req in requests])

            # Should see 2 hardware batches (size 2 each)
            assert len(model.batch_calls) == 2
            assert len(model.batch_calls[0].requests) == 2
            assert len(model.batch_calls[1].requests) == 2

    @pytest.mark.asyncio
    async def test_ordering_with_cache(self, cache):
        """Test that responses maintain order even with cache hits."""
        model = BatchCounterModel()

        async with OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=3,
            hardware_batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # First request to populate cache
            await api.request(
                Request(prompt="cached", n=1,
                        request_id="2", namespace="ns1")
            )

            # Mix of cached and new requests with out-of-order IDs
            requests = [
                Request(prompt="new", n=1,
                        request_id="3", namespace="ns1"),
                Request(prompt="cached", n=1,
                        request_id="1", namespace="ns2"),
                Request(prompt="new", n=1,
                        request_id="2", namespace="ns1")
            ]

            results = await asyncio.gather(*[api.request(req) for req in requests])

            # Results should be ordered by request_id regardless of cache status
            assert results[0].data[0].startswith("new_count")      # id "3"
            assert results[1].data[0].startswith(
                "cached_count")   # id "1" (from cache)
            assert results[2].data[0].startswith("new_count")      # id "2"

    @pytest.mark.asyncio
    async def test_ordering_within_hardware_batches(self, cache):
        """Test that ordering is maintained within hardware batches."""
        model = BatchCounterModel()

        async with OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=4,
            hardware_batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            batch = Batch(requests=[
                Request(prompt="p1", n=1,
                        request_id="4", namespace="ns1"),
                Request(prompt="p1", n=1,
                        request_id="2", namespace="ns1"),
                Request(prompt="p1", n=1,
                        request_id="3", namespace="ns1"),
                Request(prompt="p1", n=1,
                        request_id="1", namespace="ns1")
            ])

            results = await api.batch_request(batch)

            # Verify responses are ordered by request_id
            assert results[0].data[0] == "p1_count3"  # id "4"
            assert results[1].data[0] == "p1_count1"  # id "2"
            assert results[2].data[0] == "p1_count2"  # id "3"
            assert results[3].data[0] == "p1_count0"  # id "1"

            # Verify hardware batching
            assert len(model.batch_calls) == 2
            assert len(model.batch_calls[0].requests) == 2
            assert len(model.batch_calls[1].requests) == 2

    @pytest.mark.asyncio
    async def test_multiple_samples_ordering(self, cache):
        """Test ordering with requests for multiple samples."""
        model = BatchCounterModel()

        async with OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=3,
            hardware_batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            requests = [
                Request(prompt="p1", n=2,
                        request_id="2", namespace="ns1"),
                Request(prompt="p2", n=1,
                        request_id="1", namespace="ns1")
            ]

            results = await asyncio.gather(*[api.request(req) for req in requests])

            # Verify ordering and sample counts
            assert len(results[0].data) == 2  # n=2
            assert len(results[1].data) == 1  # n=1
            assert results[1].data[0].startswith("p2_count")  # id "1"
            assert all(r.startswith("p1_count") for r in results[0].data)  # id "2"


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
        model = BatchCounterModel()
        batch_size = 5

        async with OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=batch_size,
            hardware_batch_size=10, # Assuming hardware fits all requests
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
        model = BatchCounterModel()
        batch_size = 5

        async with OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=batch_size,
            hardware_batch_size=10, # Assuming hardware fits all requests
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
