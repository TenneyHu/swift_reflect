import pytest
import asyncio
import tempfile
from collections import defaultdict
from typing import List, Any
from diskcache import Cache

from src.cachesaver.typedefs import Request, Response, Batch, SingleRequestModel
from src.cachesaver.pipelines import OnlineAPI


class CountingMockModel(SingleRequestModel):
    """Mock model that tracks individual requests and returns incrementing responses."""

    def __init__(self):
        self.counter = 0
        self.calls: List[Request] = []  # Track individual requests

    async def request(self, request: Request) -> List[Any]:
        self.calls.append(request)
        responses = []
        for _ in range(request.n):
            responses.append(f"{request.prompt}_{self.counter}")
            self.counter += 1
        return Response(data=responses)


class OrderTrackingModel(SingleRequestModel):
    """Mock model that tracks request order and maintains per-prompt counters."""

    def __init__(self):
        self.prompt_counters = defaultdict(int)
        self.calls: List[Request] = []

    async def request(self, request: Request) -> List[Any]:
        self.calls.append(request)
        responses = []
        for _ in range(request.n):
            counter = self.prompt_counters[request.prompt]
            responses.append(f"{request.prompt}_count{counter}")
            self.prompt_counters[request.prompt] += 1
        return Response(data=responses)


class TestOnlineAPI:
    TEST_TIMEOUT = 0.1  # Short timeout for tests

    @pytest.fixture
    def cache(self):
        """Provide a temporary cache for each test."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio
    async def test_reuse_across_namespaces(self, cache):
        """Test that identical prompts are deduplicated and cached."""
        model = CountingMockModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=3,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Create requests with identical prompts, but different namespaces
            requests = [
                Request(prompt="test_prompt", n=1,
                        request_id="1", namespace="ns1"),
                Request(prompt="test_prompt", n=1,
                        request_id="2", namespace="ns2")
            ]

            # First batch of requests
            results1 = await asyncio.gather(*[api.request(req) for req in requests])

            # Same prompt in new namespace
            request3 = Request(prompt="test_prompt", n=1,
                               request_id="3", namespace="ns3")
            result3 = await api.request(request3)

            # Verify deduplication and caching
            assert len(model.calls) == 1  # Single request to model
            assert model.calls[0].n == 1  # One sample reused

    @pytest.mark.asyncio
    async def test_deduplication_within_namespace(self, cache):
        """Test that requests in same namespace get fresh responses."""
        model = CountingMockModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Two identical requests in same namespace
            requests = [
                Request(prompt="test", n=1,
                        request_id="1", namespace="same_ns"),
                Request(prompt="test", n=1,
                        request_id="2", namespace="same_ns")
            ]

            results = await asyncio.gather(*[api.request(req) for req in requests])

            # Verify separate responses in same namespace
            assert len(model.calls) == 1  # One request to model
            assert results[0] != results[1]  # Different responses
            assert model.calls[0].namespace == "same_ns"

    @pytest.mark.asyncio
    async def test_batching_behavior(self, cache):
        """Test that requests are collected into batches before processing."""
        model = CountingMockModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Three unique requests
            requests = [
                Request(prompt=f"unique_{i}", n=1,
                        request_id=str(i), namespace="ns1")
                for i in range(3)
            ]

            await asyncio.gather(*[api.request(req) for req in requests])

            # Verify batching before deduplication
            assert len(model.calls) == 3  # One call per unique prompt
            # First two should be processed together
            assert model.calls[0].prompt.startswith("unique_0")
            assert model.calls[1].prompt.startswith("unique_1")
            # Last one in separate batch
            assert model.calls[2].prompt.startswith("unique_2")

    @pytest.mark.asyncio
    async def test_mixed_scenario(self, cache):
        """Test combination of batching, deduplication, and caching."""
        model = CountingMockModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=3,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Mix of duplicate and unique requests
            requests = [
                Request(prompt="shared", n=1,
                        request_id="1", namespace="ns1"),
                Request(prompt="shared", n=1,
                        request_id="2", namespace="ns1"),  # Should be deduplicated
                Request(prompt="unique", n=1,
                        request_id="3", namespace="ns2")
            ]

            results1 = await asyncio.gather(*[api.request(req) for req in requests])

            # Additional request that should use cache
            request4 = Request(prompt="shared", n=1,
                               request_id="4", namespace="ns3")
            result4 = await api.request(request4)

            # Verify optimizations
            # One for deduplicated "shared", one for "unique"
            assert len(model.calls) == 2
            assert model.calls[0].prompt == "shared"
            assert model.calls[0].n == 2  # Merged requests
            assert model.calls[1].prompt == "unique"


class TestOnlineAPIOrdering:
    TEST_TIMEOUT = 0.1

    @pytest.fixture
    def cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache

    @pytest.mark.asyncio
    async def test_ordered_batch_processing(self, cache):
        """Test that responses maintain order based on request_ids."""
        model = OrderTrackingModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=3,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Create requests with out-of-order IDs
            requests = [
                Request(prompt="p1", n=1,
                        request_id="3", namespace="ns1"),
                Request(prompt="p1", n=1,
                        request_id="1", namespace="ns1"),
                Request(prompt="p1", n=1,
                        request_id="2", namespace="ns1")
            ]

            # Send requests concurrently
            results = await asyncio.gather(*[api.request(req) for req in requests])

            # Results should maintain original order (3,1,2) but responses should be ordered
            assert results[0].data[0] == "p1_count2"  # request_id "3"
            assert results[1].data[0] == "p1_count0"  # request_id "1"
            assert results[2].data[0] == "p1_count1"  # request_id "2"

    @pytest.mark.asyncio
    async def test_ordering_with_cache_hits(self, cache):
        """Test ordering when some responses come from cache."""
        model = OrderTrackingModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=2,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # First request to populate cache
            await api.request(
                Request(prompt="cached", n=1,
                        request_id="0", namespace="ns1")
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

            # Verify ordering is maintained even with cache hits
            assert results[0].data[0].startswith("new_count")      # request_id "3"
            # request_id "1" (from cache)
            assert results[1].data[0].startswith("cached_count")
            assert results[2].data[0].startswith("new_count")      # request_id "2"

    @pytest.mark.asyncio
    async def test_ordering_with_deduplication(self, cache):
        """Test ordering when requests are deduplicated."""
        model = OrderTrackingModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=3,
            timeout=self.TEST_TIMEOUT
        ) as api:
            # Create requests with duplicate prompts but out-of-order IDs
            requests = [
                Request(prompt="dup", n=1,
                        request_id="3", namespace="ns1"),
                Request(prompt="dup", n=1,
                        request_id="1", namespace="ns2"),
                Request(prompt="unique", n=1,
                        request_id="2", namespace="ns1")
            ]

            results = await asyncio.gather(*[api.request(req) for req in requests])

            # Verify responses maintain order despite deduplication
            assert results[0].data[0].startswith("dup_count")      # request_id "3"
            assert results[1].data[0].startswith("dup_count")      # request_id "1"
            assert results[2].data[0].startswith("unique_count")   # request_id "2"

            # Verify deduplication occurred
            assert len([c for c in model.calls if c.prompt == "dup"]) == 1

    @pytest.mark.asyncio
    async def test_batch_interface_ordering(self, cache):
        """Test ordering when using batch_request interface directly."""
        model = OrderTrackingModel()

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=4,
            timeout=self.TEST_TIMEOUT
        ) as api:
            batch = Batch(requests=[
                Request(prompt="p1", n=1,
                        request_id="3", namespace="ns1"),
                Request(prompt="p2", n=1,
                        request_id="1", namespace="ns1"),
                Request(prompt="p1", n=1,
                        request_id="4", namespace="ns2"),
                Request(prompt="p2", n=1,
                        request_id="2", namespace="ns2")
            ])

            results = await api.batch_request(batch)

            # Verify responses maintain original batch order but are internally ordered
            assert results[0].data[0].startswith("p1_count")   # request_id "3"
            assert results[1].data[0].startswith("p2_count")   # request_id "1"
            assert results[2].data[0].startswith("p1_count")   # request_id "4"
            assert results[3].data[0].startswith("p2_count")   # request_id "2"


class TestOnlineAPIBatching:
    TEST_TIMEOUT = 0.1  # Short timeout for tests

    @pytest.fixture
    def cache(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            with Cache(tmpdir) as cache:
                yield cache
    
    @pytest.mark.asyncio
    async def test_allow_batch_overflow_single_large_request(self, cache):
        """Test overflow with a single request larger than batch size."""
        model = CountingMockModel()
        batch_size = 5

        async with OnlineAPI(
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
            assert len(model.calls) == 1
            assert model.calls[0].n == 7
    
    @pytest.mark.asyncio
    async def test_allow_batch_overflow_fills_existing(self, cache):
        """Test overflow where a request fills and exceeds existing items."""
        model = CountingMockModel()
        batch_size = 5

        async with OnlineAPI(
            model=model,
            cache=cache,
            batch_size=batch_size,
            timeout=3,
            allow_batch_overflow=True
        ) as api:
            # Send 3 initial requests (batch size 5)
            initial_requests = [
                Request(prompt=f"init{i}", n=1, request_id=f"i{i}", namespace="test")
                for i in range(3)
            ]
            replies = [api.request(req) for req in initial_requests]

            # Create extra request that leads to overflow
            overflow_request = Request(prompt="overflow", n=7, request_id="of1", namespace="test")
            overflow_reply = api.request(overflow_request)

            # Await the results
            replies.append(overflow_reply)
            all_results = await asyncio.gather(*replies)

            # Extract results for verification
            initial_results = all_results[:3]
            overflow_result = all_results[3]

            # Verify: Should still be one batch with all 10 requests
            assert len([r.data[0] for r in initial_results]) == 3
            assert len(overflow_result.data) == 7
            assert len(model.calls) == 4
            assert model.calls[-1].n == 7 
