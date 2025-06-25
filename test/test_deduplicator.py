import pytest
from typing import List, Any
from src.cachesaver.typedefs import Request, Response, Batch, BatchRequestModel
from src.cachesaver.deduplicator import AsyncDeduplicator
from collections import defaultdict


class MockBatchModel:
    def __init__(self):
        self.calls: List[Batch] = []
        self.responses: List[List[Any]] = []

    async def batch_request(self, batch: Batch) -> List[Response]:
        self.calls.append(batch)
        # Generate responses based on n
        responses = []
        for request in batch.requests:
            responses.append(
                Response(data=[f"response_{i}" for i in range(request.n)]))
        return responses


class DeterministicMockModel:
    def __init__(self):
        self.counters = defaultdict(int)

    async def batch_request(self, batch: Batch) -> List[Response]:
        responses = []
        for request in batch.requests:
            request_responses = []
            key = (request.namespace, request.prompt)
            for _ in range(request.n):
                response = f"{request.prompt}_{request.namespace}_{self.counters[key]}"
                self.counters[key] += 1
                request_responses.append(response)
            responses.append(Response(data=request_responses))
        return responses


class TestAsyncDeduplicator:
    @pytest.mark.asyncio
    async def test_deduplication_same_namespace(self):
        """Test that identical prompts in same namespace are deduplicated"""
        mock_model = MockBatchModel()
        deduplicator = AsyncDeduplicator(model=mock_model)

        # Create requests with same prompt and namespace
        batch = Batch(requests=[
            Request(prompt="test", n=2,
                    request_id="1", namespace="test"),
            Request(prompt="test", n=3,
                    request_id="2", namespace="test"),
            Request(prompt="test", n=1,
                    request_id="3", namespace="test")
        ])

        results = await deduplicator.batch_request(batch)

        # Verify deduplication
        assert len(mock_model.calls) == 1  # Should be merged into single call
        merged_request = mock_model.calls[0].requests[0]
        assert merged_request.n == 6  # Sum of all n
        assert merged_request.prompt == "test"
        assert merged_request.namespace == "test"

        # Verify results were split correctly
        assert len(results) == 3
        assert len(results[0].data) == 2  # First request wanted 2
        assert len(results[1].data) == 3  # Second request wanted 3
        assert len(results[2].data) == 1  # Third request wanted 1

    @pytest.mark.asyncio
    async def test_no_deduplication_different_namespaces(self):
        """Test that prompts in different namespaces are not deduplicated"""
        mock_model = MockBatchModel()
        deduplicator = AsyncDeduplicator(model=mock_model)

        # Create requests with same prompt but different namespaces
        batch = Batch(requests=[
            Request(prompt="test", n=2,
                    request_id="1", namespace="ns1"),
            Request(prompt="test", n=2,
                    request_id="2", namespace="ns2")
        ])

        results = await deduplicator.batch_request(batch)

        # Verify no deduplication occurred
        assert len(mock_model.calls) == 1  # Still one batch call
        # But with two separate requests
        assert len(mock_model.calls[0].requests) == 2

        # Verify each request maintained its namespace
        requests = mock_model.calls[0].requests
        assert requests[0].namespace != requests[1].namespace

        # Verify results
        assert len(results) == 2
        assert all(len(result.data) == 2 for result in results)

    @pytest.mark.asyncio
    async def test_mixed_deduplication(self):
        """Test handling of mixed duplicates and unique requests"""
        mock_model = MockBatchModel()
        deduplicator = AsyncDeduplicator(model=mock_model)

        batch = Batch(requests=[
            Request(prompt="test1", n=2,
                    request_id="1", namespace="ns1"),
            Request(prompt="test1", n=3,
                    request_id="2", namespace="ns1"),
            Request(prompt="test2", n=1,
                    request_id="3", namespace="ns1"),
            Request(prompt="test1", n=2,
                    request_id="4", namespace="ns2")
        ])

        results = await deduplicator.batch_request(batch)

        # Verify partial deduplication
        assert len(mock_model.calls) == 1
        requests = mock_model.calls[0].requests
        assert len(requests) == 3  # Should have 3 requests after deduplication

        # Verify results
        assert len(results) == 4  # Still 4 original results
        assert len(results[0].data) == 2
        assert len(results[1].data) == 3
        assert len(results[2].data) == 1
        assert len(results[3].data) == 2

    @pytest.mark.asyncio
    async def test_response_order_preservation(self):
        """Test that deduplication doesn't change the order or content of responses"""
        mock_model = DeterministicMockModel()
        deduplicator = AsyncDeduplicator(model=mock_model)

        # Create a batch with various patterns of duplicates
        batch = Batch(requests=[
            Request(prompt="p1", n=2,
                    request_id="1", namespace="ns1"),
            Request(prompt="p1", n=1, request_id="2",
                    namespace="ns1"),  # Duplicate
            Request(prompt="p2", n=1,
                    request_id="3", namespace="ns1"),
            Request(prompt="p1", n=2, request_id="4",
                    namespace="ns2"),  # Same prompt, different ns
            Request(prompt="p2", n=1, request_id="5",
                    namespace="ns1"),  # Duplicate of p2
        ])

        # Get responses with deduplication
        mock_model.counter = 0  # Reset counter
        results_with_dedup = await deduplicator.batch_request(batch)

        # Get responses without deduplication (directly from model)
        mock_model = DeterministicMockModel()
        results_without_dedup = await mock_model.batch_request(batch)

        # Compare results
        assert len(results_with_dedup) == len(results_without_dedup)
        for with_dedup, without_dedup in zip(results_with_dedup, results_without_dedup):
            for wd, woutd in zip(with_dedup.data, without_dedup.data):
                assert wd == woutd, (
                    f"Mismatch in results: "
                    f"with dedup: {wd}, "
                    f"without dedup: {woutd}"
                )
