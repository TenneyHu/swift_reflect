import pytest
from typing import List, Any
from collections import defaultdict

from src.cachesaver.typedefs import Request, Batch, BatchRequestModel
from src.cachesaver.reordering import RequestReorderer


class PromptCounterModel(BatchRequestModel):
    """Mock model that maintains separate counters for each prompt."""

    def __init__(self):
        self.prompt_counters = defaultdict(int)
        self.calls: List[Batch] = []

    async def batch_request(self, batch: Batch) -> List[List[Any]]:
        self.calls.append(batch)
        responses = []
        for request in batch.requests:
            # Generate n responses for this request
            request_responses = []
            for _ in range(request.n):
                counter = self.prompt_counters[request.prompt]
                request_responses.append(f"{request.prompt}_count{counter}")
                self.prompt_counters[request.prompt] += 1
            responses.append(request_responses)
        return responses


class TestRequestReorderer:
    @pytest.mark.asyncio
    async def test_basic_reordering(self):
        """Test that responses are reordered according to request_ids."""
        model = PromptCounterModel()
        reorderer = RequestReorderer(model=model)

        batch = Batch(requests=[
            Request(prompt="p1", n=1,
                    request_id="2", namespace="test"),
            Request(prompt="p1", n=1,
                    request_id="1", namespace="test"),
            Request(prompt="p1", n=1,
                    request_id="3", namespace="test")
        ])

        results = await reorderer.batch_request(batch)

        # Verify the model received sorted requests
        received_batch = model.calls[0]
        assert [r.request_id for r in received_batch.requests] == ["1", "2", "3"]

        # Verify responses are in original order
        # request_id "2" was first in input
        assert results[0][0] == "p1_count1"
        # request_id "1" was second in input
        assert results[1][0] == "p1_count0"
        # request_id "3" was third in input
        assert results[2][0] == "p1_count2"

    @pytest.mark.asyncio
    async def test_different_prompts_reordering(self):
        """Test reordering with different prompts."""
        model = PromptCounterModel()
        reorderer = RequestReorderer(model=model)

        batch = Batch(requests=[
            Request(prompt="p2", n=1,
                    request_id="2", namespace="test"),
            Request(prompt="p1", n=1,
                    request_id="1", namespace="test"),
            Request(prompt="p2", n=1,
                    request_id="3", namespace="test")
        ])

        results = await reorderer.batch_request(batch)

        # Verify counters are separate per prompt
        assert results[0][0] == "p2_count0"  # First p2, request_id "2"
        assert results[1][0] == "p1_count0"  # First p1, request_id "1"
        assert results[2][0] == "p2_count1"  # Second p2, request_id "3"

    @pytest.mark.asyncio
    async def test_multiple_samples_reordering(self):
        """Test reordering with requests asking for multiple samples."""
        model = PromptCounterModel()
        reorderer = RequestReorderer(model=model)

        batch = Batch(requests=[
            Request(prompt="p1", n=2,
                    request_id="2", namespace="test"),
            Request(prompt="p2", n=1,
                    request_id="1", namespace="test")
        ])

        results = await reorderer.batch_request(batch)

        assert len(results[0]) == 2  # First request wanted 2 samples
        assert len(results[1]) == 1  # Second request wanted 1 sample
        assert results[0][0] == "p1_count0"  # First p1
        assert results[1][0] == "p2_count0"  # First p2

    @pytest.mark.asyncio
    async def test_lexicographic_ordering(self):
        """Test that request_ids are sorted lexicographically."""
        model = PromptCounterModel()
        reorderer = RequestReorderer(model=model)

        batch = Batch(requests=[
            Request(prompt="p1", n=1,
                    request_id="req10", namespace="test"),
            Request(prompt="p1", n=1,
                    request_id="req2", namespace="test")
        ])

        results = await reorderer.batch_request(batch)

        received_batch = model.calls[0]
        assert [r.request_id for r in received_batch.requests] == [
            "req10", "req2"]
        assert results[0][0] == "p1_count0"
        assert results[1][0] == "p1_count1"
