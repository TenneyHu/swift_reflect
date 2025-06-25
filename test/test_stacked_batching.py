import pytest
import asyncio
from typing import List, Any
from collections import defaultdict

from src.cachesaver.typedefs import Request, Response, Batch, BatchRequestModel
from src.cachesaver.batching import AsyncBatcher


class BatchTrackingModel(BatchRequestModel):
    """Model that tracks batch sizes and maintains per-prompt counters."""

    def __init__(self):
        self.batch_calls: List[Batch] = []
        self.prompt_counters = defaultdict(int)

    async def batch_request(self, batch: Batch) -> List[List[Any]]:
        self.batch_calls.append(batch)
        responses = []
        for request in batch.requests:
            counter = self.prompt_counters[request.prompt]
            responses.append(Response(data=[f"{request.prompt}_count{counter}"]))
            self.prompt_counters[request.prompt] += 1
        return responses


@pytest.mark.asyncio
async def test_stacked_batchers():
    """Test three stacked batchers with different batch sizes and timeouts."""
    model = BatchTrackingModel()

    async with AsyncBatcher(
        model=model,
        batch_size=2,
        timeout=0.05,
        name="inner"
    ) as inner_batcher:
        async with AsyncBatcher(
            model=inner_batcher,
            batch_size=4,
            timeout=0.1,
            name="middle"
        ) as middle_batcher:
            async with AsyncBatcher(
                model=middle_batcher,
                batch_size=3,
                timeout=0.03,
                name="outer"
            ) as outer_batcher:
                # Create test requests
                requests = [
                    Request(prompt=f"prompt_{i}",
                            n=1,
                            request_id=str(i),
                            namespace="test")
                    for i in range(10)
                ]

                # Send requests with delays
                tasks = []
                for request in requests:
                    task = asyncio.create_task(outer_batcher.request(request))
                    tasks.append(task)
                    await asyncio.sleep(0.04)  # Simulate gradual arrival

                # Wait for all requests
                results = await asyncio.gather(*tasks)

                # Verify results
                assert len(results) == 10
                assert all(isinstance(r, Response) for r in results)
                assert all(r.data[0].startswith("prompt_") for r in results)

                # Check batch sizes at model level
                batch_sizes = [len(batch.requests)
                               for batch in model.batch_calls]
                # Inner batcher limit
                assert all(size <= 2 for size in batch_sizes)
