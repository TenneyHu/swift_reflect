import pytest
import asyncio
from typing import List, Any
from src.cachesaver.typedefs import Request, Batch, BatchRequestModel, Response
from src.cachesaver.batching import AsyncBatcher


class MockBatchModel:
    def __init__(self):
        self.calls: List[Batch] = []
        self.response_delay: float = 0.0

    async def batch_request(self, batch: Batch) -> List[List[Any]]:
        self.calls.append(batch)
        if self.response_delay > 0:
            await asyncio.sleep(self.response_delay)
        return [Response(data=["response"]) for req in batch.requests]


class TestAsyncBatcher:
    @pytest.mark.asyncio
    async def test_batching_mechanism(self):
        """Test that requests are properly batched according to batch_size"""
        mock_model = MockBatchModel()
        batch_size = 2

        async with AsyncBatcher(model=mock_model, batch_size=batch_size) as batcher:
            # Create 5 requests - should result in 3 batches (2, 2, 1)
            requests = [
                Request(prompt=f"test{i}", n=1,
                        request_id=str(i), namespace="test")
                for i in range(4)
            ]

            # Send requests concurrently
            tasks = [batcher.request(req) for req in requests]
            await asyncio.gather(*tasks)

            # Verify batching
            assert len(mock_model.calls) == 2
            assert len(mock_model.calls[0].requests) == batch_size
            assert len(mock_model.calls[1].requests) == batch_size

    @pytest.mark.asyncio
    async def test_timeout_mechanism(self):
        """Test that incomplete batches are processed after timeout"""
        mock_model = MockBatchModel()
        batch_size = 3
        timeout = 0.5

        async with AsyncBatcher(model=mock_model, batch_size=batch_size,
                                timeout=timeout) as batcher:
            # Send 2 requests (not enough to fill a batch)
            requests = [
                Request(prompt=f"test{i}", n=1,
                        request_id=str(i), namespace="test")
                for i in range(2)
            ]

            start_time = asyncio.get_event_loop().time()
            tasks = [batcher.request(req) for req in requests]
            await asyncio.gather(*tasks)
            elapsed_time = asyncio.get_event_loop().time() - start_time

            # Verify timeout behavior
            assert elapsed_time >= timeout
            assert len(mock_model.calls) == 1
            assert len(mock_model.calls[0].requests) == 2

    @pytest.mark.asyncio
    async def test_multiple_samples_per_request(self):
        """Test handling of n > 1"""
        mock_model = MockBatchModel()
        batch_size = 2

        async with AsyncBatcher(model=mock_model, batch_size=batch_size) as batcher:
            # Create a request asking for 3 samples
            request = Request(prompt="test", n=4,
                              request_id="1", namespace="test")

            results = await batcher.request(request)

            # Verify results and batching
            assert len(results.data) == 4
            assert len(mock_model.calls) == 2  # Should be split into 2 batches
            assert len(mock_model.calls[0].requests) == 2
            assert len(mock_model.calls[1].requests) == 2

    @pytest.mark.asyncio
    async def test_concurrent_requests(self):
        """Test handling multiple concurrent requests"""
        mock_model = MockBatchModel()
        mock_model.response_delay = 0.1  # Add small delay to simulate processing
        batch_size = 2

        async with AsyncBatcher(model=mock_model, batch_size=batch_size) as batcher:
            # Create 4 concurrent requests
            requests = [
                Request(prompt=f"test{i}", n=1,
                        request_id=str(i), namespace="test")
                for i in range(4)
            ]

            # Send them all at once
            tasks = [batcher.request(req) for req in requests]
            await asyncio.gather(*tasks)

            # Verify batching under concurrent load
            assert len(mock_model.calls) == 2
            assert all(len(batch.requests) == batch_size
                       for batch in mock_model.calls)

    @pytest.mark.asyncio
    async def test_cleanup_on_context_exit(self):
        """Test that worker task is properly cleaned up when context exits"""
        mock_model = MockBatchModel()
        batch_size = 2

        async with AsyncBatcher(model=mock_model, batch_size=batch_size, timeout=0.1) as batcher:
            # Verify worker task is running
            assert not batcher.worker_task.done()

            # Do some work
            request = Request(prompt="test", n=1,
                              request_id="1", namespace="test")
            await batcher.request(request)

        # Verify worker task is cleaned up
        assert batcher.worker_task.done()

    @pytest.mark.asyncio
    async def test_allow_batch_overflow_single_large_request(self):
        """Test overflow with a single request larger than batch size."""
        mock_model = MockBatchModel()
        batch_size = 5
        timeout = 0.1  # Short timeout

        async with AsyncBatcher(
            model=mock_model,
            batch_size=batch_size,
            timeout=timeout,
            allow_batch_overflow=True
        ) as batcher:
            # Single request larger than batch size
            request1 = Request(prompt="large", n=7, request_id="r1", namespace="test")
            result1 = await batcher.request(request1)

            # Verify: Should process as one batch despite size
            assert len(result1.data) == 7
            assert len(mock_model.calls) == 1
            assert len(mock_model.calls[0].requests) == 7

    @pytest.mark.asyncio
    async def test_allow_batch_overflow_fills_existing(self):
        """Test overflow where a request fills and exceeds existing items."""
        mock_model = MockBatchModel()
        batch_size = 5
        timeout = 3  # Short timeout

        async with AsyncBatcher(
            model=mock_model,
            batch_size=batch_size,
            timeout=timeout,
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
            assert len(mock_model.calls) == 1
            assert len(mock_model.calls[0].requests) == 10 # 3 initial + 7 overflow

