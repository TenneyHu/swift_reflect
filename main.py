import asyncio
import logging
import tempfile
from typing import List, Any
from collections import defaultdict
from diskcache import Cache

from src.cachesaver.typedefs import Request, Batch, BatchRequestModel
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
            responses.append(request_responses)
        return responses


async def main():
    # Configure logging
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    logger = logging.getLogger(__name__)

    # Create temporary cache
    with tempfile.TemporaryDirectory() as tmpdir:
        cache = Cache(tmpdir)

        model = BatchCounterModel()
        api = OrderedLocalAPI(
            model=model,
            cache=cache,
            collection_batch_size=3,
            hardware_batch_size=2,
            timeout=1
        )

        requests = [
            Request(prompt="p1", n=2,
                    request_id="2", namespace="ns1"),
            Request(prompt="p2", n=1,
                    request_id="1", namespace="ns1")
        ]

        try:
            results = await asyncio.gather(*[api.request(req) for req in requests])

            # Debug logging
            logger.debug("Results: %s", results)
            logger.debug("Number of model calls: %d", len(model.batch_calls))
            for i, batch in enumerate(model.batch_calls):
                logger.debug("Batch %d requests: %s", i, batch.requests)

            # Verify ordering and sample counts
            assert len(results[0]) == 2  # n=2
            assert len(results[1]) == 1  # n=1
            assert results[1][0].startswith("p2_count")  # id "1"
            assert all(r.startswith("p1_count") for r in results[0])  # id "2"

            logger.info("All assertions passed!")
        except Exception as e:
            logger.error("Test failed: %s", str(e))
            raise


if __name__ == "__main__":
    asyncio.run(main())
