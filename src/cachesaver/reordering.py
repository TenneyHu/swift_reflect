from typing import List, Any
from .typedefs import Request, Batch, SingleRequestModel, BatchRequestModel


class RequestReorderer(BatchRequestModel):
    """
    Ensures deterministic ordering of responses based on request_id.

    Requests are ordered by request_id before being sent to the model,
    and responses are reordered to match the original request order.
    """

    def __init__(self, model: BatchRequestModel):
        self.model = model

    async def batch_request(self, batch: Batch) -> List[List[Any]]:
        """Reorder batch requests by request_id and restore original order in responses."""
        # Create mapping of original positions
        assert len({req.request_id for req in batch.requests}) == len([req.request_id for req in batch.requests]),\
            f"Request IDs must be unique in a reorderer's batch: {[req.request_id for req in batch.requests]}"
        
        original_order = {req.request_id: i for i,
                          req in enumerate(batch.requests)}

        # Sort requests by request_id
        sorted_requests = sorted(batch.requests, key=lambda x: x.request_id)
        sorted_batch = Batch(requests=sorted_requests)

        # Get responses for sorted requests
        sorted_responses = await self.model.batch_request(sorted_batch)

        # Restore original order
        reordered_responses = [None] * len(sorted_responses)
        for req, resp in zip(sorted_requests, sorted_responses):
            original_idx = original_order[req.request_id]
            reordered_responses[original_idx] = resp

        return reordered_responses
