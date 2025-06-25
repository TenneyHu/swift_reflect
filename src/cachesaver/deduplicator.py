import asyncio
from collections import defaultdict
from typing import List, Any
from dataclasses import replace

from deepdiff import DeepHash

from .typedefs import Request, Response, Batch, BatchRequestModel


class AsyncDeduplicator(BatchRequestModel):
    def __init__(self, model: BatchRequestModel, correctness: bool = False):
        # Actual API
        self.model = model
        self.correctness = correctness # Happy to remove this if not needed

    async def batch_request(self, batch: Batch) -> List[Response]:
        # find duplicate prompts
        key2request_and_namespace = {}
        key2requests = defaultdict(list)

        # requests with the same prompt and namespace are identical
        # prompt&namespace are hashed to a key, then all requests under this key are bundled together
        for request in batch.requests:
            request_and_namespace = (
                request.hash(),
                request.namespace,
                "_".join(request.request_id.split("_")[:-1]) if self.correctness else None
                )
            key = DeepHash(request_and_namespace)[request_and_namespace]

            key2request_and_namespace[key] = request_and_namespace
            key2requests[key].append(request)

        # bundle requests and send them as one
        merged_requests = []
        for key in key2request_and_namespace:
            request_and_namespace = key2request_and_namespace[key]
            requests_to_merge = key2requests[key]

            # first entry in metadata is the number of desired answers
            total = sum([m.n for m in requests_to_merge])

            # we now dispatch all identical prompts in one request
            merged_request = replace(requests_to_merge[0], n=total)
            merged_requests.append(merged_request)

        responses = await self.model.batch_request(
            Batch(requests=merged_requests))

        # map the results back to the keys
        key2results = {key: result for key, result in zip(
            key2request_and_namespace.keys(), responses)}

        # Flags whether the result was duplicated or not
        for key, result in key2results.items():
            if result.cached:
                flags = [not flag for flag in result.cached] # reversed cached flags
                # The first duplicated result that is not cached is the original.
                # The remaining uncached results are the duplicates.
                for i, flag in enumerate(flags):
                    if flag:
                        flags[i] = not flags[i]
                        break
            else:
                flags = [False] + [True] * (len(result.data) - 1)
            key2results[key] = key2results[key]._replace(duplicated=flags)

        # split the results back into the individual requests
        results = []
        for request in batch.requests:
            request_and_namespace = (
                request.hash(),
                request.namespace,
                "_".join(request.request_id.split("_")[:-1]) if self.correctness else None
                )
            key = DeepHash(request_and_namespace)[request_and_namespace]

            # our API always works with lists of requests and then for each request a list of responses
            # correspondingly this is a list of lists
            # but this result corresponds to the one deduplicated request (with many responses)
            # so it should be a list of length one
            result = key2results[key]
            metadata = key2requests[key].pop(0)

            assert metadata.n == request.n
            assert len(result.data) >= metadata.n

            results.append(
                Response(
                    data=result.data[:metadata.n],
                    cached=result.cached[:metadata.n] if result.cached else [False]*len(result.data[:metadata.n]),
                    duplicated = result.duplicated[:metadata.n]
                )
            )

            key2results[key]= Response(
                data=result.data[metadata.n:],
                cached=result.cached[metadata.n:] if result.cached else [False]*len(result.data[metadata.n:]),
                duplicated=result.duplicated[metadata.n:]
            )
        return results
