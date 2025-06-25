import asyncio
import time
import logging
from typing import List, Any
from dataclasses import replace
from .typedefs import Request, Batch, Response, SingleRequestModel, BatchRequestModel
from .resources import AsyncResource

logger = logging.getLogger(__name__)


class AsyncBatcher(AsyncResource, SingleRequestModel, BatchRequestModel):
    """
    Groups requests into batches using a queue-based producer-consumer pattern.
    Requests wait at most `timeout` seconds for the batch to be filled.
    If the batch is not filled in time, it is sent to the model as is.
    """
    SHUTDOWN_SIGNAL = (
        None, None, None)  # (timestamp, request, future) tuple for shutdown

    def __init__(self, model: BatchRequestModel, batch_size: int, timeout: float = 30,
                 name: str = "unnamed", allow_batch_overflow: bool = False):
        self.name = name
        self.batch_size = batch_size
        self.timeout = timeout
        self.model = model
        self.queue = asyncio.Queue()
        self.allow_batch_overflow = allow_batch_overflow

        # Start background worker
        self.worker_task = asyncio.create_task(self._batch_worker())
        logger.debug("[Batcher %s] Initialized with batch_size=%d, timeout=%f",
                     self.name, batch_size, timeout)

    async def cleanup(self):
        """Cleanup the background worker task."""
        logger.debug("[Batcher %s] Initiating shutdown", self.name)
        # Wake up worker with shutdown signal
        await self.queue.put(self.SHUTDOWN_SIGNAL)
        await self.worker_task
        logger.debug("[Batcher %s] Worker shut down cleanly", self.name)

    async def request(self, request: Request) -> List[Any]:
        """Handle a single request, potentially generating multiple samples."""
        logger.debug("[Batcher %s] Received request: %s", self.name, request)

        futures = []
        # Use put_nowait for all but the last request if num_desired > 1
        num_to_put_nowait = request.n - 1
        for i in range(request.n):
            single_request = replace(request, n=1, request_id=f"{request.request_id}_{i}")
            # Create future in current event loop
            future = asyncio.get_event_loop().create_future()
            timestamp = time.time()
            item_to_put = (timestamp, single_request, future)

            if i < num_to_put_nowait:
                try:
                    self.queue.put_nowait(item_to_put)
                    logger.debug(
                        "[Batcher %s] put_nowait item %s. Queue size: %d",
                        self.name, single_request.request_id, self.queue.qsize()
                    )
                except asyncio.QueueFull:
                    # If queue is full even with put_nowait, we have to wait
                    # This might break the "keep together" goal, but avoids losing requests
                    logger.warning(
                        "[Batcher %s] Queue full during put_nowait, falling back to await put.", self.name)
                    await self.queue.put(item_to_put)
                    logger.debug(
                        "[Batcher %s] await put item %s (fallback). Queue size: %d",
                        self.name, single_request.request_id, self.queue.qsize()
                    )
            else:
                # Await putting the last item (or the only item if num_desired == 1)
                await self.queue.put(item_to_put)
                logger.debug(
                    "[Batcher %s] await put item %s (last). Queue size: %d",
                    self.name, single_request.request_id, self.queue.qsize()
                )


            futures.append(future)

        # Wait for all samples
        responses = await asyncio.gather(*futures)
        response = Response(
            data=[r.data[0] for r in responses],
            cached=[r.cached[0] if r.cached is not None else False for r in responses],
            duplicated=[r.duplicated[0] if r.duplicated is not None else False for r in responses] 
        )
        return response  # Unwrap single-item responses

    async def batch_request(self, batch: Batch) -> List[List[Any]]:
        """Process a batch of requests."""
        return await asyncio.gather(*[self.request(req) for req in batch.requests])

    async def _batch_worker(self):
        """Background worker that processes requests in batches."""
        try:
            while True:
                batch = []
                futures = []
                requests = []

                try:
                    # Get first request to start the batch
                    first_item = await self.queue.get()
                    logger.debug(
                        "[Batcher %s] await get first item. Queue size: %d",
                        self.name, self.queue.qsize()
                    )

                    # Check for shutdown signal
                    if first_item == self.SHUTDOWN_SIGNAL:
                        logger.debug(
                            "[Batcher %s] Received shutdown signal", self.name)
                        break

                    first_timestamp, first_request, first_future = first_item
                    batch.append(first_request)
                    futures.append(first_future)
                    requests.append(first_request)

                    # Collect more requests up to batch_size or timeout
                    while len(batch) < self.batch_size:
                        now = time.time()
                        max_wait = max(0, self.timeout -
                                       (now - first_timestamp))

                        try:
                            item = await asyncio.wait_for(
                                self.queue.get(), timeout=max_wait)
                            logger.debug(
                                "[Batcher %s] await get subsequent item. Queue size: %d",
                                self.name, self.queue.qsize()
                            )

                            # Check for shutdown signal
                            if item == self.SHUTDOWN_SIGNAL:
                                logger.debug("[Batcher %s] Received shutdown signal mid-batch",
                                             self.name)
                                # Process current batch before shutting down
                                break

                            timestamp, request, future = item
                            batch.append(request)
                            futures.append(future)
                            requests.append(request)
                        except asyncio.TimeoutError:
                            logger.debug("[Batcher %s] Timeout waiting for batch completion",
                                         self.name)
                            break
                    
                    # Add overflow items if allowed
                    if self.allow_batch_overflow:
                        logger.debug(
                            "[Batcher %s] Checking for overflow items.", self.name)
                        try:
                            while True:  # Drain queue without waiting
                                item = self.queue.get_nowait()
                                logger.debug(
                                    "[Batcher %s] get_nowait item in overflow. Queue size: %d",
                                    self.name, self.queue.qsize()
                                )
                                if item == self.SHUTDOWN_SIGNAL:
                                    # If shutdown received during drain, just stop draining.
                                    # The main loop will handle the signal again.
                                    logger.debug(
                                        "[Batcher %s] Shutdown signal hit during overflow drain.", self.name)
                                    break
                                timestamp, request, future = item
                                batch.append(request)
                                futures.append(future)
                                requests.append(request)
                                logger.debug(
                                    "[Batcher %s] Added overflow item %s.", self.name, request.request_id)
                        except asyncio.QueueEmpty:
                            pass  # No more items available immediately

                    # Add overflow items if allowed
                    if self.allow_batch_overflow:
                        logger.debug(
                            "[Batcher %s] Checking for overflow items.", self.name)
                        try:
                            while True:  # Drain queue without waiting
                                item = self.queue.get_nowait()
                                logger.debug(
                                    "[Batcher %s] get_nowait item in overflow. Queue size: %d",
                                    self.name, self.queue.qsize()
                                )
                                if item == self.SHUTDOWN_SIGNAL:
                                    # If shutdown received during drain, just stop draining.
                                    # The main loop will handle the signal again.
                                    logger.debug(
                                        "[Batcher %s] Shutdown signal hit during overflow drain.", self.name)
                                    break
                                timestamp, request, future = item
                                batch.append(request)
                                futures.append(future)
                                requests.append(request)
                                logger.debug(
                                    "[Batcher %s] Added overflow item %s.", self.name, request.request_id)
                        except asyncio.QueueEmpty:
                            pass  # No more items available immediately

                    # Process batch if we have any requests
                    if batch:
                        try:
                            logger.debug("[Batcher %s] Processing batch of size %d",
                                         self.name, len(batch))
                            responses = await self.model.batch_request(Batch(requests=requests))
                            for future, response in zip(futures, responses):
                                if not future.done():
                                    future.set_result(response)
                        except Exception as e:
                            logger.error("[Batcher %s] Batch processing failed: %s",
                                         self.name, str(e))
                            for future in futures:
                                if not future.done():
                                    future.set_exception(e)

                except Exception as e:
                    logger.error("[Batcher %s] Error processing batch: %s",
                                 self.name, str(e))
                    for future in futures:
                        if not future.done():
                            future.set_exception(e)

        except Exception as e:
            logger.error("[Batcher %s] Worker failed with unexpected error: %s",
                         self.name, str(e))
            # Make sure to cancel any pending futures before exiting
            for future in futures:
                if not future.done():
                    future.set_exception(e)
            raise
        finally:
            # Cancel any remaining futures when shutting down
            while not self.queue.empty():
                item = await self.queue.get()
                if item == self.SHUTDOWN_SIGNAL:
                    continue
                _, _, future = item
                if not future.done():
                    future.cancel()
            logger.debug(
                "[Batcher %s] Worker shut down, all futures resolved", self.name)
