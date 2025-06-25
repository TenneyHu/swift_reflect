import asyncio
from contextlib import AbstractAsyncContextManager
from typing import Any


class AsyncRoundRobinLimiter(AbstractAsyncContextManager):

    def __init__(self):
        self.queue = asyncio.Queue()
        self.resources = []

    async def __aenter__(self):
        resource = await self.queue.get()
        assert resource.in_use is False, "Resource is already in use"
        resource.in_use = True
        return resource

    def add_resource(self, data: Any = None):
        resource = AsyncRoundRobinResource(self, data)
        self.resources.append(resource)
        self.queue.put_nowait(resource)

    async def __aexit__(self, exc_type, exc, tb):
        return None


class AsyncRoundRobinResource:

    def __init__(self, limiter: AsyncRoundRobinLimiter, data: Any = None):
        self.data = data
        self.in_use = False
        self.limiter = limiter

    def reschedule(self, time_taken: float, amount_used: float):
        self.limiter.queue.put_nowait(self)

    def free(self, time_taken: float, amount_used: float):
        self.reschedule(time_taken, amount_used)
        self.in_use = False
