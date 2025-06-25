import pytest
import asyncio
import tempfile
from typing import AsyncGenerator
from src.cachesaver.batching import AsyncBatcher
from diskcache import Cache


@pytest.fixture
async def cache() -> AsyncGenerator[Cache, None]:
    """Provide a temporary cache that auto-cleans."""
    with tempfile.TemporaryDirectory() as tmpdir:
        async with Cache(tmpdir) as cache:
            yield cache
