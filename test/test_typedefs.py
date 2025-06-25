import pytest
from typing import List, Any
from src.cachesaver.typedefs import SingleRequestModel, BatchRequestModel, Request, Batch, Response


class TestSingleRequestModel:
    class ValidModel:
        async def request(self, request: Request) -> List[Any]:
            return ["response"]

    class InvalidModel:
        async def wrong_method(self, request: Request) -> List[Any]:
            return ["response"]

    def test_protocol_compliance(self):
        valid_model = self.ValidModel()
        invalid_model = self.InvalidModel()
        assert isinstance(valid_model, SingleRequestModel)
        assert not isinstance(invalid_model, SingleRequestModel)

    @pytest.mark.asyncio
    async def test_valid_implementation(self):
        model = self.ValidModel()
        request = Request(prompt="test", n=1,
                          request_id="1", namespace="test")
        result = await model.request(request)
        assert isinstance(result, list)
        assert isinstance(result[0], str)


class TestBatchRequestModel:
    class ValidModel:
        async def batch_request(self, batch: Batch) -> List[List[Any]]:
            return [["response"] for _ in batch.requests]

    class InvalidModel:
        async def different_method_name(self, batch: Batch) -> List[List[Any]]:
            return [["response"] for _ in batch.requests]

    def test_protocol_compliance(self):
        valid_model = self.ValidModel()
        invalid_model = self.InvalidModel()
        assert isinstance(valid_model, BatchRequestModel)
        assert not isinstance(invalid_model, BatchRequestModel)

    @pytest.mark.asyncio
    async def test_valid_implementation(self):
        model = self.ValidModel()
        batch = Batch(requests=[
            Request(prompt="test1", n=1,
                    request_id="1", namespace="test"),
            Request(prompt="test2", n=2,
                    request_id="2", namespace="test")
        ])
        result = await model.batch_request(batch)
        assert isinstance(result, list)
        assert all(isinstance(x, list) for x in result)
        assert all(isinstance(item, str)
                   for sublist in result for item in sublist)


class TestAsyncBatcher:
    class ValidBatchModel:
        async def batch_request(self, batch: Batch) -> List[Response]:
            return [Response(data=["response"]) for _ in batch.requests]

    @pytest.mark.asyncio
    async def test_protocol_compliance(self):
        from src.cachesaver.batching import AsyncBatcher
        async with AsyncBatcher(model=self.ValidBatchModel(), batch_size=2) as batcher:
            assert isinstance(batcher, SingleRequestModel)
            assert isinstance(batcher, BatchRequestModel)

    @pytest.mark.asyncio
    async def test_batch_request(self):
        from src.cachesaver.batching import AsyncBatcher
        async with AsyncBatcher(model=self.ValidBatchModel(), batch_size=2) as batcher:
            batch = Batch(requests=[
                Request(prompt="test1", n=1,
                        request_id="1", namespace="test"),
                Request(prompt="test2", n=3,
                        request_id="2", namespace="test")
            ])
            result = await batcher.batch_request(batch)
            assert isinstance(result, list)
            assert all(isinstance(x, Response) for x in result)
