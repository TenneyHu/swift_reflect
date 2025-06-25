# Cachesaver
Cachesaver is a high-efficiency caching library for experiments with large language models (LLMs), designed to minimize costs, improve reproducibility, and streamline debugging. The library enables caching of multiple responses per query, tracks usage for unique sampling across runs, and ensures statistical integrity. Built on Python’s `asyncio`, Cachesaver supports asynchronous execution, request batching, prompt deduplication, and race-condition prevention.

## Installation

```bash
# Install with test dependencies (Will also publish in pypi soon).
git clone https://github.com/au-clan/cachesaver-core.git
cd cachesaver-core
pip install -e ".[test]"

# Run tests to verify everything works
pytest test/ -v
```
## Usage

### Defining a request
To make a call to the Cachesaver the prompt needs to be wrapped in our `Request` class. The exact prompt (`prompt`) and number of samples requested (`n`) have to be specified. The `request_id` needs to be unique. 

Samples are stored in a cache and can later be reused but a single sample is only used once within a namespace. So specifying `namespace` correclty will affect whether the sample will be requested by an LLM or retrieved from the cache. For example, 2 agents working together towards solving a puzzle should share the same namespace. On the other hand, agents working on different puzzles should not share a namespace.

```python
from cachesaver.typedefs import Request, Batch

request = Request(
    prompt="What's the meaning of life",
    n=1, 
    request_id="unique_request_id",
    namespace="any_namespace")
```

If you'd like to specify more parameters within a `Request` such as temperature, you could redefine it using:

```python
from cachesevar.typedefs import Request
from dataclasses import dataclass

@dataclass(frozen=True)
class Request(Request):
    temperature: float
```

### Defining an online LLM model
To use Cachesaver in your LLM experiments a class for the actual model has to be defined. In the simplest case, all needed is a wrapper around an LLM API client. The role of this wrapper is to parse the inputs and the outputs for the client. Specifically, get the prompt and number of samples from the `Request`, as well as to create a `Response` object. Keep in mind that `Response.data` is a `List[Any]` so you can save anything you want there, not just the response messsage content.
```python
import asyncio
from typing import List, Any
from src.cachesaver.typedefs import Request, Batch, Response, SingleRequestModel, BatchRequestModel

class OnlineLLM(SingleRequestModel, BatchRequestModel):
    def __init__(self, client: Any, model: str):
        self.client = client
        self.model = model

    async def request(self, request: Request) -> Response:
        completion = await self.client.chat.completions.create(
            messages = [
                {
                    "role" : "user",
                    "content" : request.prompt
                }
            ],
            model = self.model,
            n = request.n,
        )
        response = Response(
            data = [choice.message.content for choice in completion.choices]
        )
        return response
    
    async def batch_request(self, batch: Batch) -> List[Response]:
        requests = [self.request(request) for request in batch.requests]
        completions = await asyncio.gather(*requests)
        return completions
```

The LLM API client mentioned above, can be basically anything as long as it's asynchronous. For example in the above configuration, any of the following clients can be inserted: [AsyncOpenAI](https://github.com/openai/openai-python/blob/f66d2e6fdc51c4528c99bb25a8fbca6f9b9b872d/src/openai/_client.py#L278), [AsyncTogether](https://github.com/togethercomputer/together-python/blob/b24a5c3f64202cd9ae4c064b92c07d7793ad3dc9/src/together/client.py#L92) or [AsyncGroq](https://github.com/groq/groq-python/blob/9f14aacde8c5b5a7cd6110ad0ba722235706d432/src/groq/_client.py#L219). For example:
```python
from together import AsyncTogether

client = AsyncTogether()
model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
model = OnlineLLM(
    client=client,
    model_name=model_name
)
```

**Todo: Example with local LLM**

### Integrating with CacheSaver
Once the LLM class is ready, it can be seamlessly integrated into the CacheSaver pipeline. The final variables to be configured are the `batch_size` that you'd like to use and `timeout`, the time a batch might remain unfilled before it is executed.

```python
from cachesaver.pipelines import onineAPI
from diskcache import Cache

# Cache to store responses
cache = Cache(f"./caches/test")

pipeline = OnlineAPI(
    model=model,
    cache=cache,
    batch_size=5,
    timeout=1
)
```

`OnlineAPI` is a typical pipeline that looks like this:
- Batcher: Collects requests until a batch is full or a timeout occurs
- Reorderer: Ensures that the responses are given in the same order with their corresponding requests.
- Deduplicator: Ensures that if the same prompt is sent multiple times, only one request is sent to the backend
- Cacher: Reuses computations wherever possible
- Your model: The object that offers a `request(prompt)` method

Our pipeline stages will use the `namespace` and `request_id` parameters to ensure that requests are processed correctly. But your model can just ignore them.

### The Response object
The `Response` object is a wrapper around the data generated by the previously defined model. Everything that the model generates is included in `Response.data`. For example, if a request was made with number of samples `n` being 2, `Response.data` will be of the type `List[str, str]` (assuming only the message content is returned by the model). On top of that `Response.cached` displays whether each response was retrieved from the cache and `Response.duplicated` was duplicated.

A note on duplicatation: Imagine within a batch a number of requests of the same prompt requesting 5 samples (reminder: number of such requests <= 5). Cachesaver will recognize this and take the appropriate steps so that input tokens are only billed once. Then 1 of the five samples will be considered the original and the rest will be considered duplicated. The "original" sample is selected randomly.
