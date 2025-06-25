
## OUTDATED INSTRUCTIONS BELOW

### Quick start

If you are using OpenAI, cachesaver is a simple drop-in solution to cache your requests across runs:
```python
# this is the original OpenAI client
aclient = AsyncOpenAI()
OPENAI_KEY = os.getenv("OPENAI_API_KEY")
cache = Cache("cache_dir")

with cache:
    
    # prevents rate limiting errors when using 3rd party APIs
    # here we're creating 4 resources out of one API key
    # this will mean there are at most 4 concurrent requests to this key
    limiter = AsyncRoundRobinLimiter()
    for _ in range(4):
        limiter.add_resource(OPENAI_KEY)

    # our wrapper
    cached_api = AsyncCachedOpenAIAPI(aclient=aclient, cache=cache, limiter=limiter)
    
    
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": "Give me 10 comma separated random numbers between 0 and 100."
            }
        ],
        "model": "gpt-3.5-turbo",
    }
    
    response_cached = await cached_api.create_chat_completion(prompt)
```

Please refer to `examples/async_openai_wrapper.py` for a comprehensive example.

This example relies on a simple convenience wrapper. The `cachesaver` package itself is agnostic to the backend and can wrap any object that offers a `request(prompt)` method. Convenience wrappers for huggingface are work-in-progress, if you need them now, please open an issue or take a look at the code in `src/cachesaver/thirdparty_wrappers/openai_wrapper.py`, for inspiration how you can implement your own.

### Idiomatic usage

Here's pseudocode that shows how we use cachesaver for internal projects. 
```python

async def process_sample(api, sample, config):
    # this function contains your algorithm
    # one important detail: we need you to set up namespace for this run of the algorithm
    # do this once, at the top of this function, reuse the same namespace in all requests
    namespace = make_unique_name(sample, hyperparameters)
    
    # you can set up complicated sequences of prompts, parse replies, branch, loop, etc
    # just use the api like this
    reply = await api.request(prompt, namespace=namespace)
    
    # finally return the result for this sample
    return result

async def process_dataset(api, dataset, config):
    # one ablation run, applies the algorithm to all samples in the dataset, for one specific config
    coroutines = []
    for sample in dataset:
        coroutines.append(process_sample(api, sample, config))
    
    return await asyncio.gather(*coroutines)

async def main():
    
    # prepare the evaluation data and hyperparameter ablations
    dataset = [...]
    configs = [...]
    
    # set up the model you want to cache
    aclient = AsyncOpenAI()
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    cache = Cache("cache_dir")
    with cache:
        
        limiter = AsyncRoundRobinLimiter()
        for _ in range(4):
            limiter.add_resource(OPENAI_KEY)
    
        cached_api = AsyncCachedOpenAIAPI(aclient=aclient, cache=cache, limiter=limiter)
        
        coroutines = []
        for config in configs:
            coroutines.append(process_dataset(cached_api, dataset, config))
        
        results = await asyncio.gather(*coroutines)
    
    # do something with the results

```

The caching saves data across runs. This makes your experiments fail-safe, as you can always resume from where you left off.

If you're running ablation studies, there might be prompts that are reused across different hyperparameter configurations. Our namespacing mechanism makes sure that within a namespace, all samples are independent. But if you reuse the same prompt across different namespaces, the samples will be recycled. This vastly reduces the cost of running ablation studies.

If you're using a local model, your cost efficiency depends on collecting prompts into batches. Let's say your algorithm is a sequence of prompts 1, 2, 3 that all depend on each other: The reply to prompt 1 is parsed and used in prompt 2, the reply to prompt 2 is parsed and used in prompt 3, and so on.
This makes it impossible to batch the prompts of one run of the algorithm. You can redesign your algorithm to work multiple samples at once, but this is often not feasible. You may have loops and branches in your code.
With cachesaver you don't have to worry about this. You can simply send sequential requests to the `api` in your `process_sample` code and await the result. The cachesaver will collect requests until a batch is full and then process them efficiently. This completely decouples the design of your algorithm from the efficiency of the request processing.

### Rate Limiting, Resource Management

When using a thirdparty API you need to be careful not to run into rate-limiting errors. The example code above will create `D x C` requests almost immediately (where `D` is the number of samples in the dataset and `C` is the number of configurations).

Our caching API is built around asynchronous limiters and will gate requests like this:
```
# code taken from the openai thirdparty wrapper
async with self.limiter as resource:
    self.aclient.api_key = resource.data
```

For now we offer only a simple round robin limiter. You can hand over your API keys as resources to the limiter and there will be only one request per resource.
In this example, at most 4 concurrent requests are sent to OpenAI:
```
limiter = AsyncRoundRobinLimiter()
for _ in range(4):
    limiter.add_resource(OPENAI_KEY)

```

### Compatibility, integrating your own model

Let's say you have an object that offers a `request(prompt)` method. We offer a simple set of wrapper classes that can
make your use of this object more efficient.

All our wrappers also expose a `request(prompt)` method, so you can mix and match them as you need (and easily add your
own).

We assume `request` has a couple of extra parameters, but they have default values, so you don't have to worry about
them:

```python
def request(prompt, n, namespace='default', request_id=-1):
    """
    :param prompt: the prompt to be processed
    :param n: the number of completions desired
    :param namespace: the namespace of the request
    :param request_id: the id of the request
    """
```

A typical pipeline would look like this:
 - Batcher: Collects requests until a batch is full or a timeout occurs
 - Deduplicator: Ensures that if the same prompt is sent multiple times, only one request is sent to the backend
 - Cacher: Reuses computations wherever possible
 - Your model: The object that offers a `request(prompt)` method

Our pipeline stages will use the `namespace` and `request_id` parameters to ensure that requests are processed correctly. But your model can just ignore them and simply call into `model.generate`, `openai.client.create_chat_completion` or any other computationally expensive API.

### TBD: Resource managers
When using thirdparty models, you may need to limit the number of concurrent requests, to avoid rate limiting errors. We offer a simple resource manager mechanism that can cycle through API keys and limit the number of concurrent requests per key. Please refer to the `src/cachesaver/async_engine/resource_managers` module. This is used in the openai thirdparty wrapper.

As of now we offer an unlimited resource manager which always immediately sends a request, as well as a round-robin scheduler which has a fixed number of resources, sends just one request per resource, and then cycles through the resources.

A smart, rate-limiting implementation of the leaky-bucket algorithm is work-in-progress.
