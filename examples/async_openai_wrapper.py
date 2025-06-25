import asyncio
import os
import shutil
import tempfile

from diskcache import Cache
from openai import AsyncOpenAI

from cachesaver.thirdparty_wrappers.openai_wrapper import AsyncCachedOpenAIAPI
from cachesaver.async_engine.resource_managers import AsyncRoundRobinLimiter

async def main():
    prompt = {
        "messages": [
            {
                "role": "user",
                "content": "Give me 10 comma separated random numbers between 0 and 100."
            }
        ],
        "model": "gpt-3.5-turbo",
        "n": 1,
        "temperature": 1.0,
        "max_tokens": 100,
        "top_p": 1.0,
        "frequency_penalty": 0.0,
        "presence_penalty": 0.0,
    }

    # using the original openai API
    aclient = AsyncOpenAI()
    # response_original = await aclient.chat.completions.create(**prompt)
    # print(response_original.choices[0].message.content)

    # using the cached openai API
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")
    cache_dir = tempfile.mkdtemp()
    cache = Cache(cache_dir)

    with cache:

        # limiters can be used to avoid rate limiting errors
        # for local models this is not relevant, but when using a third-party API it is important to constrain the
        # number of concurrent requests
        # for now we offer a simple round robin scheduler
        limiter = AsyncRoundRobinLimiter()
        for _ in range(4):
            limiter.add_resource(OPENAI_KEY)

        cached_api = AsyncCachedOpenAIAPI(aclient=aclient, cache=cache, limiter=limiter)

        response_cached = await cached_api.create_chat_completion(prompt)
        print(response_cached.choices[0].message.content)

        # the cached API reuses samples across runs, within one run, you'll always get new data
        response_cached = await cached_api.create_chat_completion(prompt)
        # this will probably look different from the first reply
        print(response_cached.choices[0].message.content)

        # the caching mechanism needs a refactored cost tracking
        # instead of token usage associated with individual replies, we now track tokens on an api level
        num_requests_sent = len(cached_api.usages)
        print(f"There have been {num_requests_sent} requests made.")

        # to re-use samples within one run, we support namespaces
        # within a namespace, samples are iid
        # across namespaces, the caching mechanism may reuse samples
        # the namespace A is new, so the cache can use the samples already created above,
        # no new requests to openai are sent
        prompt["n"] = 2
        response_A = await cached_api.create_chat_completion(prompt, namespace="A")
        assert num_requests_sent == len(cached_api.usages)
        print(f"No additional requests have been made.")

        # if a new namespace is set up and asks for 3 samples, we'll have to send an openai request
        prompt["n"] = 3
        response_B = await cached_api.create_chat_completion(prompt, namespace="B")
        assert len(cached_api.usages) == 3
        print(f"One additional request has been made.")

        # the prompt says n=3, so we expect 3 iid responses
        # the first two samples that we've created (and that were used within namespace A) are not used in B yet
        # so we need only one new sample, our api realizes this and optimizes the query
        # the first two samples will be reused
        assert response_B.choices[0].message.content == response_A.choices[0].message.content
        assert response_B.choices[1].message.content == response_A.choices[1].message.content

        # the async API can handle batches of prompts
        # as soon as a batch is full, it's sent off to the API
        # there's also a timeout to prevent waiting indefinitely
        cached_api.batcher.batch_size = 2
        num_requests = len(cached_api.usages)
        coroutines = []
        del prompt["n"]
        for i in range(2):
            coroutines.append(cached_api.create_chat_completion({**prompt, "n": i + 1}, namespace="B"))
        _ = await asyncio.gather(*coroutines)

        assert len(cached_api.usages) == num_requests + 1

    shutil.rmtree(cache_dir)


if __name__ == "__main__":
    asyncio.run(main())
