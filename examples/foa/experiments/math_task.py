import asyncio
import os
import random
import tempfile

from datasets import load_dataset
from diskcache import Cache
from openai import AsyncAzureOpenAI

from foa.agents.math_task import MathAgent
from foa.states.math_task import MathState
from examples.foa.utils import extract_last_boxed_answer
from cachesaver.thirdparty_wrappers.openai_wrapper import AsyncCachedOpenAIAPI


async def foa(problem, api, foa_options, namespace):
    num_agents = foa_options["num_agents"]
    k = foa_options["k"]
    num_steps = foa_options["num_steps"]
    gamma = foa_options["gamma"]

    states = []
    for i in range(num_agents):
        states.append(
            MathState(question=problem["problem"], reasoning_chain="", reference_solution=problem["solution_num"],
                      randomness=i))

    # list of state2value dicts
    state2value_at_t = []

    for step in range(num_steps):
        # mutation phase
        coroutines = []
        for idx, state in enumerate(states):
            coroutines.append(MathAgent.step(state, api, namespace, id=idx))
        states = await asyncio.gather(*coroutines)

        # check for solutions
        # we're creating a list "solved" here with entries being True/False
        # this list is also used to return from the function in the end
        coroutines = []
        for idx, state in enumerate(states):
            coroutines.append(MathAgent.evaluate(state, namespace, id=idx))
        solved = await asyncio.gather(*coroutines)

        if any(solved):
            break

        if step % k != 0:
            continue

        # selection phase

        # evaluate the states
        coroutines = []
        unique_states = list(set(states))
        for idx, state in enumerate(unique_states):
            coroutines.append(MathAgent.value(state, api, namespace, id=idx))
        values = await asyncio.gather(*coroutines)

        state2value = dict(zip(unique_states, values))
        state2value_at_t.append(state2value)

        # discount past values by gamma
        state2value_discounted = {}
        t = 0
        for state2value in state2value_at_t[::-1]:
            for state, value in state2value.items():
                state2value_discounted[state] = gamma ** t * value

            t += 1

        # convert to probability distribution
        all_states = list(state2value_discounted.keys())
        all_values = list(state2value_discounted.values())

        # ToDo: here we can plug in different selection strategies
        probas = [v / sum(all_values) for v in all_values]

        # sample new states
        new_states = random.choices(all_states, probas, k=num_agents)

        states = new_states

    # time's up or we've found an early solution
    return any(solved)


async def run_configuration(dataset, api, foa_options):
    """
    runs foa on all problems in the dataset
    :param tasks:
    :param foa_options:
    :return:
    """

    # run foa on all problems
    coroutines = []
    for problem in dataset:
        unique_name = problem["problem"] + problem["solution"]
        coroutines.append(foa(problem, api, foa_options, unique_name))

    results = await asyncio.gather(*coroutines)

    return results


async def main(dataset):
    foa_options = {
        "num_agents": 1,
        "k": 2,
        "num_steps": 20,
        "gamma": 0.
    }

    # set up the original API
    # aclient = AsyncOpenAI()

    # using the cached openai API
    OPENAI_KEY = os.getenv("OPENAI_API_KEY")

    # Create OpenAI client
    aclient = AsyncAzureOpenAI(
        api_key=OPENAI_KEY,
        azure_endpoint="https://key-1-18k-loc2.openai.azure.com",
        api_version="2024-02-15-preview"
    )

    cache_dir = tempfile.mkdtemp()
    cache_dir = "./cache/foa_math_cache"
    cache = Cache(cache_dir)

    with cache:
        cached_api = AsyncCachedOpenAIAPI(aclient=aclient, cache=cache, private_key=OPENAI_KEY)

        results = await run_configuration(dataset, cached_api, foa_options)
    print(f"Results: {results}")


if __name__ == "__main__":
    # Load dataset
    dataset = load_dataset("hendrycks/competition_math", split='test', trust_remote_code=True)

    # Extract the numerical solution
    dataset = dataset.map(lambda x: {"solution_num": extract_last_boxed_answer(x["solution"])})

    # drop the first 5 entries, they are used in prompts
    dataset = dataset.select(range(5, len(dataset)))

    num_test = 1
    dataset = dataset.train_test_split(test_size=num_test, shuffle=True, seed=42)
    val_dataset = dataset["train"]
    test_dataset = dataset["test"]
    print("starting")
    asyncio.run(main(test_dataset))
