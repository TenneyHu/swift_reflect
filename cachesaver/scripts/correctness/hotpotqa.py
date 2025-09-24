import os
import re
import time
import asyncio
import logging
import argparse
import numpy as np
from diskcache import Cache
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from together import AsyncTogether
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.getcwd())

from cachesaver.pipelines import OnlineAPI
from src.utils import tokens2cost, clean_log
from src.algorithms import *
from src.models import OnlineLLM, API
from src.typedefs import DecodingParameters
from src.tasks.hotpotqa import *
from dotenv import load_dotenv

load_dotenv()


def build_method(method_name: str, args, params: DecodingParameters, api: API, config: OmegaConf):
# Setup the method
    
    if method_name == "reflect_summary":
        agents = AgentDictReflectSummary(
            react_agent=AgentReactHotpotQA,
            reflect_agent=AgentReflectHotpotQA,
            summary_agent=AgentSummaryHotpotQA,
            react_params=params,
            reflect_params=params,
            summary_params=params,
        )
        method = AlgorithmReflectSummary(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_trials=args.trials,
            max_steps_per_trial=config.reflect_summary.max_steps_per_trial,
        )
    elif method_name == "reflexion_react":
        agents = AgentDictReflexionReact(
            react_agent=AgentReactHotpotQA,
            reflect_agent=AgentReflectHotpotQA,
            react_params=params,
            reflect_params=params,
        )
        method = AlgorithmReflexionReact(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_trials=args.trials,
            max_steps_per_trial=config.reflexion_react.max_steps_per_trial,
        )
    elif method_name == "reflect_prev_k":
        agents = AgentDictReflectPrevK(
            react_agent=AgentReactHotpotQA,
            reflect_agent=AgentReflectHotpotQA,
            react_params=params,
            reflect_params=params,
        )
        method = AlgorithmReflectPrevK(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_trials=args.trials,
            max_steps_per_trial=config.reflect_prev_k.max_steps_per_trial,
            k=config.reflect_prev_k.k,
        )

    else:
        raise NotImplementedError(f"Method {method_name} is not implemented yet.")
    return method

async def run(args, cache_path):
    # Cache to be used
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    # LLM Provider
    if args.provider == "openai":
        client = AsyncOpenAI()
    elif args.provider == "together":
        client = AsyncTogether()
    elif args.provider == "local":
        raise NotImplementedError("Local client is not implemented yet.")
    else:
        raise ValueError("Invalid provider. Choose 'openai', 'together', or 'local'.")
    
    # CacheSaver model layer
    if args.provider in ["openai", "together"]:
        model = OnlineLLM(client=client)
    else:
        raise NotImplementedError("Local model is not implemented yet.")
    
    # CacheSaver Pipeline: Batcher -> Reorderer -> Deduplicator -> Cache -> Model
    pipeline = OnlineAPI(
                    model=model,
                    cache=cache,
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    allow_batch_overflow=True,
                    correctness=bool(args.correctness)
                    )
    
    # Cachesaver additional layer for wrapping: API -> Pipeline
    api = API(
        pipeline=pipeline,
        model=args.model
    )

    # Decoding parameters
    params = DecodingParameters(
        temperature=args.temperature,
        max_completion_tokens=args.max_completion_tokens,
        top_p=args.top_p,
        stop=args.stop,
        logprobs=args.logprobs
    )

    # Config for framework hyperpaarameters
    config = OmegaConf.load(args.conf_path)

    # Build the method
    method = build_method(args.method, args, params, api, config)

    # Load the dataset
    current_state = BenchmarkHotpotQA(path=args.dataset_path, split=args.split)

    for trial in range(1, args.trials + 1):

        start = time.time()
        results = await method.benchmark(
            benchmark=current_state,
            share_ns=True,
            trial=trial,
            max_trial=args.trials,
            cache=args.value_cache,
        )
        end = time.time()
        current_state = results
        finished = []
        correct = []
        for _, result in results:
            evaluations = EnvironmentHotpotQA.evaluate(result)
            finished.append(False if len(evaluations) == 0 else evaluations[0])
            correct.append(1.0 if len(evaluations) == 0 else evaluations[1])

        perc_finished = sum(finished) / len(finished)
        perc_correct = sum(correct) / len(correct)
        costs = {key:tokens2cost(api.tokens[key], args.model)["total"] for key in api.tokens.keys()}
        run_time = end - start


        logger.info(f"Finished: {perc_finished:.2f} (trial {trial})")
        logger.info(f"Correct: {perc_correct:.2f} (trial {trial})")
        logger.info(f"Costs: {costs} (trial {trial})")
        logger.info(f"Correct (detailed): {correct} (trial {trial})")
        #logger.info(f"Run time: {run_time:.2f} seconds (trial {trial})")
        #logger.info(f"Tokens (detailed): {api.tokens} (trial {trial})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Solve HotpotQA using LLMs.")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (e.g., 'openai', 'together', 'groq')")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for the API (optional)")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="LLM model identifier")
    parser.add_argument("--batch_size", type=int, default=1, help="CacheSaver's batch size")
    parser.add_argument("--timeout", type=float, default=10.0, help="CacheSaver's timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for the model")
    parser.add_argument("--max_completion_tokens", type=int, default=256, help="Max completion tokens")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top P for the model")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials to run")
    parser.add_argument("--stop", type=str, nargs="+", default=None, help="Stop sequence(s) for the model (e.g. Observation)")
    parser.add_argument("--logprobs", action="store_true", help="Enable logprobs for the model (required by some agents)")
    parser.add_argument("--dataset_path", type=str, default="./datasets/dataset_hotpotqa.csv.gz", help="Path to the HotPotQA dataset (CSV.GZ file)")
    parser.add_argument("--split", type=str, default="mini", help="Split of the dataset (e.g., 'mini', 'train', 'test')")
    parser.add_argument("--method", type=str, required=True, help="Method to use (e.g., 'foa', 'tot_bfs', 'got', 'reflexion_react')")
    parser.add_argument("--conf_path", type=str, default="./scripts/frameworks/hotpotqa/hotpotqa.yaml", help="Path to the YAML configuration file for method hyperparameters")
    parser.add_argument("--value_cache", action="store_true", help="Enable value caching in agents like Evaluate/SelfEvaluate")
    parser.add_argument("--correctness", type=int, default=0, help="CacheSaver: 0 for default, 1 for 'correct' original impl.")
    args = parser.parse_args()

    filename = f"logs/correctness/{args.model.split('/')[-1]}/hotpotqa/{args.method}.log"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="a")
    logger.info("#"*50)

    # Load previous content
    with open(filename, "r") as f:
        contents = f.read()

    
    cache_path = f"caches/correctness/hotpotqa/{args.method}/snsb_1"
    asyncio.run(run(args, cache_path=cache_path))

    logger.info("\n"*3)
    clean_log(filename)
