import os
import re
import time
import asyncio
import logging
import argparse
from typing import Any, Dict, Type, TypedDict
import numpy as np
from diskcache import Cache
from openai import AsyncOpenAI
from omegaconf import OmegaConf
from together import AsyncTogether
from groq import AsyncGroq
logger = logging.getLogger(__name__)

import sys
sys.path.append(os.getcwd())
from cachesaver.pipelines import OnlineAPI
from src.algorithms.reflexion_wrapper import AlgorithmReflexionWrapper, AgentDictReflexionWrapper
from src.algorithms.react import AlgorithmReact, AgentDictReact
from src.utils import tokens2cost, clean_log
from src.algorithms import *
from src.models import OnlineLLM, API
from src.models import GroqAPILLM
from src.typedefs import Algorithm, DecodingParameters
from src.tasks.hotpotqa import *

def build_method(method_name: str, params: DecodingParameters, api: API, config: OmegaConf):
    
    agent_act_hpqa = AgentActHotpotQA()
    agent_bfs_hpqa = AgentBfsHotpotQA()
    agent_react_hpqa = AgentReactHotpotQA()
    agent_agg_hpqa = AgentAggregateHotpotQA()
    agent_eval_hpqa = AgentEvaluateHotpotQA()
    if method_name.startswith("reflexion_"):
        inner_method_name = method_name.split("reflexion_", 1)[1]
        
        inner_solver_class: Type[Algorithm]
        inner_solver_agents_dict: TypedDict # This will be AgentDictFOA, AgentDictTOT etc.
        inner_solver_main_config: Dict[str, Any]

        if inner_method_name == "foa":
            inner_solver_class = AlgorithmFOA
            inner_solver_agents_dict = AgentDictFOA(
                step=AgentActHotpotQA, evaluate=AgentEvaluateHotpotQA,
                step_params=params, eval_params=params)
            inner_solver_main_config = OmegaConf.to_container(config.foa, resolve=True)
        elif inner_method_name == "tot_bfs":
            inner_solver_class = AlgorithmTOT
            inner_solver_agents_dict = AgentDictTOT(
                step=AgentBfsHotpotQA, evaluate=AgentEvaluateHotpotQA,
                step_params=params, eval_params=params)
            inner_solver_main_config = OmegaConf.to_container(config.tot_bfs, resolve=True)
        elif inner_method_name == "got":
            inner_solver_class = AlgorithmGOT
            inner_solver_agents_dict = AgentDictGOT(
                step=AgentActHotpotQA, aggregate=AgentAggregateHotpotQA, evaluate=AgentEvaluateHotpotQA,
                step_params=params, aggregate_params=params, eval_params=params)
            inner_solver_main_config = OmegaConf.to_container(config.got, resolve=True)
        elif inner_method_name == "react":
            inner_solver_class = AlgorithmReact
            inner_solver_agents_dict = AgentDictReact(
                step=AgentReactHotpotQA, step_params=params)
            if not hasattr(config, 'react') or not hasattr(config.react, 'num_steps'):
                    raise ValueError(f"Config for 'react.num_steps' not found for inner method '{inner_method_name}'")
            inner_solver_main_config = OmegaConf.to_container(config.react, resolve=True)
        # elif inner_method_name == "rap":
        #     inner_solver_class = AlgorithmRAP
        #     inner_solver_agents_dict = AgentDictRAP(
        #         step=AgentActHotpotQA, # Or a specific RAP step agent
        #         evaluate=AgentEvaluateHotpotQA, # Or a specific RAP eval agent
        #         step_params=params, eval_params=params)
        #     inner_solver_main_config = OmegaConf.to_container(config.rap, resolve=True)
        else:
            raise NotImplementedError(f"Inner method '{inner_method_name}' for ReflexionWrapper not recognized.")

        wrapper_agents_config = AgentDictReflexionWrapper(
            reflect_agent_class=AgentReflectHotpotQA, # Pass the Agent CLASS
            reflect_params=params 
        )
        if not hasattr(config, 'reflexion_wrapper') or not hasattr(config.reflexion_wrapper, 'num_trials'):
            raise ValueError(f"Config for 'reflexion_wrapper.num_trials' not found for method '{method_name}'")
    
        method = AlgorithmReflexionWrapper(
            model=api,
            agents=wrapper_agents_config,
            env=EnvironmentHotpotQA,
            inner_solver_class=inner_solver_class,
            inner_solver_agents_instances=inner_solver_agents_dict, # Pass the prepared agent dict
            inner_solver_main_config=inner_solver_main_config,
            num_trials=config.reflexion_wrapper.num_trials
        )
    elif method_name == "het_foa":
        step_agents = []

        # build the fleet of agents here
        step_agents.append({
            "agent": AgentActHotpotQA,
            "params": params,
            "num_agents": config.het_foa.num_agents - config.het_foa.num_agents // 2,
        })

        step_agents.append({
            "agent": AgentReactHotpotQA,
            "params": params,
            "num_agents": config.het_foa.num_agents // 2,
        })

        # step_agents.append({
        #     "agent": AgentTerminalReflexionGame24,
        #     "params": params,
        #     "num_agents": 2,
        # })

        agents = AgentDictHeterogenousFOA(
            evaluate=AgentEvaluateHotpotQA,
            eval_params=params,
            step_agents=step_agents
        )

        logger.info(f"Using these agents for Heterogenous FOA:")
        for i in range(len(agents["step_agents"])):
            logger.info(f"{step_agents[i]['agent'].__name__} ({agents['step_agents'][i]['num_agents']}): Temperature: {agents['step_agents'][i]['params'].temperature}, Top P: {agents['step_agents'][i]['params'].top_p}")


        method = AlgorithmHeterogenousFOA(
            model=api,
            agents=agents,
            env=EnvironmentHotpotQA,
            num_agents=config.het_foa.num_agents,
            num_steps=config.het_foa.num_steps,
            k=config.het_foa.k,
            backtrack=config.het_foa.backtrack,
            resampling=config.het_foa.resampling,
            origin=config.het_foa.origin,
            min_steps=config.het_foa.min_steps,
            num_evaluations=config.het_foa.num_evaluations,
        )

    elif method_name == "foa":
        agents = AgentDictFOA(step=AgentActHotpotQA, evaluate=AgentEvaluateHotpotQA, step_params=params, eval_params=params)
        method = AlgorithmFOA(model=api, agents=agents, env=EnvironmentHotpotQA, **OmegaConf.to_container(config.foa, resolve=True))
    elif method_name == "tot_bfs":
        agents = AgentDictTOT(step=AgentBfsHotpotQA, evaluate=AgentEvaluateHotpotQA, step_params=params, eval_params=params)
        method = AlgorithmTOT(model=api, agents=agents, env=EnvironmentHotpotQA, **OmegaConf.to_container(config.tot_bfs, resolve=True))
    elif method_name == "got":
        agents = AgentDictGOT(step=AgentActHotpotQA, aggregate=AgentAggregateHotpotQA, evaluate=AgentEvaluateHotpotQA, step_params=params, aggregate_params=params, eval_params=params)
        method = AlgorithmGOT(model=api, agents=agents, env=EnvironmentHotpotQA, **OmegaConf.to_container(config.got, resolve=True))
    elif method_name == "react":
        if not hasattr(config, 'react') or not hasattr(config.react, 'num_steps'):
             raise ValueError(f"Config for 'react.num_steps' not found for method '{method_name}'")
        agents = AgentDictReact(step=AgentReactHotpotQA, step_params=params)
        method = AlgorithmReact(model=api, agents=agents, env=EnvironmentHotpotQA, **OmegaConf.to_container(config.react, resolve=True))
    # Add other base methods (e.g., RAP)
    # elif method_name == "rap":
    #     agents = AgentDictRAP(step=AgentActHotpotQA, evaluate=AgentEvaluateHotpotQA, step_params=params, eval_params=params)
    #     method = AlgorithmRAP(model=api, agents=agents, env=EnvironmentHotpotQA, **OmegaConf.to_container(config.rap, resolve=True))
    else:
        raise NotImplementedError(f"Method {method_name} is not implemented yet.")
    return method

async def run(args, trial, cache_path):
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    cache = Cache(cache_path)

    if args.provider == "openai":
        client_params = {"base_url": args.base_url} if args.base_url else {}
        if args.base_url and ("localhost" in args.base_url or "127.0.0.1" in args.base_url):
            client_params["api_key"] = "dummy-key" # For local vLLM
        client = AsyncOpenAI(**client_params)
    elif args.provider == "together":
        client = AsyncTogether() # Assumes TOGETHER_API_KEY is in env
    elif args.provider == "groq":
        client = AsyncGroq() # Assumes GROQ_API_KEY is in env
    else:
        raise ValueError("Invalid provider. Choose 'openai' or 'together'. Local not fully supported yet.")
    

    if args.provider == 'groq':
        model_layer = OnlineLLM(client=client, max_n=1)
    else:
        model_layer = OnlineLLM(client=client)
    
    pipeline = OnlineAPI(
                    model=model_layer,
                    cache=cache,
                    batch_size=args.batch_size,
                    timeout=args.timeout,
                    allow_batch_overflow=True,
                    correctness=bool(args.correctness)
                    )
    
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
    method = build_method(args.method, params, api, config)

    # Load the dataset
    benchmark = BenchmarkHotpotQA(path=args.dataset_path, split=args.split)

    # Run the method
    start = time.time()
    results = await method.benchmark(
        benchmark=benchmark,
        share_ns=True,
        cache=args.value_cache,
    )
    end = time.time()

    finished = []
    correct = []
    for result in results:
        evaluations = sorted([EnvironmentHotpotQA.evaluate(state) for state in result], key=lambda x: x[1])
        finished.append(False if len(evaluations) == 0 else evaluations[-1][0])
        correct.append(1.0 if len(evaluations) == 0 else evaluations[-1][1])
    perc_finished = sum(finished) / len(finished)
    perc_correct = sum(correct) / len(correct)
    costs = {key:tokens2cost(api.tokens[key], args.model)["total"] for key in api.tokens.keys()}
    latency = {
        "mean": np.mean(api.latencies), 
        "std": np.std(api.latencies),
        "max": np.max(api.latencies), 
        "min": np.min(api.latencies), 
        "total": np.sum(api.latencies)
        }
    reuse = {
        "mean": np.mean(list(api.reuse.values())),
        "std": np.std(list(api.reuse.values())),
        "max": np.max(list(api.reuse.values())),
        "min": np.min(list(api.reuse.values())),
        "median": np.median(list(api.reuse.values())),
    }
    run_time = end - start
    throughput = len(benchmark) / run_time

    logger.info(f"Finished: {perc_finished:.2f} (trial {trial})")
    logger.info(f"Correct: {perc_correct:.2f} (trial {trial})")
    logger.info(f"Costs: {costs} (trial {trial})")
    logger.info(f"Latency: {latency['mean']} (trial {trial})")
    logger.info(f"Run time: {run_time:.2f} seconds (trial {trial})")
    logger.info(f"Throughput: {throughput:.2f} puzzles/second (trial {trial})")

    logger.info(f"Correct (deailed): {correct} (trial {trial})")
    logger.info(f"Tokens (detailed): {api.tokens} (trial {trial})")
    logger.info(f"Calls (detailed): {api.calls} (trial {trial})")
    logger.info(f"Reuse (detailed): {reuse} (trial {trial})")
    
    print("All good.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run HotPotQA experiments using LLMs.")
    parser.add_argument("--provider", type=str, default="openai", help="LLM provider (e.g., 'openai', 'together', 'groq')")
    parser.add_argument("--base_url", type=str, default=None, help="Base URL for the API (optional)")
    parser.add_argument("--model", type=str, default="gpt-4.1-nano", help="LLM model identifier")
    parser.add_argument("--batch_size", type=int, default=1, help="CacheSaver's batch size")
    parser.add_argument("--timeout", type=float, default=10.0, help="CacheSaver's timeout in seconds")
    parser.add_argument("--temperature", type=float, default=0.2, help="Temperature for the model")
    parser.add_argument("--max_completion_tokens", type=int, default=256, help="Max completion tokens")
    parser.add_argument("--top_p", type=float, default=1.0, help="Top P for the model")
    parser.add_argument("--stop", type=str, nargs="+", default=None, help="Stop sequence(s) for the model (e.g. Observation)")
    parser.add_argument("--logprobs", action="store_true", help="Enable logprobs for the model (required by some agents)")
    parser.add_argument("--dataset_path", type=str, default="./datasets/dataset_hotpotqa.csv.gz", help="Path to the HotPotQA dataset (CSV.GZ file)")
    parser.add_argument("--split", type=str, default="mini", help="Split of the dataset (e.g., 'mini', 'train', 'test')")
    parser.add_argument("--method", type=str, required=True, help="Method to use (e.g., 'foa', 'tot_bfs', 'got', 'reflexion_react')")
    parser.add_argument("--conf_path", type=str, default="./scripts/frameworks/hotpotqa/hotpotqa.yaml", help="Path to the YAML configuration file for method hyperparameters")
    parser.add_argument("--value_cache", action="store_true", help="Enable value caching in agents like Evaluate/SelfEvaluate")
    parser.add_argument("--correctness", type=int, default=0, help="CacheSaver: 0 for default, 1 for 'correct' original impl.")
    args = parser.parse_args()
    
    filename = f"logs/frameworks/{args.model.split('/')[-1]}/hotpotqa/{args.method}.log"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    logging.basicConfig(level=logging.INFO, filename=filename, filemode="a")
    logger.info("#"*50)

    # Load previous content
    with open(filename, "r") as f:
        contents = f.read()
    
    if args.batch_size == 1:
        previous_trials = [int(num) for num in re.findall(r"Shared Namespace \(trial (\d+)\)", contents)]
        trial = max(previous_trials) + 1 if previous_trials else 1
        logger.info(f"Shared Namespace (trial {trial})")
        cache_path = f"caches/frameworks/{args.method}/sns_{trial}"
    else:
        previous_trials = [int(num) for num in re.findall(r"Shared Namespace and Batch \(trial (\d+)\)", contents)]
        trial = max(previous_trials) + 1 if previous_trials else 1
        logger.info(f"Shared Namespace and Batch (trial {trial})")
        cache_path = f"caches/frameworks/hotpotqa"

    asyncio.run(run(args, trial=trial, cache_path=cache_path))
    logger.info("\n"*3)
    clean_log(filename)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    