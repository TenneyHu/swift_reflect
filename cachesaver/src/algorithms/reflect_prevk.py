import random
import os
import logging
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED, StateReturningAgent
from ..utils import Resampler, log_states, log_agents
logger = logging.getLogger('reflect_prevk_logger')
logger.setLevel(logging.INFO)
# handler = logging.FileHandler('logs/reflect_prevk_logs.log')
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)


class AgentDictReflectPrevK(TypedDict):
    step: Agent
    evaluate: Agent
    step_params: DecodingParameters
    eval_params: DecodingParameters
    
def wrap_agent_in_env(agent_class, env):
    class WrappedAgent(agent_class, StateReturningAgent):
        @staticmethod
        async def act(model: Model, state: State, n: int, namespace: str, request_id: str, params: DecodingParameters):
            actions = await agent_class.act(model=model, state=state, n=n, namespace=namespace, request_id=request_id, params=params)
            new_states = [env.step(state, action) for action in actions]
            return new_states


class AlgorithmReflectPrevK(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictReflectPrevK,
                 env: Environment,
                 num_steps: int,
                 origin: float,
                 min_steps:int,
                 num_evaluations: int,
                 k: int,
                 ):
        super().__init__(model, agents, env)
        
        self.step_agent = agents["step"]
        self.eval_agent = agents["evaluate"]
        
        self.step_params = agents["step_params"]
        self.eval_params = agents["eval_params"]
        
        self.num_steps = num_steps
        self.origin = origin
        self.min_steps = min_steps
        self.num_evaluations = num_evaluations
        self.k = k
        if isinstance(env, type):
            self.task_name = env.__module__.split('.')[-2]
        else:
            self.task_name = env.__class__.__module__.split('.')[-2]
        
        log_file = 'logs/reflect_prevk.log'
        if logger.hasHandlers():
            logger.handlers.clear()

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        logger.info(50*'#')

    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        state = state.clone(randomness=random.randint(0, MAX_SEED))
        
        logger.info(f'reflect_prevk_logs-{self.task_name}-{idx}-fleet: {log_agents([{"agent": self.step_agent, "params": self.step_params, "num_agents": 1}])}')
        
        print('initial problem state:')
        print(state.puzzle)
        
        solved = False
        
        for step in range(self.num_steps):
            print(f"Step {step} ({idx})")
            
            if solved:
                print(f"Problem ({idx}) solved at step {step}")
                break
                
            logger.info(f"reflect_prevk_logs-{self.task_name}-{idx}-{step}-agentinputs: {log_states([state])}")
            
            # The agent returns a list of states.
            new_states = await self.step_agent.act(
                model=self.model,
                state=state,
                n=1,
                namespace=namespace,
                k=self.k,
                request_id=f"idx{idx}-step{step}-{hash(state)}",
                params=self.step_params
            )
            
            if not new_states:
                logger.warning(f"Agent returned no new states for index {idx}, step {step}. Stopping.")
                break
            state = new_states[0]
            
            logger.info(f"reflect_prevk_logs-{self.task_name}-{idx}-{step}-agentouts: {log_states(new_states)}")
            logger.info(f"reflect_prevk_logs-{self.task_name}-{idx}-{step}-statewins: {[self.env.evaluate(s)[1] == 1 for s in new_states]}")
            logger.info(f"reflect_prevk_logs-{self.task_name}-{idx}-{step}-statefails: {[self.env.is_final(s) for s in new_states]}")
            logger.info(f"reflect_prevk_logs-{self.task_name}-{idx}-{step}-reflections: {[len(s.reflections) for s in new_states]}")
            
            if self.env.evaluate(state)[1] == 1:
                solved = True
                print(f"Problem ({idx}) solved at step {step}")
                break
        return [state]
    
    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark-{index}",
                value_cache=None
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results
    