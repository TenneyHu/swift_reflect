# src/algorithms/reflexion_wrapper.py
import random
import logging
import asyncio
from typing import TypedDict, List, Type, Dict, Any

from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from ..tasks.hotpotqa.state import StateHotpotQA 

logger = logging.getLogger(__name__)

class AgentDictReflexionWrapper(TypedDict):
    """ TypedDict for the agents used directly by the ReflexionWrapper. """
    reflect_agent_class: Type[Agent] 
    reflect_params: DecodingParameters

class AlgorithmReflexionWrapper(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictReflexionWrapper,
                 env: Environment,
                 inner_solver_class: Type[Algorithm],
                 inner_solver_agents_instances: TypedDict,
                 inner_solver_main_config: Dict[str, Any],
                 num_trials: int
                ):
        super().__init__(model, agents, env) 

        self.reflect_agent_class = agents["reflect_agent_class"]
        self.reflect_params = agents["reflect_params"]
        self.model_for_reflection = model 
        self.inner_solver = inner_solver_class(
            model=model,
            agents=inner_solver_agents_instances,
            env=env,
            **inner_solver_main_config
        )
        self.num_trials = num_trials

    async def solve(self, idx: int, initial_state: StateHotpotQA, namespace: str, value_cache: dict = None) -> List[StateHotpotQA]:
        current_puzzle_state: StateHotpotQA = initial_state.clone(randomness=random.randint(0, MAX_SEED))
        
        overall_best_state_from_inner_solver: StateHotpotQA = initial_state 
        max_overall_reward = -float('inf') 
        is_initial_final, initial_reward = self.env.evaluate(initial_state)
        if is_initial_final and initial_reward == 1.0:
            logger.info(f"ReflexionWrapper: Task {idx}: Initial state already solved.")
            return [initial_state]
        if initial_reward > max_overall_reward: 
            max_overall_reward = initial_reward
            overall_best_state_from_inner_solver = initial_state

        for trial in range(self.num_trials):
            logger.info(f"ReflexionWrapper: Task {idx}, Trial {trial + 1}/{self.num_trials}. Current reflections: {len(current_puzzle_state.reflections)}")
            print(f"ReflexionWrapper: Task {idx}, Trial {trial + 1}/{self.num_trials}. Reflections: {len(current_puzzle_state.reflections)}")

            state_for_inner_solver_trial = current_puzzle_state.clone(
                randomness=random.randint(0, MAX_SEED),
                reset_trajectory=True 
            )
            
            inner_solver_resulting_states: List[StateHotpotQA] = await self.inner_solver.solve(
                idx=idx, 
                state=state_for_inner_solver_trial, 
                namespace=f"{namespace}-inner_trial{trial}",
                value_cache=value_cache 
            )

            current_trial_best_state_from_inner_solver = state_for_inner_solver_trial 
            current_trial_max_reward_from_inner_solver = -float('inf')

            if not inner_solver_resulting_states:
                logger.warning(f"ReflexionWrapper: Task {idx}, Trial {trial + 1}: Inner solver returned no states.")
                current_trial_best_state_from_inner_solver = state_for_inner_solver_trial
                _, current_trial_max_reward_from_inner_solver = self.env.evaluate(state_for_inner_solver_trial)

            else:
                for s_res in inner_solver_resulting_states:
                    is_final, reward = self.env.evaluate(s_res)
                    if is_final and reward == 1.0:
                        logger.info(f"ReflexionWrapper: Task {idx}, Trial {trial + 1}: Solved successfully by inner solver.")
                        return [s_res] 
                    if reward > current_trial_max_reward_from_inner_solver:
                        current_trial_max_reward_from_inner_solver = reward
                        current_trial_best_state_from_inner_solver = s_res
            
            if current_trial_max_reward_from_inner_solver > max_overall_reward:
                max_overall_reward = current_trial_max_reward_from_inner_solver
                overall_best_state_from_inner_solver = current_trial_best_state_from_inner_solver

            if trial < self.num_trials - 1:
                logger.info(f"ReflexionWrapper: Task {idx}, Trial {trial + 1}: Inner solver best reward {current_trial_max_reward_from_inner_solver:.2f}. Generating reflection.")
                reflection_text = await self.reflect_agent_class.act(
                    model=self.model_for_reflection,
                    state=current_trial_best_state_from_inner_solver, 
                    namespace=namespace,
                    request_id=f"idx{idx}-reflectwrapper-trial{trial}-{hash(current_trial_best_state_from_inner_solver)}",
                    params=self.reflect_params
                )
                logger.info(f"ReflexionWrapper: Task {idx}, Trial {trial + 1}: Reflection: {reflection_text}")
                
                current_puzzle_state = current_puzzle_state.clone(new_reflection=reflection_text)
            elif trial == self.num_trials - 1 :
                logger.info(f"ReflexionWrapper: Task {idx}: All {self.num_trials} trials completed. Best overall reward: {max_overall_reward:.2f}")

        return [overall_best_state_from_inner_solver]

    async def benchmark(self, benchmark: Benchmark, share_ns: bool = False, cache: bool = True):
        value_cache_instance = {} if cache else None
        
        solve_coroutines = [
            self.solve(
                idx=index,
                initial_state=state, # Make sure state is StateHotpotQA
                namespace="benchmark" if share_ns else f"benchmark-{index}",
                value_cache=value_cache_instance
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results