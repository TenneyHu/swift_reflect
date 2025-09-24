import random
import os
import logging
import asyncio
from typing import TypedDict
from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED, StateReturningAgent
from ..utils import Resampler, log_states, log_agents
from ..tasks.hotpotqa.state import StateHotpotQA
logger = logging.getLogger(__name__)

class AgentDictReflectPrevK(TypedDict):
    react_agent: Agent  # The agent that attempts to solve, e.g., AgentReactHotpotQA
    reflect_agent: Agent # The agent that generates reflections, e.g., AgentReflectHotpotQA
    # Optional: evaluate_agent for intermediate checks, but Reflexion often checks env.evaluate
    react_params: DecodingParameters
    reflect_params: DecodingParameters

class AlgorithmReflectPrevK(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictReflectPrevK,
                 env: Environment,
                 num_trials: int,         # Max number of attempts (trials)
                 max_steps_per_trial: int, # Max steps within each attempt
                 k: int                     
                ):
        super().__init__(model, agents, env)

        self.react_agent = agents["react_agent"]
        self.reflect_agent = agents["reflect_agent"]
        self.react_params = agents["react_params"]
        self.reflect_params = agents["reflect_params"]

        self.num_trials = num_trials
        self.k = k
        self.max_steps_per_trial = max_steps_per_trial

    async def solve(self, idx: int, initial_state: StateHotpotQA, trial: int, max_trial: int, namespace: str, value_cache: dict = None):
        """
        Attempts to solve the puzzle over a number of trials, using reflections.
        value_cache is not directly used by this specific Reflexion loop logic,
        but kept for consistency if agents use it.
        """
        is_final, reward = self.env.evaluate(initial_state)
        if is_final:
            if reward == 1.0:
                logger.info(f"Task {idx} Already solved.")
                return (idx, initial_state)

        trial_state: StateHotpotQA = initial_state.clone(
            randomness=random.randint(0, MAX_SEED),
            reset_trajectory=True,  # Reset steps for new trial
            new_trials=initial_state.trials + 1 if initial_state.trials is not None else 1
        )

        for step in range(self.max_steps_per_trial):
            logger.info(f"  Step {step + 1}/{self.max_steps_per_trial} (Task {idx}, Trial {trial}/{max_trial})")
                
            action_list = await self.react_agent.act(
                model=self.model,
                state=trial_state,
                n=1, 
                namespace=namespace,
                request_id=f"idx{idx}-trial{trial}-step{step}-{hash(trial_state)}",
                params=self.react_params
            )
                
            if not action_list:
                logger.warning(f"Task {idx}, Trial {trial}, Step {step}: React agent returned no action.")
                break # End current trial step if no action

            action = action_list[0]
                
            # Execute the action
            trial_state = self.env.step(trial_state, action)
                
            # Check for solution
            is_final, reward = self.env.evaluate(trial_state)
            if is_final:
                if reward == 1.0:
                    logger.info(f"Task {idx}, Trial {trial}: Solved successfully.")
                    return (idx, trial_state) 
                else:
                    logger.info(f"Task {idx}, Trial {trial}: Reached Finish action, but incorrect.")
                    break 
            
        if trial < max_trial:
            reflection_text = await self.reflect_agent.act(
                model=self.model,
                state=trial_state, # Pass the state that includes the failed trajectory
                namespace=namespace,
                request_id=f"idx{idx}-trial{trial}-reflect-{hash(trial_state)}",
                params=self.reflect_params
            )
            
            trial_state = trial_state.clone(new_reflection=reflection_text, prev_k=self.k)
            logger.info(f"Task {idx}, Trial {trial  }: New Reflection: {reflection_text}, length of reflections: {len(trial_state.reflections)}")
        else:
            logger.info(f"Task {idx}: All {self.num_trials} trials completed. Puzzle not solved.")
  
        return (idx, trial_state)
       

    async def benchmark(self, benchmark: Benchmark, trial: int, max_trial: int, share_ns: bool = False, cache: bool = True):
        # `cache` here refers to the value_cache for evaluators, not used directly by solve's loop
        # but passed down in case agents use it.
        value_cache_instance = {} if cache else None
        
        solve_coroutines = []
        for index, state in benchmark:

            solve_coroutines.append(
                self.solve(
                    idx=index,
                    initial_state=state, # Pass the initial state from the benchmark
                    trial=trial,
                    max_trial=max_trial,
                    namespace="benchmark" if share_ns else f"benchmark-{index}",
                    value_cache=value_cache_instance # Pass the cache if agents need it
                )
            )
        
        results = await asyncio.gather(*solve_coroutines)
        # `results` will be a list of lists of states. Typically, each inner list will have one state.
        return results