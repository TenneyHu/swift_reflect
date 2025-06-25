# src/algorithms/reflexion_react.py
import random
import logging
import asyncio
from typing import TypedDict, List

from ..typedefs import Algorithm, Model, Agent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from ..tasks.hotpotqa.state import StateHotpotQA # Assuming StateHotpotQA is the relevant state type

logger = logging.getLogger(__name__)

class AgentDictReflexionReact(TypedDict):
    react_agent: Agent  # The agent that attempts to solve, e.g., AgentReactHotpotQA
    reflect_agent: Agent # The agent that generates reflections, e.g., AgentReflectHotpotQA
    # Optional: evaluate_agent for intermediate checks, but Reflexion often checks env.evaluate
    react_params: DecodingParameters
    reflect_params: DecodingParameters

class AlgorithmReflexionReact(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictReflexionReact,
                 env: Environment,
                 num_trials: int,         # Max number of attempts (trials)
                 max_steps_per_trial: int # Max steps within each attempt
                ):
        super().__init__(model, agents, env)

        self.react_agent = agents["react_agent"]
        self.reflect_agent = agents["reflect_agent"]
        self.react_params = agents["react_params"]
        self.reflect_params = agents["reflect_params"]

        self.num_trials = num_trials
        self.max_steps_per_trial = max_steps_per_trial

    async def solve(self, idx: int, initial_state: StateHotpotQA, namespace: str, value_cache: dict = None) -> List[StateHotpotQA]:
        """
        Attempts to solve the puzzle over a number of trials, using reflections.
        value_cache is not directly used by this specific Reflexion loop logic,
        but kept for consistency if agents use it.
        """
        current_puzzle_state: StateHotpotQA = initial_state.clone(randomness=random.randint(0, MAX_SEED))
        
        all_trial_final_states = []

        for trial in range(self.num_trials):
            logger.info(f"Task {idx}, Trial {trial + 1}/{self.num_trials}")
            print(f"Task {idx}, Trial {trial + 1}/{self.num_trials} | Reflections: {len(current_puzzle_state.reflections)}")

            # State for the current trial, starts fresh except for reflections
            trial_state = current_puzzle_state.clone(
                randomness=random.randint(0, MAX_SEED),
                reset_trajectory=True # Clears current_state and steps, keeps reflections
            )

            for step in range(self.max_steps_per_trial):
                print(f"  Step {step + 1}/{self.max_steps_per_trial} (Task {idx}, Trial {trial + 1})")
                
                # Generate action using the react agent (which is reflection-aware)
                action_list = await self.react_agent.act(
                    model=self.model,
                    state=trial_state,
                    n=1, # Reflexion typically generates one trajectory
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
                        logger.info(f"Task {idx}, Trial {trial + 1}: Solved successfully.")
                        return [trial_state] # Return the single successful state
                    else:
                        logger.info(f"Task {idx}, Trial {trial + 1}: Reached Finish action, but incorrect.")
                        break # End current trial, proceed to reflection

            # End of trial (either max steps reached or Finish action led to incorrect answer)
            all_trial_final_states.append(trial_state) # Store the final state of this trial

            is_final, reward = self.env.evaluate(trial_state)
            if reward == 1.0: # Should have been caught above, but as a safeguard
                 logger.info(f"Task {idx}, Trial {trial + 1}: Verified solved at end of trial.")
                 return [trial_state]

            # If not the last trial and not solved, generate reflection
            if trial < self.num_trials - 1:
                logger.info(f"Task {idx}, Trial {trial + 1}: Failed. Generating reflection.")
                # The 'trial_state' contains the full trajectory (current_state, steps) of the failed trial
                reflection_text = await self.reflect_agent.act(
                    model=self.model,
                    state=trial_state, # Pass the state that includes the failed trajectory
                    namespace=namespace,
                    request_id=f"idx{idx}-trial{trial}-reflect-{hash(trial_state)}",
                    params=self.reflect_params
                )
                logger.info(f"Task {idx}, Trial {trial + 1}: Reflection: {reflection_text}")

                # Update current_puzzle_state with the new reflection for the next trial
                # It already contains previous reflections.
                current_puzzle_state = current_puzzle_state.clone(new_reflection=reflection_text)
            else:
                logger.info(f"Task {idx}: All {self.num_trials} trials completed. Puzzle not solved.")

        # If never solved, return all final states from each trial (or just the last one)
        # Depending on what `benchmark` expects, you might return only trial_state (last one)
        # or a list of all states achieved at the end of each trial.
        # For now, let's return the state from the last trial, as it's the most "evolved".
        if all_trial_final_states:
            return [all_trial_final_states[-1]]
        return [initial_state] # Fallback, should not happen if trials > 0

    async def benchmark(self, benchmark: Benchmark, share_ns: bool = False, cache: bool = True):
        # `cache` here refers to the value_cache for evaluators, not used directly by solve's loop
        # but passed down in case agents use it.
        value_cache_instance = {} if cache else None
        
        solve_coroutines = []
        for index, state in benchmark:
            # Ensure the state passed to solve is of the correct type (e.g., StateHotpotQA)
            # The benchmark might yield generic State, so cast or ensure compatibility.
            if not isinstance(state, StateHotpotQA):
                # This might require adapting how BenchmarkHotpotQA creates states or how StateHotpotQA is structured
                # For now, assuming benchmark yields StateHotpotQA instances.
                logger.error(f"Benchmark item {index} is not of type StateHotpotQA. Skipping.")
                # Or, attempt a conversion if possible and meaningful.
                # For simplicity, we'll assume it's correct for now.
                pass

            solve_coroutines.append(
                self.solve(
                    idx=index,
                    initial_state=state, # Pass the initial state from the benchmark
                    namespace="benchmark" if share_ns else f"benchmark-{index}",
                    value_cache=value_cache_instance # Pass the cache if agents need it
                )
            )
        
        results = await asyncio.gather(*solve_coroutines)
        # `results` will be a list of lists of states. Typically, each inner list will have one state.
        return results