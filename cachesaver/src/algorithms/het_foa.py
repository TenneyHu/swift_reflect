import random
import logging
import asyncio
from typing import TypedDict, Optional
from dataclasses import replace
from ..typedefs import Algorithm, Model, AgentDict, Agent, StateReturningAgent, ValueFunctionRequiringAgent, ValueFunctionUsingAgent, Environment, DecodingParameters, State, Benchmark, MAX_SEED
from ..utils import *
import os

logger = logging.getLogger('het_foa_logger')
logger.setLevel(logging.INFO)
# handler = logging.FileHandler('logs/het_foa_logs.log')
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)

DEBUG = False  # Set to True to enable debug prints


class AgentDictHeterogenousFOA(TypedDict):
    evaluate: Agent
    eval_params: DecodingParameters
    step_agents: list[AgentDict]



class AlgorithmHeterogenousFOA(Algorithm):
    def __init__(self,
                 model: Model,
                 agents: AgentDictHeterogenousFOA,
                 env: Environment,
                 num_agents: int,
                 num_steps: int,
                 k: int,
                 backtrack: float,
                 resampling: str,
                 origin: float,
                 min_steps: int,
                 num_evaluations: int
                 ):
        super().__init__(model, agents, env)

        self.eval_agent = agents["evaluate"]
        self.eval_params = agents["eval_params"]

        self.step_agents = agents["step_agents"]

        self.num_agents = num_agents
        self.num_steps = num_steps
        self.k = k
        self.backtrack = backtrack
        self.resampling = resampling
        self.origin = origin
        self.min_steps = min_steps
        self.num_evaluations = num_evaluations
        self.input_state_values = []

        log_file = 'logs/het_foa.log'
        for agent_dict in self.step_agents:
            agent_name = agent_dict['agent'].__name__
            if 'Reflect' in agent_name:
                log_file = 'logs/het_foa_with_reflect.log'
                break
        
        if logger.hasHandlers():
            logger.handlers.clear()

        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.INFO)
        logger.addHandler(handler)

        logger.info('#################################################################')


    def _get_ith_agent_dict(self, i) -> AgentDict:
        """Get the i-th agent in the fleet and its parameters."""
        if i >= self.num_agents or i < 0:
            raise IndexError(f"Agent index {i} out of range.")

        agent_dict_index = -1
        while i >= 0:
            agent_dict_index += 1
            i -= self.step_agents[agent_dict_index]["num_agents"]

        return self.step_agents[agent_dict_index]
    

    def _wrap_agent_in_env(self, agent_class):
        if issubclass(agent_class, StateReturningAgent):
            return agent_class
                
        class EnvWrappedAgent(agent_class, StateReturningAgent):
            @staticmethod
            async def act(model: Model, state: State, n: int, namespace: str, request_id: str, params: DecodingParameters):
                actions = await agent_class.act(model=model, state=state, n=n, namespace=namespace, request_id=request_id, params=params)
                new_states = [self.env.step(state, action) for action in actions]

                # TODO: REMOVE
                if DEBUG:
                    print('states returned by react agent:')
                    for s in new_states:
                        print(f"State {hash(s)}: steps={len(s.steps)}, value={s.value}, reflections={len(s.reflections)}")
                    input()

                return new_states
        
        EnvWrappedAgent.__name__ = f"EnvWrapped{agent_class.__name__}"
        EnvWrappedAgent.__qualname__ = f"EnvWrapped{agent_class.__qualname__}"
        EnvWrappedAgent.__module__ = agent_class.__module__
        EnvWrappedAgent.__doc__ = f"EnvWrapped version of {agent_class.__name__} to work with the environment."
        EnvWrappedAgent.__annotations__ = agent_class.__annotations__.copy()

        return EnvWrappedAgent


    def _wrap_agent_in_value_function(self, agent_class):
        if not issubclass(agent_class, ValueFunctionRequiringAgent) or issubclass(agent_class, ValueFunctionUsingAgent):
            return agent_class
                
        class ValueFunctionWrappedAgent(agent_class, ValueFunctionUsingAgent):
            @staticmethod
            async def act(model: Model, state: State, n: int, namespace: str, request_id: str, params: DecodingParameters):
                value_agent = AgentDict(
                    agent=self.eval_agent,
                    params=self.eval_params,
                    model=self.model,
                    num_agents=self.num_evaluations
                )

                return await agent_class.act(
                    model=model, 
                    state=state, 
                    n=n, 
                    namespace=namespace, 
                    request_id=request_id, 
                    params=params, 
                    value_agent=value_agent
                )
        
        ValueFunctionWrappedAgent.__name__ = f"ValueFunctionWrapped{agent_class.__name__}"
        ValueFunctionWrappedAgent.__qualname__ = f"ValueFunctionWrapped{agent_class.__qualname__}"
        ValueFunctionWrappedAgent.__module__ = agent_class.__module__
        ValueFunctionWrappedAgent.__doc__ = f"ValueFunctionWrapped version of {agent_class.__name__} to work with the value function."
        ValueFunctionWrappedAgent.__annotations__ = agent_class.__annotations__.copy()

        return ValueFunctionWrappedAgent


    def _wrap_agent(self, agent_class):
        agent_class = self._wrap_agent_in_env(agent_class)
        agent_class = self._wrap_agent_in_value_function(agent_class)
        return agent_class


    async def solve(self, idx: int, state: State, namespace: str, value_cache: dict = None):
        randomness = idx
        random.seed(randomness)
        resampler = Resampler(randomness)

        # set the value of inital state
        state = replace(state, value=self.origin*self.backtrack*self.backtrack)


        # log all agents
        logger.info(f'het_foa_logs-{idx}-fleet: {log_agents(self.step_agents)}')

        print('initial problem state:')
        print(state.puzzle)

        # Records of previously visited states (state_identifier, state_value, state)
        visited_states = [("INIT", self.origin, state)]

        # Initialize state for each agent
        states = [state.clone(randomness=random.randint(0, MAX_SEED)) for _ in range(self.num_agents)]

        # Wrap action agents in the environment
        for i in range(len(self.step_agents)):
            agent_class = self.step_agents[i]['agent']
            self.step_agents[i]['agent'] = self._wrap_agent(agent_class)
        
        solved = False
        for step in range(self.num_steps):
            print(f"Step {step} ({idx})")

            if DEBUG:
                print(f"Current resampled states:")
                for i, s in enumerate(states):
                    print(f"State {hash(s)}: steps={len(s.steps)}, value={s.value}, reflections={len(s.reflections)}")
                input()

            if solved:
                print(f'Problem ({idx}) solved at step {step}')
                break


            logger.info(f"het_foa_logs-{idx}-{step}-agentinputs: {log_states(states)}")

            # prepare data for backtracking agent
            self.input_state_values = [state.value for state in states]

            # Generate actions for each state
            agent_coroutines = [
                self._get_ith_agent_dict(i)['agent'].act(
                    model=self._get_ith_agent_dict(i).get('model') or self.model,
                    state=state,
                    n=1,
                    namespace=namespace,
                    request_id=f"idx{idx}-step{step}-{hash(state)}-agent{i}",
                    params=self._get_ith_agent_dict(i)['params']
                )
                for i, state in enumerate(states)
            ]
            agent_responses = await asyncio.gather(*agent_coroutines)
            states = []
            for agent_states in agent_responses:
                states.extend(agent_states)
            

            logger.info(f"het_foa_logs-{idx}-{step}-agentouts: {log_states(states)}")
            logger.info(f"het_foa_logs-{idx}-{step}-statewins: {[self.env.evaluate(s)[1] == 1 for s in states]}")

            # Early stop in case any state is solved
            if any(self.env.evaluate(state)[1] == 1 for state in states):
                solved = True
                print(f'Problem ({idx}) solved at step {step+1}')
                break

            # Filter previously visited states records
            remaining_steps = self.num_steps - (step + 1)
            visited_states = [(identifier, value*self.backtrack, state) for identifier, value, state in visited_states]
            visited_states = [state for state in visited_states if  remaining_steps >= self.min_steps - len(state[2].steps)]

            # Pruning : Failed = Finished not correctly
            failed = [i for i, state in enumerate(states) if self.env.is_final(state)]

            logger.info(f"het_foa_logs-{idx}-{step}-statefails: {[self.env.is_final(s) for s in states]}")

            if visited_states != []:
                replacements, _ = resampler.resample(visited_states.copy(), len(failed), self.resampling)
            else:
                replacements, _ = resampler.resample([("", 1, state) for state in states], len(failed), resampling_method="linear")
            states = [replacements.pop(0) if i in failed else state for i, state in enumerate(states)]


            logger.info(f'het_foa_logs-{idx}-{step}-agentreplacements: {log_states(states)}')

            # Evaluation phase
            if step < self.num_steps-1 and self.k and step % self.k == 0:

                # Evaluate the states
                value_coroutines = [
                    self.eval_agent.act(
                        model=self.model,
                        state=state,
                        n=self.num_evaluations,
                        namespace=namespace,
                        request_id=f"idx{idx}-evaluation{step}-{hash(state)}-agent{i}",
                        params=self.eval_params,
                        cache=value_cache
                    )
                    for i, state in enumerate(states)
                ]
                values = await asyncio.gather(*value_coroutines)

                # Update previously visited states records
                logger.info(f'het_foa_logs-{idx}-{step}-values: {values}')
                
                for i, value in enumerate(values):
                    states[i] = replace(states[i], value=value)

                    # TODO: REMOVE
                    if DEBUG:
                        print(f'value assigned to state {hash(states[i])}: {value}')

                    if i not in failed:
                        visited_states.append((f"{i}.{step}", value, states[i]))

                # Resampling
                states, resampled_idxs = resampler.resample(visited_states, self.num_agents, self.resampling)

        return states

    async def benchmark(self, benchmark: Benchmark, share_ns: bool=False, cache: bool=True):
        cache = {} if cache else None
        solve_coroutines = [
            self.solve(
                idx=index,
                state=state,
                namespace="benchmark" if share_ns else f"benchmark_{index}",
                value_cache=cache
            )
            for index, state in benchmark
        ]
        results = await asyncio.gather(*solve_coroutines)
        return results
