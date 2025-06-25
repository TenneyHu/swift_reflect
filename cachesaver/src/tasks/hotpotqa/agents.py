import re
import random
from urllib import response
import numpy as np
from typing import List, Tuple
import itertools
import asyncio
from dataclasses import replace

from . import prompts as prompts
from .environment import EnvironmentHotpotQA
from .state import StateHotpotQA
from ...typedefs import AgentDict, Agent, StateReturningAgent, ValueFunctionRequiringAgent, Model, DecodingParameters


class AgentActHotpotQA(Agent):
    """
    Agent performing the Act operation for the HotpotQA task.
    Can use reflections if provided in the state.
    """
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_act[:num_examples]]
        )

        if state.reflections:
            reflection_str = "\n\n".join(state.reflections) # Join with double newline for clarity
            prompt_template = prompts.act_with_reflect
            current_prompt = prompt_template.format(
                reflections_header=prompts.REFLECTION_HEADER,
                reflections=reflection_str,
                examples=examples,
                question=state.puzzle,
                current_state=state.current_state
            )
        else:
            prompt_template = prompts.act
            current_prompt = prompt_template.format(
                examples=examples, question=state.puzzle, current_state=state.current_state
            )
        
        responses = await model.request(
            prompt=current_prompt,
            n=n, # If n > 1, multiple responses will be generated
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        patterns = r"(\b\w+)\s*(\[[^\]]*\])"
        # If n > 1, 'responses' is a list of strings. Each string can contain multiple actions.
        # The original code processes each response string.
        proposals = []
        for response_text in responses:
            matches = re.findall(patterns, response_text)
            for match_tuple in matches:
                if match_tuple: # ensure match is not empty
                    proposals.extend(join_matches(match_tuple)) # join_matches expects a tuple
        
        # If `n` for AgentActHotpotQA is meant to control the number of *distinct actions*
        # from a single LLM call that might list multiple actions, this is fine.
        # If `n` is meant for `model.request(n=...)` to get `n` independent completions,
        # and each completion is one action, then the parsing might need adjustment or `n=1` for this agent.
        # Given GOT uses num_generate with AgentActHotpotQA, it seems `n` here is for model.request.
        return proposals 


class AgentBfsHotpotQA(Agent):
    """
    Agent performing the BFS operation for the HotpotQA task.
    Can use reflections if provided in the state.
    """
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        # n: int, # BFS typically generates a list of actions from one prompt
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_bfs[:num_examples]]
        )

        if state.reflections:
            reflection_str = "\n\n".join(state.reflections)
            prompt_template = prompts.bfs_with_reflect
            current_prompt = prompt_template.format(
                reflections_header=prompts.REFLECTION_HEADER,
                reflections=reflection_str,
                examples=examples,
                question=state.puzzle,
                current_state=state.current_state
            )
        else:
            prompt_template = prompts.bfs
            current_prompt = prompt_template.format(
                examples=examples, question=state.puzzle, current_state=state.current_state
            )
        
        # BFS agent usually generates a single response string containing multiple newline-separated actions
        response_list = await model.request(
            prompt=current_prompt,
            n=1, 
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        
        proposals = [r.strip() for r in response_list[0].split("\n") if r.strip()]
        return proposals


class AgentAggregateHotpotQA(Agent):
    """
    Agent performing the Aggregate operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of the k best actions for the HotpotQA task.
        """

        if len(actions) == 0:
            return []  # No actions to aggregate

        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_aggregate[:num_examples]]
        )
        actions = "\n".join(action for action in actions)
        prompt = prompts.aggregate.format(
            examples=examples,
            question=state.puzzle,
            current_state=state.current_state,
            k=k,
            actions=actions,
        )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        pattern = r"(\b\w+)\s*(\[[^\]]*\])"
        aggregate_actions = [
            join_matches(match) for match in re.findall(pattern, responses[0]) if match
        ]
        return list(itertools.chain(*aggregate_actions))


class AgentReactHotpotQA(Agent):
    """
    Agent performing the ReAct operation for the HotpotQA task.
    Can optionally use reflections from previous trials.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of n thought-action pairs for the HotpotQA task.
        """
        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_react[:num_examples]]
        )
        prompt = prompts.react.format(
            examples=examples, question=state.puzzle, current_state=state.current_state
        )
        
        if state.reflections:
            reflection_str = "\n".join(state.reflections)
            prompt_template = prompts.react_with_reflect
            current_prompt = prompt_template.format(
                reflections_header=prompts.REFLECTION_HEADER,
                reflections=reflection_str,
                examples=examples,
                question=state.puzzle,
                current_state=state.current_state
            )
        else:
            prompt_template = prompts.react
            current_prompt = prompt_template.format(
                examples=examples, question=state.puzzle, current_state=state.current_state
            )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        react_actions = [r.strip() for r in responses]
        return react_actions

        
class AgentSelfEvaluateHotpotQA(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for HotpotQA.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns a value estimation for the current state based on self-evaluation.
        """
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt based on whether we're evaluating a final answer or intermediate step
        if state.steps and "Finish" in state.steps[-1]:
            # Evaluating a final answer
            answer = state.steps[-1].replace("Finish[", "").replace("]", "")
            prompt = prompts.self_evaluate_answer.format(
                question=state.puzzle, steps="\n".join(state.steps), answer=answer
            )
        else:
            # Evaluating intermediate reasoning steps
            last_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                current_state=state.current_state, step=last_step
            )

        eval_params = DecodingParameters(
            temperature=params.temperature,
            max_completion_tokens=params.max_completion_tokens,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=True,
        )

        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=eval_params,
        )

        # Calculate the average probability of "Yes" across all responses
        yes_probabilities = []
        for response in responses:
            # Get the logprobs for the first token after the prompt
            if hasattr(response, "logprobs") and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                # Look for Yes token probability
                yes_prob = next(
                    (
                        prob
                        for token, prob in first_token_logprobs.items()
                        if token.lower() in ["yes", "yes.", "yes!"]
                    ),
                    0.0,
                )
                yes_probabilities.append(np.exp(yes_prob))

        if yes_probabilities:
            value = sum(yes_probabilities) / len(yes_probabilities)
            value = value * 20
        else:
            value = 0.001

        if cache is not None:
            cache[state.current_state] = value

        return value


class AgentEvaluateHotpotQA(Agent):
    """
    Agent performing the Evaluate operation for the HotpotQA task.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns an evaluations for the HotpotQA task.
        """
        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        if state.value is not None:
            return state.value        


        # Format the prompt
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_evaluate[:num_examples]]
        )
        prompt = prompts.evaluate.format(
            examples=examples, question=state.puzzle, current_state=state.current_state
        )

        # Generate the responses
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        values = []
        pattern = r"\b(?:correctness[\s_]?score|score for correctness|correctness)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"

        for response in responses:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                value = float(match.group(1))
            else:
                # print(f"Unable to parse value from response : {response}")
                value = 1
            values.append(value)
        value = sum(values)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        
        return value


class AgentReflectHotpotQA(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ):
        num_examples = min(2, len(prompts.examples_reflect))
        examples_str = "(Example Reflection)\n" + "\n\n(Example Reflection)\n".join(
        [example for example in prompts.examples_reflect[:num_examples]]
        )
        
        scratchpad = state.current_state

        prompt = prompts.reflect.format(
            examples=examples_str,
            question=state.puzzle,
            scratchpad=scratchpad
        )
        
        responses = await model.request(
            prompt=prompt,
            n=1, 
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        reflection_text = responses[0].strip()
        return reflection_text
        

class AgentTerminalReflectHotpotQA(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        actions = await AgentActHotpotQA.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional terminal reflection logic
        states = [EnvironmentHotpotQA.step(state, action) for action in actions]

        reflection_coroutines = []
        reflected_state_idxs = []
        for i, s in enumerate(states):
            if not EnvironmentHotpotQA.is_final(s):
                continue

            # found a successful state
            if EnvironmentHotpotQA.evaluate(s)[1] == 1:
                return [s]
            
            # if the state has failed, we need to reflect on it
            reflection_coroutines.append(
                AgentReflectHotpotQA.act(
                    model=model,
                    state=s,
                    n=1,
                    namespace=namespace,
                    request_id=f"{request_id}-reflect-{i}",
                    params=params,
                )
            )

            reflected_state_idxs.append(i)
        
        if len(reflection_coroutines) == 0:
            return states
        
        thoughts = await asyncio.gather(*reflection_coroutines)
        
        for i in reflected_state_idxs:
            states[i] = StateHotpotQA(
                puzzle=state.puzzle,
                current_state=state.puzzle,
                steps=[],
                answer=state.answer,
                docstore=state.docstore,
                randomness=state.randomness,
                reflections=[thoughts.pop(0)] + state.reflections,
                parent=state,
            )

        return states


class AgentValueReduceReflectHotpotQA(StateReturningAgent, ValueFunctionRequiringAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        value_agent: AgentDict
    ) -> List[str]:
        actions = await AgentActHotpotQA.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional reflection logic: when value of new states is lower than current state
        states = [EnvironmentHotpotQA.step(state, action) for action in actions]
        failed_states = []
        non_terminal_states = []
        value_reduced_states = []
        new_states = []

        for s in states:
            # return the winning state if it exists
            if EnvironmentHotpotQA.evaluate(s)[0] == 1:
                return [s]
            
            # add failing states to the list
            if EnvironmentHotpotQA.is_final(s):
                failed_states.append(s)
            else:
                non_terminal_states.append(s)        
        
        # get values for non terminal states
        if len(non_terminal_states) > 0:
            value_coroutines = [
                value_agent["agent"].act(
                    model=value_agent.get('model') or model,
                    state=s,
                    n=value_agent['num_agents'],
                    namespace=namespace,
                    request_id=f"{request_id}-value-{i}",
                    params=value_agent['params'],
                ) for i, s in enumerate(non_terminal_states)
            ]
            values = await asyncio.gather(*value_coroutines)
            for i in range(len(non_terminal_states)):
                if values[i] >= state.value:
                    # if the value of the new state is higher than the current state, keep it
                    new_states.append(replace(non_terminal_states[i], value=values[i]))
                else:
                    value_reduced_states.append(replace(non_terminal_states[i], value=values[i]))

        # get values for failed states
        if len(failed_states) > 0:
            for i in range(len(failed_states)):
                value_reduced_states.append(replace(failed_states[i], value=0))
        
        if len(value_reduced_states) == 0:
            # if no states were reduced, return the new states
            return new_states
        

        # reflect on the value reduced states
        reflection_coroutines = []
        for i, s in enumerate(value_reduced_states):
            reflection_coroutines.append(
                AgentReflectHotpotQA.act(
                    model=model,
                    state=s,
                    n=1,
                    namespace=namespace,
                    request_id=f"{request_id}-reflect-{i}",
                    params=params,
                )
            )

        thoughts = await asyncio.gather(*reflection_coroutines)
        num_thoughts = len(thoughts)

        for i in range(num_thoughts):
            old_state_with_thought = state.clone()
            old_state_with_thought.reflections.insert(0, thoughts.pop(0))

            # TODO: Adjust the value of the state after reflection
            new_value = state.value # small increase in value for reflection
            new_states.append(replace(old_state_with_thought, value=new_value))

        return new_states
    
    
class AgentReflectSummaryHotpotQA(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[StateHotpotQA]:
        # 1. Act, using summary if available
        num_examples = 2
        examples = "(Example)\n" + "\n\n(Example)\n".join(
            [example for example in prompts.examples_act[:num_examples]]
        )

        
        if state.summary:
            print(f"Using summary: {state.summary}")
            print(len(state.reflections))
            prompt_template = prompts.react_with_reflect
            current_prompt = prompt_template.format(
                reflections_header=prompts.REFLECTION_SUMMARY_HEADER,
                reflections=state.summary,
                examples=examples,
                question=state.puzzle,
                current_state=state.current_state
            )
        else:
            prompt_template = prompts.react
            current_prompt = prompt_template.format(
                examples=examples, question=state.puzzle, current_state=state.current_state
            )

        responses = await model.request(
            prompt=current_prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        patterns = r"(\b\w+)\s*(\[[^\]]*\])"
        actions = []
        for response_text in responses:
            matches = re.findall(patterns, response_text)
            for match_tuple in matches:
                if match_tuple:
                    actions.extend(join_matches(match_tuple))

        states = [EnvironmentHotpotQA.step(state, action) for action in actions]

        # 2. Reflect on terminal states
        reflection_coroutines = []
        reflected_state_idxs = []
        for i, s in enumerate(states):
            if not EnvironmentHotpotQA.is_final(s):
                continue

            if EnvironmentHotpotQA.evaluate(s)[1] == 1:
                return [s]  # Found a successful state

            reflection_coroutines.append(
                AgentReflectHotpotQA.act(
                    model=model,
                    state=s,
                    n=1,
                    namespace=namespace,
                    request_id=f"{request_id}-reflect-{i}",
                    params=params,
                )
            )
            reflected_state_idxs.append(i)

        if not reflection_coroutines:
            return states

        new_reflections = await asyncio.gather(*reflection_coroutines)

        # 3. Summarize reflections
        all_reflections = state.reflections + new_reflections
        summarization_prompt = prompts.summarize_reflections.format(
            reflections="\n".join(all_reflections)
        )

        summary_responses = await model.request(
            prompt=summarization_prompt,
            n=1,
            request_id=f"{request_id}-summary",
            namespace=namespace,
            params=params,
        )
        summary = summary_responses[0].strip()

        # 4. Create new states for the next trial with summary
        new_reflections_iter = iter(new_reflections)
        for i in reflected_state_idxs:
            states[i] = StateHotpotQA(
                puzzle=state.puzzle,
                current_state=state.puzzle,  # Reset for new trial
                steps=[],
                answer=state.answer,
                docstore=state.docstore,
                randomness=states[i].randomness,
                reflections=state.reflections + [next(new_reflections_iter)],
                parent=state,
                summary=summary,
            )

        return states
    
    
class AgentReflectPrevKHotpotQA(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHotpotQA,
        n: int, # Number of responses/actions to generate
        namespace: str,
        k: int, # Number of reflections to keep
        request_id: str,
        params: DecodingParameters,
    ) -> List[StateHotpotQA]:
        actions = await AgentActHotpotQA.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional terminal reflection logic
        states = [EnvironmentHotpotQA.step(state, action) for action in actions]

        reflection_coroutines = []
        reflected_state_idxs = []
        for i, s in enumerate(states):
            if not EnvironmentHotpotQA.is_final(s):
                continue

            # found a successful state
            if EnvironmentHotpotQA.evaluate(s)[1] == 1:
                return [s]
            
            # if the state has failed, we need to reflect on it
            reflection_coroutines.append(
                AgentReflectHotpotQA.act(
                    model=model,
                    state=s,
                    n=1,
                    namespace=namespace,
                    request_id=f"{request_id}-reflect-{i}",
                    params=params,
                )
            )

            reflected_state_idxs.append(i)
        
        if len(reflection_coroutines) == 0:
            return states
        
        thoughts = await asyncio.gather(*reflection_coroutines)
        
        for i in reflected_state_idxs:
            states[i] = StateHotpotQA(
                puzzle=state.puzzle,
                current_state=state.puzzle,
                steps=[],
                answer=state.answer,
                docstore=state.docstore,
                randomness=state.randomness,
                reflections=[thoughts.pop(0)] + state.reflections[:k], # Only keep last k reflections
                parent=state,
                summary=state.summary,
            )
            
        print(f'length of reflections: {len(states[0].reflections)}')

        return states


# ---Helper functions---
def join_matches(matches) -> List[str]:
    """
    Joins matched strings from a regex search into a single string.
    """
    if not matches: # Handle empty matches
        return []
    
    if isinstance(matches, tuple) and all(isinstance(m, str) for m in matches): # A single match tuple
        return ["".join(matches)]
    
    elif isinstance(matches, list) and all(isinstance(item, tuple) for item in matches):
        return ["".join(match_tuple) for match_tuple in matches]
    
    if isinstance(matches, tuple) and all(isinstance(s, str) for s in matches):
        return ["".join(matches)]
    
    if isinstance(matches[0], str):
        matches = [matches]
    return ["".join(match) for match in matches]
