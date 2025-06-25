import re
import random
from urllib import response
import numpy as np
from typing import List, Tuple
import itertools
import asyncio
from dataclasses import replace

from . import prompts as prompts
from .state import StateGame24, state_enumerator
from ...typedefs import AgentDict, Agent, StateReturningAgent, ValueFunctionRequiringAgent, Model, DecodingParameters

from .environment import EnvironmentGame24

act_cache = {}
env = EnvironmentGame24

class AgentActGame24(Agent):
    """ """

    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps)
                + "\nAnswer: "
            )
        else:
            context_str = get_context(state)
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers, context_str=context_str)

        if prompt in act_cache:
            proposals = act_cache[prompt][:n]
            act_cache[prompt] = act_cache[prompt][n:]
        else:
            proposals = []
            act_cache[prompt] = []

        while len(proposals) < n:
            # Generate the response
            response = await model.request(
                prompt=prompt,
                n=1,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )
            # Parse the response
            if state.current_state != "24":
                response = [response[0].rpartition(")")[0] + ")"]
            proposals.extend(r.strip() for r in response[0].split("\n"))
            if "Possible next steps:" in proposals:
                proposals.remove("Possible next steps:")

        random.seed(state.randomness)
        random.shuffle(proposals)
        act_cache[prompt].extend(proposals[n:])
        actions = proposals[:n]
        return actions


class AgentAggregateGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated actions for the Game of 24 task.
        """
        if len(actions) == 0:
            return []
        
        if len(state.current_state.split(" ")) == 1:
            return actions

        # Format the prompt
        proposals = ""
        for idx, action in enumerate(actions):
            proposals += f"({idx + 1}) " + action + "\n"

        context_str = get_context(state)
        prompt = prompts.aggregate.format(
            state=state.current_state, proposal=proposals.strip(), n_select_sample=k,
            context_str=context_str
        )

        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        try:
            selected_indexes = [int(i.strip()) - 1 for i in re.findall(r"\d+", responses[0])]
            selected_actions = [actions[i] for i in selected_indexes if i < len(actions)]
        except:
            selected_actions = []
        return selected_actions


class AgentBfsGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of actions for the Game of 24 task.
        """
        # Format the prompt
        if len(state.current_state.strip().split(" ")) == 1:
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps).strip()
                + "\nAnswer: "
            )

        else:
            context_str = get_context(state)
            current_numbers = get_current_numbers(state)
            prompt = prompts.bfs.format(input=current_numbers, context_str=context_str)

        # Generate the response
        response = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        if state.current_state != "24":
            response = [response[0].rpartition(")")[0] + ")"]
        proposals = [r.strip() for r in response[0].split("\n")]
        if "Possible next steps:" in proposals:
            proposals.remove("Possible next steps:")
        return proposals


class AgentEvaluateGame24(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns a value for the given state
        """

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt
        if state.steps and "left" not in state.steps[-1]:
            formula = get_formula(state)
            prompt = prompts.evaluate_answer.format(input=state.puzzle, answer=formula)
        else:
            prompt = prompts.evaluate.format(input=state.current_state)

        # Format the request
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        codes = [r.split("\n")[-1].lower().strip() for r in responses]
        code_map = {r"impossible": 0.001, r"likely": 1, r"sure": 20}
        value = 0
        for pattern, weight in code_map.items():
            matches = [code for code in codes if re.search(pattern, code)]
            value += weight * len(matches)

        # Cache the value
        if cache is not None:
            cache[state.current_state] = value
        return value


class AgentReactGame24(Agent):
    """
    Agent for React algorithm
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        # Format the prompt
        if state.current_state == "24":
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps)
                + "\nAnswer: "
            )
        else:
            context_str = get_context(state)
            current_numbers = get_current_numbers(state)
            prompt = prompts.react.format(input=current_numbers, context_str=context_str)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        proposals = [r.split("Possible next step:")[-1].strip() for r in responses]
        return proposals


class AgentRapGame24(Agent):
    """
    Agent for React algorithm
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        if state.current_state == "24":
            prompt = (
                prompts.cot.format(input=state.puzzle, context_str='')
                + "\nSteps:\n"
                + "\n".join(state.steps)
                + "\nAnswer: "
            )
        else:
            current_numbers = get_current_numbers(state)
            context_str = get_context(state)
            prompt = prompts.react.format(input=current_numbers, context_str=context_str)

        responses = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        proposals = [r.strip() for r in responses]
        return proposals


class AgentSelfEvaluateGame24(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for Game24.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """

    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:

        if cache is not None and state.current_state in cache:
            return cache[state.current_state]

        # Format the prompt based on whether we're evaluating a final answer or intermediate step
        if state.steps and "left" not in state.steps[-1]:
            # Evaluating a final answer
            formula = get_formula(state)
            prompt = prompts.self_evaluate_answer.format(
                input=state.puzzle, answer=formula, steps="\n".join(state.steps)
            )
        else:
            # Evaluating intermediate reasoning steps
            current_numbers = get_current_numbers(state)
            last_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                input=current_numbers,
                step=last_step,
                previous_steps=(
                    "\n".join(state.steps[:-1]) if len(state.steps) > 1 else ""
                ),
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
                yes_probabilities.append(
                    np.exp(yes_prob)
                )  # Convert logprob to probability

        if yes_probabilities:
            value = sum(yes_probabilities) / len(yes_probabilities)
            value = value * 20
        else:
            value = 0.001

        if cache is not None:
            cache[state.current_state] = value

        return value
    
    
class AgentReflectGame24(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        num_examples = min(2, len(prompts.examples_reflect))
        examples = prompts.examples_reflect[:num_examples]
        examples_str = ""
        if examples:
            examples_str = "(Example Reflection)\n" + "\n\n(Example Reflection)\n".join(examples) + "\n\n"
        
        previous_trial = "\n".join(state.steps) + "\n" + state.current_state
        prompt = prompts.reflect_rafa.format(
            examples=examples_str,
            question=state.puzzle,
            previous_trial=previous_trial
        )
        
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        return responses[0].strip()
    
class AgentTerminalReflectGame24(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[StateGame24]:
        actions = await AgentActGame24.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )
        
        states = [env.step(state, action) for action in actions]
        reflection_coroutines = []
        reflected_state_idxs = []
        for i, s in enumerate(states):
            if not env.is_final(s):
                continue
            
            # found a successful state
            if env.evaluate(s)[1] == 1:
                return [s]
            
            reflection_coroutines.append(
                AgentReflectGame24.act(
                model=model,
                state=s,
                n=1,
                namespace=namespace,
                request_id=f"{request_id}-reflect-{i}",
                params=params,
            ))
            reflected_state_idxs.append(i)
            
        if len(reflection_coroutines) == 0:
            return states
        
        thoughts = await asyncio.gather(*reflection_coroutines)
        
        for i in reflected_state_idxs:
            states[i] = StateGame24(
                puzzle=state.puzzle,
                current_state=state.current_state,
                steps=state.steps,
                randomness=state.randomness,
                reflections=[thoughts.pop(0)] + state.reflections,
                parent=state,
                summary=state.summary
            )
        
        return states

    
class AgentReflectSummaryGame24(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateGame24,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[StateGame24]:
        actions = await AgentActGame24.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )
    
        # Generate the next states
        states = [env.step(state, action) for action in actions]
        
        reflection_coroutines = []
        reflected_state_indices = []
        
        for i, s in enumerate(states):
            if env.is_final(s):
                if env.evaluate(s)[1] == 1:
                    # found a successful state
                    return [s]
                else:
                    # It's a failed terminal state, so we reflect.
                    reflection_coroutines.append(
                        AgentReflectGame24.act(
                            model=model,
                            state=s,
                            n=1,
                            namespace=f"{namespace}-reflect-{i}",
                            request_id=f"{request_id}-reflect-{i}",
                            params=params,
                        )
                    )
                    reflected_state_indices.append(i)
        
        if not reflection_coroutines:
            return states
        
        thoughts = await asyncio.gather(*reflection_coroutines)
        
        thought_idx = 0
        for i in reflected_state_indices:
            s = states[i]
            new_reflection_list = [thoughts[thought_idx]] + s.reflections
            states[i] = replace(s, reflections=new_reflection_list)
            thought_idx += 1
            
        summary_coroutines = []
        for i in reflected_state_indices:
            s = states[i] # this is the state with updated reflections
            all_reflections = s.reflections
            
            summary_prompt = prompts.summarize_reflections.format(reflections="\n\n".join(all_reflections))
            summary_coroutines.append(
                model.request(
                    prompt=summary_prompt,
                    n=1,
                    request_id=f"{request_id}-summary-{i}",
                    namespace=f"{namespace}-summary-{i}",
                    params=params,
                )
            )
        
        if not summary_coroutines:
             return states

        summaries = await asyncio.gather(*summary_coroutines)
        
        summary_idx = 0
        for i in reflected_state_indices:
            new_summary = summaries[summary_idx][0].strip()
            states[i] = replace(states[i], summary=new_summary)
            summary_idx += 1
            
        return states



def get_current_numbers(state: StateGame24) -> str:
    """
    Returns the current numbers in the state.
    """
    last_line = state.current_state.strip().split("\n")[-1]
    return last_line.split("left: ")[-1].split(")")[0]


def get_context(state: StateGame24) -> str:
    context_str = ''
    if state.summary:
        print(f"Using summary: {state.summary}")
        context_str = f"\n{prompts.REFLECTION_SUMMARY_HEADER}\n{state.summary}\n\n"
    return context_str


def get_formula(state: StateGame24) -> str:
    if state.steps:
        formula = state.steps[-1].lower().replace("answer: ", "")
        return formula
    else:
        # Should do some error handling here but for the moment we'll take it as it is
        return ""
