import re
import asyncio
import logging
from typing import List
from dataclasses import replace
from . import prompts as prompts
from .state import StateSciBench, state_enumerator
from ...typedefs import Agent, Model, DecodingParameters, StateReturningAgent, ValueFunctionRequiringAgent, AgentDict
from .environment import EnvironmentSciBench

logger = logging.getLogger(__name__)

class AgentActSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"     
        if state.summary:
            print(f"Using summary: {state.summary}")
            prompt_template=prompts.act_with_reflect
            current_prompt=prompt_template.format(
                reflections=state.summary,
                problem=state.puzzle,
                existing_steps=existing_steps,
            )
        elif state.reflections:
            print(f"Using reflections: {state.reflections}")
            reflection_str = "\n\n".join(state.reflections)
            prompt_template=prompts.act_with_reflect  
            current_prompt=prompt_template.format(
                reflections=reflection_str,
                problem=state.puzzle,
                existing_steps=existing_steps,
            )
        elif (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()):
            current_prompt=prompts.summary.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        else:
            current_prompt=prompts.act.format(
                problem=state.puzzle, existing_steps=existing_steps,
            )
        # Generate the response
        responses = await model.request(
            prompt=current_prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        
        for r in responses:
            print(r)
            print("---")

        # Parse the response
        # proposals = [r.strip().split("\n")[:5] for r in responses]
        # proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        # return proposals
        # patterns = r"(\b\w+)\s*(\[[^\]]*\])"
        # proposals = []
        # for response_text in responses:
        #     matches = re.findall(patterns, response_text)
        #     for match_tuple in matches:
        #         if match_tuple: # ensure match is not empty
        #             proposals.extend(join_matches(match_tuple)) # join_matches definition unavailable
        
        # New simpler parsing:
        # Treat each non-empty, stripped response string as a potential proposal.
        proposals = [r.strip() for r in responses if r.strip()]
        
        return proposals 


class AgentReactSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"
        if state.summary:
            print(f"Using summary: {state.summary}")
            prompt_template=prompts.react_with_reflect
            current_prompt=prompt_template.format(
                reflections=state.summary,
                problem=state.puzzle, existing_steps=existing_steps
            )
        elif state.reflections:
            reflection_str = "\n\n".join(state.reflections)
            prompt_template=prompts.react_with_reflect
            current_prompt=prompt_template.format(
                reflections=reflection_str,
                problem=state.puzzle, existing_steps=existing_steps
            )
        elif (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ): 
            current_prompt=prompts.summary.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        else:
            current_prompt=prompts.react.format(
                problem=state.puzzle, existing_steps=existing_steps
            )

        # Generate the response
        responses = await model.request(
            prompt=current_prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        for r in responses:
            print(r)
            print("---")
        # Parse the response
        proposals = [r.strip().split("\n")[:5] for r in responses]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals


class AgentBfsSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the action for the SciBench task.
        """
        # Format the prompt
        existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"            
        if state.summary:
            prompt_template=prompts.bfs_with_reflect
            current_prompt=prompt_template.format(
                reflections=state.summary,
                problem=state.puzzle, existing_steps=existing_steps
            )
        elif state.reflections:
            reflection_str = "\n\n".join(state.reflections)
            prompt_template=prompts.bfs_with_reflect
            current_prompt=prompt_template.format(
                reflections=reflection_str,
                problem=state.puzzle, existing_steps=existing_steps
            )
        elif (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):
            current_prompt=prompts.summary.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        else:
            current_prompt=prompts.bfs.format(
                problem=state.puzzle, existing_steps=existing_steps
            )
        # Generate the response
        responses = await model.request(
            prompt=current_prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        # Parse the response
        proposals = [
            "Next step: " + step.strip()
            for step in responses[0].split("Next step:")
            if step.strip()
        ]
        proposals = [r.strip().split("\n")[:5] for r in proposals]
        proposals = [parse_proposal(r, state.step_n, existing_steps) for r in proposals]
        return proposals


class AgentAggregateSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated action for SciBench task.
        """
        if len(actions) == 0:
            return []
        
        if (len(state.values) > 0 and state.values[max(state.values)] >= 0.9) or (
            len(state.steps) > 0 and "answer is" in state.steps[-1].lower()
        ):  # some hacky stuff from rest-mcts*
            return actions

        # Format the prompt
        steps = "\n".join(actions)
        prompt = prompts.aggregate.format(problem=state.puzzle, k=k, steps=steps)

        # Generate the response
        responses = await model.request(
            prompt=prompt,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the response
        try:
            indexes = [int(i.strip()) - 1 for i in re.findall(r"\d+", responses[0])]
            out = [actions[i] for i in indexes if i < len(actions)]
        except Exception as e:
            out = []
        return out


class AgentEvaluateSciBench(Agent):

    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:

        # Check if the state is already in the cache
        if cache is not None and state.current_state in cache:
            value = cache[state.current_state]
        else:
            # Format the promp
            num_examples = 3
            examples = "Example:\n" + "\n\nExample:\n".join(
                [example for example in prompts.examples_evaluate[:num_examples]]
            )
            existing_steps = "\n".join(state.steps) if len(state.steps) > 0 else "None\n"
            
            if state.summary:
                prompt_template=prompts.evaluate_with_reflect
                current_prompt=prompt_template.format(
                    reflections=state.summary,
                    problem=state.puzzle, existing_steps=existing_steps,
                    examples=examples,
                )
            elif state.reflections:
                reflection_str = "\n\n".join(state.reflections)
                prompt_template=prompts.evaluate_with_reflect
                current_prompt=prompt_template.format(
                    reflections=reflection_str,
                    problem=state.puzzle, existing_steps=existing_steps,
                    examples=examples,
                )
            else:
                current_prompt = prompts.evaluate.format(
                    examples=examples,
                    problem=state.puzzle,
                    existing_steps=existing_steps,
                )

            # Generate the response
            responses = await model.request(
                prompt=current_prompt,
                n=n,
                request_id=request_id,
                namespace=namespace,
                params=params,
            )

            # Parse the response
            values = [parse_value(r) for r in responses]
            value = sum(values) / len(values)

            # Cache the value
            if cache is not None:
                cache[state.current_state] = value
            state.values[state.step_n] = value
        return value
    
    
class AgentReflectSciBench(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ):
        num_examples = min(3, len(prompts.examples_reflect))
        examples_str = "(Example Reflection)\n" + "\n\n(Example Reflection)\n".join(
        [example for example in prompts.examples_reflect[:num_examples]]
        )
        
        scratchpad = state.current_state
        previous_evaluation_score = state.values[state.step_n-1] if state.step_n > 0 else 0
        evaluation_score = state.values[state.step_n]

        # print(f"State {state_enumerator.get_id(state)}: Value changed from {previous_evaluation_score} to {evaluation_score},generating reflection")
        prompt = prompts.reflect.format(
            examples=examples_str,
            problem=state.puzzle,
            scratchpad=scratchpad,
            evaluation_score=evaluation_score
        )
        
        responses = await model.request(
            prompt=prompt,
            n=1, 
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        reflection_text = responses[0].strip()
        print(f"Generated reflection for state {state_enumerator.get_id(state)}")
        return reflection_text
    
class AgentTerminalReflectSciBench(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        actions = await AgentActSciBench.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional terminal reflection logic
        states = [EnvironmentSciBench.step(state, action) for action in actions]

        reflection_coroutines = []
        reflected_state_idxs = []
        for i, s in enumerate(states):
            if not EnvironmentSciBench.is_final(s):
                continue

            # found a successful state
            if EnvironmentSciBench.evaluate(s)[0] == 1:
                return [s]
            
            # if the state has failed, we need to reflect on it
            reflection_coroutines.append(
                AgentReflectSciBench.act(
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
        
        gathered_thoughts = await asyncio.gather(*reflection_coroutines)
        
        thought_idx = 0
        for list_idx in reflected_state_idxs:
            s_being_reflected = states[list_idx] # This is the child state s that failed
            
            # The reflections for s_being_reflected should be its own existing reflections
            # (which it would have inherited or started with) plus the new thought.
            new_reflections = [gathered_thoughts[thought_idx]] + s_being_reflected.reflections
            
            states[list_idx] = replace(s_being_reflected, 
                                       reflections=new_reflections)
            thought_idx += 1

        return states
    
class AgentValueReduceReflectSciBench(StateReturningAgent, ValueFunctionRequiringAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        value_agent: AgentDict
    ) -> List[str]:
        actions = await AgentActSciBench.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional reflection logic: when value of new states is lower than current state
        states = [EnvironmentSciBench.step(state, action) for action in actions]
        failed_states = []
        non_terminal_states = []
        value_reduced_states = []
        new_states = []

        for s in states:
            # return the winning state if it exists
            if EnvironmentSciBench.evaluate(s)[0] == 1:
                return [s]
            
            # add failing states to the list
            if EnvironmentSciBench.is_final(s):
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
                AgentReflectSciBench.act(
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
            new_value = None # small increase in value for reflection
            new_states.append(replace(old_state_with_thought, value=new_value))

        return new_states


class AgentReflectSummarySciBench(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateSciBench,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[StateSciBench]:
        actions = await AgentActSciBench.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        states = [EnvironmentSciBench.step(state, action) for action in actions]
        
        failed_states = []
        non_terminal_states = []
        reflection_coroutines = []
        reflected_state_idxs = []

        for i, s in enumerate(states):
            if EnvironmentSciBench.is_final(s):
                # if the state is final, check if it is successful
                if EnvironmentSciBench.evaluate(s)[1] == 1:
                    return [s]
                else:
                    failed_states.append(s)
            else:
                non_terminal_states.append(s)

            # if the state is not final, reflect on it
            s.values[s.step_n] = EnvironmentSciBench.evaluate(s)[1]
            reflection_coroutines.append(
                AgentReflectSciBench.act(
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
        
        thought_idx = 0
        for list_idx in reflected_state_idxs:
            s_being_reflected = states[list_idx] # This is the child state s that failed
            print(f"Reflecting on state {state_enumerator.get_id(s_being_reflected)}")
            
            # The reflections for s_being_reflected should be its own existing reflections
            # (which it would have inherited or started with) plus the new thought.
            new_reflections = [thoughts[thought_idx]] + s_being_reflected.reflections
            
            states[list_idx] = replace(s_being_reflected, 
                                       reflections=new_reflections)
            thought_idx += 1

        summary_coroutines = []
        for i, original_state_idx in enumerate(reflected_state_idxs):
            reflected_state = states[original_state_idx]
            all_reflections = reflected_state.reflections + [thoughts[i]]
            
            states[original_state_idx] = replace(
                reflected_state,
                reflections=all_reflections
            )
            
            summary_prompt = prompts.reflection_summary.format(reflections="\n\n".join(all_reflections))
            summary_coroutines.append(
                model.request(
                    prompt=summary_prompt,
                    n=1,
                    request_id=f"{request_id}-summary-{i}",
                    namespace=namespace,
                    params=params
                )
            )

        summaries = await asyncio.gather(*summary_coroutines)

        summary_idx = 0
        for original_state_idx in reflected_state_idxs:
            states[original_state_idx] = replace(
                states[original_state_idx],
                summary=summaries[summary_idx][0].strip()
            )
            summary_idx += 1
            
        return states

# ---Helper functions---#
def parse_proposal(response: List[str], step_n: int, existing_steps: str) -> str:
    p = ""
    for _ in response:
        p = p + _ + " "
    p = p.strip()

    if "Next step:" in p:
        stp = p.split("Next step:")[1].strip()
        if len(stp) < 2:
            # print('Output step too short!\n')
            return ""
        if stp in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + stp

    elif "Step" in p and ":" in p:
        pre_len = len(p.split(":")[0])
        p_ = p[pre_len:]
        p_ = p_.split("Step")[0].strip()
        if len(p_) < 4:
            # print('Output step too short!\n')
            return ""
        p_ = p_[1:].strip()
        if p_ in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + p_

    else:
        p_ = p.strip()
        if len(p_) < 3:
            # print('Output step too short!\n')
            return ""
        if p_ in existing_steps:
            # print('Output step repeated!\n')
            return ""

        revised_ = "Step " + str(step_n) + ": " + p_
    revised = revised_ + "\n"
    return revised


def parse_value(response: str, low=0.0, high=1.0) -> float:
    out_value = low

    # score expected in output
    if "score" not in response.lower():
        return out_value

    response = response.lower().split("score")[-1].strip()
    try:
        match = re.findall(r"-?[0-9]+\.?[0-9]*", response)[-1]
        out_value = float(match)
        out_value = min(max(low, out_value), high)
    except Exception as e:
        out_value = low
    return out_value

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
