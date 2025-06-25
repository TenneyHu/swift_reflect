from typing import List
import re
import itertools
import numpy as np
import asyncio
from dataclasses import replace

from . import prompts as prompts
from .state import StateHumanEval
from .environment import EnvironmentHumanEval
from ...typedefs import Request, Agent, AgentDict, StateReturningAgent, ValueFunctionRequiringAgent, Model, DecodingParameters

class AgentActHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns actions generated for the HumanEval task.
        """

        #TODO: REMOVE
        if len(state.reflections) > 0:
            print(f'acting on state with {len(state.reflections)} reflections')

        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        # Generate the response
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": state.current_state},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        actions = [r.strip() for r in responses]
        return actions


class AgentAggregateHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        actions: List[str],
        k: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns the aggregated actions for the HumanEval task.
        """
        if len(actions) == 0:
            return []

        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)
        user_prompt = prompts.aggregate_prompt.format(
            prompt=state.current_state, k=k, implementations="\n".join(actions)
        )

        # Generate the response
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": user_prompt},
            ],
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        

        # Parse the response
        try:
            indexes = [int(i.strip()) - 1 for i in re.findall(r'\d+', responses[0])]
            aggregate_actions = [actions[i] for i in indexes if i < len(actions)]
        except:
            aggregate_actions = []
        return aggregate_actions


class AgentBfsHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns actions generated for the HumanEval task.
        """
        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        ### change n, depending on how many to generate
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_BFS.format(lang=language, n=5)
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": state.current_state},
            ],
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        response_text = responses[0]

        code_blocks = re.findall(r'(```.*?```)', response_text, flags=re.DOTALL)

        # Strip each code block
        actions = [block.strip() for block in code_blocks]

        return actions


class AgentEvaluateHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        cache: dict = None,
    ) -> float:
        """
        Returns the evaluation score for the HumanEval task.
        """
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        user_prompt = prompts.evaluation_prompt.format(
            prompt=state.puzzle,  # The function signature + docstring
            implementation=state.current_state  # The code you want to evaluate
        )

        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": user_prompt},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )
        value = sum_overall_scores(responses)
        return value


class AgentReactHumanEval(Agent):
    """
    Agent performing the ReAct operation for the HumanEval task.
    """
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int,
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        """
        Returns a list of n thought-action pairs for the HumanEval task.
        """

        #TODO: REMOVE
        if len(state.reflections) > 0:
            print(f'reacting on state with {len(state.reflections)} reflections')

        # Format the prompt
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)
        react_prompt = prompts.react.format(
            prompt=state.puzzle,
            current_state=state.current_state
        )

        # Generate the responses
        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": react_prompt},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        # Parse the responses
        react_actions = [r.strip() for r in responses]
        return react_actions


class AgentSelfEvaluateHumanEval(Agent):
    """
    Agent that performs self-evaluation of reasoning steps for HumanEval.
    Uses the LLM's own estimation of correctness by evaluating each reasoning step.
    Uses the probability of "Yes" as a reward signal for correct reasoning.
    """
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
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
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        if state.steps and "Finish" in state.steps[-1]:
            # Evaluating a final answer
            answer = state.steps[-1].replace("Finish[", "").replace("]", "")
            prompt = prompts.self_evaluate_answer.format(
                prompt=state.puzzle,
                steps='\n'.join(state.steps),
                answer=answer
            )
        else:
            # Evaluating intermediate reasoning steps
            last_step = state.steps[-1] if state.steps else ""
            prompt = prompts.self_evaluate_step.format(
                prompt=state.puzzle,
                current_state=state.current_state,
                step=last_step
            )

        eval_params = DecodingParameters(
            temperature=params.temperature,
            max_completion_tokens=params.max_completion_tokens,
            top_p=params.top_p,
            stop=params.stop,
            logprobs=True
        )

        responses = await model.request(
            prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": prompt},
            ],
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=eval_params,
        )

        # Calculate the average probability of "Yes" across all responses
        yes_probabilities = []
        for response in responses:
            # Get the logprobs for the first token after the prompt
            if hasattr(response, 'logprobs') and response.logprobs:
                first_token_logprobs = response.logprobs[0]
                # Look for Yes token probability
                yes_prob = next((prob for token, prob in first_token_logprobs.items() 
                               if token.lower() in ['yes', 'yes.', 'yes!']), 0.0)
                yes_probabilities.append(np.exp(yes_prob))  # Convert logprob to probability

        if yes_probabilities:
            value = sum(yes_probabilities) / len(yes_probabilities)
            value = value * 20  # Scale up the value similar to Game24
        else:
            value = 0.001

        if cache is not None:
            cache[state.current_state] = value

        return value


class AgentReflectHumanEval(Agent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ):
        lang = "python" if "def" in state.puzzle else "rust"
        
        examples_list = prompts.examples_reflect_py if lang == "python" else prompts.examples_reflect_rs

        num_examples = min(2, len(examples_list))
        examples_str = "(Example Reflection)\n" + "\n\n(Example Reflection)\n".join(
        [example for example in examples_list[:num_examples]]
        )
        
        scratchpad = state.current_state

        prompt = prompts.reflect.format(
            lang=lang,
            examples=examples_str,
            function_impl=state.current_state,
        )
        
        responses = await model.request(
            prompt=prompt,
            n=1, 
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        reflection_text = responses[0].strip().replace("(END OF OUTPUT FORMAT)", "").strip()

        # TODO: REMOVE
        print('\nreflection generated!')

        return reflection_text


class AgentValueReduceReflectHumanEval(StateReturningAgent, ValueFunctionRequiringAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
        value_agent: AgentDict
    ) -> List[str]:
        actions = await AgentActHumanEval.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        # additional reflection logic: when value of new states is lower than current state
        states = [EnvironmentHumanEval.step(state, action) for action in actions]
        failed_states = []
        non_terminal_states = []
        value_reduced_states = []
        new_states = []

        for s in states:
            # return the winning state if it exists
            if EnvironmentHumanEval.evaluate(s)[0] == 1:
                return [s]
            
            # add failing states to the list
            if EnvironmentHumanEval.is_final(s):
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
                AgentReflectHumanEval.act(
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
            new_value = state.value + 0.5 # small increase in value for reflection
            new_states.append(replace(old_state_with_thought, value=new_value))

        return new_states



class AgentTerminalReflectHumanEval(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[str]:
        actions = await AgentReactHumanEval.act(
            model=model,
            state=state,
            n=n,
            namespace=namespace,
            request_id=request_id,
            params=params,
        )

        states = [EnvironmentHumanEval.step(state, action) for action in actions]

        reflection_coroutines = []
        reflected_idxs = []
        for i, s in enumerate(states):
            # if any of the states won, just return that state
            if EnvironmentHumanEval.evaluate(s)[1] == 1:
                return [s]
        
            # If failed terminal state, create the thought
            if EnvironmentHumanEval.is_final(s):
                reflected_idxs.append(i)
                reflection_coroutines.append(
                    AgentReflectHumanEval.act(
                        state=s,
                        model=model,
                        namespace=namespace,
                        n=1,
                        request_id=f"{request_id}-reflect-{i}",
                        params=params,
                    )
                )
        
        if len(reflection_coroutines) == 0:
            return states
        
        thoughts = await asyncio.gather(*reflection_coroutines)
        
        for i in reflected_idxs:
            s_reflections = list(states[i].reflections)
            s_reflections.insert(0, thoughts.pop(0))
            states[i] = replace(states[i], reflections=s_reflections)  # small increase in value for reflection


        print(f'Terminal Reflect Agent returning states:')
        for s in states:
            print(f"State {hash(s)}: steps={len(s.steps)}, value={s.value}, reflections={len(s.reflections)}")
        input()

        return states

class AgentReflectSummaryHumanEval(StateReturningAgent):
    @staticmethod
    async def act(
        model: Model,
        state: StateHumanEval,
        n: int, # Number of responses/actions to generate
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> List[StateHumanEval]:
        
        # 1. Act part (like AgentActHumanEval)
        language = "py" if "def" in state.puzzle else "rs"
        instruct = prompts.SIMPLE_CHAT_INSTRUCTION_V2.format(lang=language)

        if state.summary:
            print(f'using summary: {state.summary}')
            if language == "py":
                user_prompt = prompts.summary_act_prompt_py.format(
                    summary=state.summary,
                    puzzle=state.puzzle
                )
            else:
                user_prompt = prompts.summary_act_prompt_rs.format(
                    summary=state.summary,
                    puzzle=state.puzzle
                )
            prompt = [
                {"role": "user", "content": user_prompt}
            ]
        else:
             prompt=[
                {"role": "system", "content": instruct},
                {"role": "user", "content": state.current_state},
            ]

        actions = await model.request(
            prompt=prompt,
            n=n,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        actions = [r.strip() for r in actions]
        
        # 2. Create new states from actions
        states = [EnvironmentHumanEval.step(state, action) for action in actions]
        
        # 3. Reflect and summarize on failed states
        new_states = []
        for i, s in enumerate(states):
            _solved, score = EnvironmentHumanEval.evaluate(s)
            
            if score == 1.0:
                # if a solution is found, we can just return it
                return [s]
            
            # Failed state: reflect and summarize
            reflection_text = await AgentReflectHumanEval.act(
                model=model,
                state=s,
                n=1,
                namespace=namespace,
                request_id=f"{request_id}-reflect-{i}",
                params=params,
            )
            
            new_reflections = s.reflections + [reflection_text]
            
            summary_text = await AgentReflectSummaryHumanEval.summarize(
                model=model,
                reflections=new_reflections,
                namespace=namespace,
                request_id=f"{request_id}-summary-{i}",
                params=params,
            )
            
            final_state = replace(s, reflections=new_reflections, summary=summary_text)
            new_states.append(final_state)
            
        return new_states

    @staticmethod
    async def summarize(
        model: Model,
        reflections: List[str],
        namespace: str,
        request_id: str,
        params: DecodingParameters,
    ) -> str:
        if not reflections:
            return ""

        summarization_prompt_text = prompts.summarize_reflections_prompt.format(
            reflections="\n\n---\n\n".join(reflections)
        )

        responses = await model.request(
            prompt=summarization_prompt_text,
            n=1,
            request_id=request_id,
            namespace=namespace,
            params=params,
        )

        return responses[0].strip()




# Helper function
def sum_overall_scores(evaluations):
    values = []
    pattern = r"\b(?:overall[\s_]?score|score)\b(?:\s*(?:is|=|:|was|stands at|of))?\s*(-?\d+(?:\.\d+)?)"
    
    for evaluation in evaluations:
        match = re.search(pattern, evaluation, re.IGNORECASE)
        if match:
            value = float(match.group(1))
        else:
            value = 1
        values.append(value)
    value = sum(values)

    return value