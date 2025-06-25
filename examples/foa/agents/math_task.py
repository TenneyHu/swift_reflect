import asyncio

from foa.prompts.math_task import foa_step_prompt, value_prompt
from foa.states.math_task import MathState
from examples.foa.utils import grade_answer
from examples.foa.utils import extract_last_boxed_answer
from cachesaver.thirdparty_wrappers.openai_wrapper import AsyncCachedOpenAIAPI


class MathAgent:
    @staticmethod
    async def step(state: MathState, api: AsyncCachedOpenAIAPI, namespace: str, id: int) -> MathState:

        # set up a prompt
        user_message = foa_step_prompt.replace("{question}", state.question)
        user_message = user_message.replace("{reasoning_chain}", state.reasoning_chain)

        # eh, the api expects all this config. ToDo: not so nice to hardcode this here.
        prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "model": "gpt-35-turbo-0125",
            "n": 1,
            "temperature": 1.0,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # make the request
        response = await api.create_chat_completion(prompt, id, namespace)
        await asyncio.sleep(0.1)

        # the prompt instructs to do just one step
        new_reasoning_chain = state.reasoning_chain + "\n" + response.choices[0].message.content

        # update the state
        state = MathState(question=state.question, reasoning_chain=new_reasoning_chain, randomness=state.randomness)
        return state

    @staticmethod
    async def value(state: MathState, api: AsyncCachedOpenAIAPI, namespace: str, id: int) -> MathState:

        # set up a prompt
        user_message = value_prompt.replace("{question}", state.question)
        user_message = user_message.replace("{reasoning_chain}", state.reasoning_chain)

        # again, maybe better to move this out.
        prompt = {
            "messages": [
                {
                    "role": "user",
                    "content": user_message
                }
            ],
            "model": "gpt-35-turbo-0125",
            "n": 1,
            "temperature": 1.0,
            "max_tokens": 1000,
            "top_p": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
        }

        # make the request
        response = await api.create_chat_completion(prompt, id, namespace)

        # the prompt instructs to do just one step
        assessment = response.choices[0].message.content.strip().lower()

        value_map = {
            "sure": 10,
            "maybe": 1,
            "impossible": 0.1
        }

        value = None
        for key, val in value_map.items():
            if key in assessment:
                value = val
                break

        assert value is not None, "here we could add a default value. For now I'd like this to throw an error"

        return value

    @staticmethod
    async def evaluate(state: MathState, namespace: str, id: int) -> MathState:

        # extract the answer from the reasoning chain
        answer = extract_last_boxed_answer(state.reasoning_chain)

        # grade the answer
        grade = grade_answer(state.reference_solution, answer)

        return grade
