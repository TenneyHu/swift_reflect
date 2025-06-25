import asyncio
from openai import AsyncOpenAI
from cachesaver.typedefs import Request, Batch,Response, SingleRequestModel, BatchRequestModel
from src.models import OnlineLLM
from src.tasks.game24.agents import AgentTerminalReflexionGame24, AgentActGame24, AgentReactGame24
from src.tasks.game24.state import StateGame24
from diskcache import Cache
from typing import List
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
client = AsyncOpenAI(api_key=OPENAI_API_KEY)  


prompt = '''You are an advanced reasoning agent that can improve based on self reflection. You will be given a previous reasoning trial where the task was to use numbers and basic arithmetic operations (+ - * /) to obtain 24. In each step, two of the remaining numbers were chosen to obtain a new number. You were unsuccessful in answering the question either because you reached the wrong answer, or you used up your set number of reasoning steps, or your actions were inefficient. In a few sentences, diagnose a possible reason for failure or inefficiency and devise a new, concise, high level plan that aims to mitigate the same failure. Use complete sentences. Also, in the given data, mark each state as 'sure' or 'impossible'. Give 'sure' if the state is correct and can lead to 24 and give 'impossible' if the state is incorrect or illegal. You have been given a specific output format to follow. Strictly follow the output format.
Output format:
Diagnosis: <your diagnosis>
States:
<Data state 1> <label for state 1 (sure/impossible)>
<Data state 2> <label for state 2 (sure/impossible)>
<Data state 3> <label for state 3 (sure/impossible)>
...
<Data state n> <label for state n (sure/impossible)>
(END OF OUTPUT FORMAT)

Previous trial:
State: 4 5 7 8
Action: 4 + 5 = 9
State: 7 8 9
Action: 9 + 7 = 16
State: 8 16
Action: 16 - 8 = 8
State: 8

(END OF PREVIOUS TRIAL)

Output:
'''


class LLM(SingleRequestModel, BatchRequestModel):
    def __init__(self,client:AsyncOpenAI,model:str='gpt-4'):
        self.client=client
        self.model=model
    
    async def request(self, prompt, n, request_id, namespace, params)->Response:
        completion=await client.chat.completions.create(
            messages=[{"role":"user","content":prompt}],
            model='gpt-4.1-nano',
            n=1,
            temperature=0.7,
            max_completion_tokens=256
        )
        response=Response([choice.message.content for choice in completion.choices])
        return response.data

    async def batch_request(self,batch:Batch)->List[Response]:
        requests=[self.request(request) for request in batch.requests]
        completions=await asyncio.gather(*requests)
        return completions


async def main()->List[any]:     
    model=LLM(
        client=client,
        model='gpt-4.1-nano'
    )

    # state = StateGame24(
    #     puzzle="4 5 7 8",
    #     current_state="8",
    #     steps=['4 + 5 = 9 (left: 7 8 9)', '9 + 7 = 16 (left: 8 16)', '16 - 8 = 8 (left: 8)'],
    #     randomness=42,
    #     context=""
    # )


    # while input('continue? (y/n): ').lower() != 'n':
    #     new_state = await AgentTerminalReflexionGame24.act(
    #         model=model,
    #         state=state,
    #         n=1,
    #         namespace="game24",
    #         request_id="game24_1",
    #         params=None
    #     )

    #     state = new_state[0]

    resp = await model.request(
        prompt=prompt,
        n=1,
        request_id="game24_1",
        namespace="game24",
        params=None
    )

    print(resp[0])

    return resp[0]


if __name__ == "__main__":
    response=asyncio.run(main())



'''



4 + 4 = 8 (left: 8 8 8)  
4 - 4 = 0 (left: 8 8 0)  
8 - 4 = 4 (left: 4 8 8)  
8 + 4 = 12 (left: 4 8 12)  
8 / 4 = 2 (left: 8 2 8)













'''
