from gc import is_finalized
import random
from typing import Tuple
import re

from ...typedefs import State, Environment
from .state import StateLogiQA
from ...typedefs import MAX_SEED

class EnvironmentLogiQA(Environment):
    name = 'logiqa'
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, state: State, action: str) -> State:
        assert isinstance(state, StateLogiQA)
        action = action.strip()
        
        new_state = state.clone()
        new_state.steps.append(action)

        # Extract final answer if present
        match = re.search(r'answer is \((\w)\)', action.lower())
        if match:
            new_state.final_answer = match.group(1).upper()
            
        return new_state
    

    @staticmethod
    def is_valid(state: StateLogiQA, action: str) -> bool:
        """
        Checks if the action taken is valid.
        """
        raise NotImplementedError("Action validation logic is not implemented yet.")
    
    def is_final(self, state: State) -> bool:
        assert isinstance(state, StateLogiQA)
        return state.final_answer is not None
    
    def evaluate(self, state: State) -> tuple[bool, float]:
        assert isinstance(state, StateLogiQA)
        if state.final_answer is None:
            return False, 0.0

        is_correct = state.final_answer == state.answer
        return is_correct, 1.0 if is_correct else 0.0


# ---Helper functions---
def get_answer(text) -> str:
    valid_options = "abcd"
    action_taken = text.strip().lower()
    if action_taken not in valid_options and len(action_taken) == 1:
        action_taken = valid_options[int(action_taken)-1]
    elif action_taken not in valid_options:
        action_taken = action_taken.replace(".", " ").split(" ")[0].strip()

    return action_taken