from typing import Tuple, List
from dataclasses import dataclass
from ...typedefs import Environment, State
from .state import StateMathArena
import re

@dataclass
class EnvironmentMathArena(Environment):
    """Environment for MathArena task."""

    name = 'matharena'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def step(self, state: State, action: str) -> State:
        assert isinstance(state, StateMathArena)
        action = action.strip()
        
        new_state = state.clone()
        new_state.steps.append(action)

        match = re.search(r'final answer is (.*)', action.lower())
        if match:
            new_state.final_answer = match.group(1).strip()
            
        return new_state

    def is_final(self, state: State) -> bool:
        assert isinstance(state, StateMathArena)
        return state.final_answer is not None

    def evaluate(self, state: State) -> tuple[bool, float]:
        assert isinstance(state, StateMathArena)
        if state.final_answer is None:
            return False, 0.0

        is_correct = state.final_answer == state.answer
        return is_correct, 1.0 if is_correct else 0.0

    @staticmethod
    def is_valid(state: StateMathArena) -> bool:
        """
        Checks if the current state is valid.

        Args:
            state (StateMathArena): State to check

        Returns:
            bool: True if state is valid
        """
        if not state.problem or not state.parsed_problem or not state.answer:
            return False

        if not isinstance(state.steps, list):
            return False

        valid_prefixes = ("Analyze[", "Explain[", "Finish[")
        
        for step in state.steps:
            if not isinstance(step, str):
                return False
            if not any(step.startswith(prefix) for prefix in valid_prefixes):
                return False
            if not step.endswith("]"):
                return False

        return True

    @staticmethod
    def get_valid_actions(state: StateMathArena) -> List[str]:
        """
        Returns list of valid actions for current state.

        Args:
            state (StateMathArena): Current state

        Returns:
            List[str]: List of valid actions
        """
        actions = [
            "Analyze[problem]",
            "Analyze[solution approach]",
            "Explain[math concepts]",
            "Explain[solution steps]"
        ]

        # Allow finishing only after at least 2 analysis/explanation steps
        if len(state.steps) >= 2:
            actions.append("Finish[answer]")

        return actions

    @staticmethod
    def apply_action(state: StateMathArena, action: str) -> StateMathArena:
        """
        Applies an action to the current state.

        Args:
            state (StateMathArena): Current state
            action (str): Action to apply

        Returns:
            StateMathArena: New state after applying action
        """
        new_state = state.copy()
        new_state.steps.append(action)
        return new_state