from dataclasses import dataclass, field
from typing import List, Dict

from ...typedefs import State

class StateEnumerator:
    def __init__(self):
        self.state_hashes = []
        self.hash_to_id = {}

    def get_id(self, state: "StateSciBench") -> int:
        state_hash = hash(state)
        if state_hash not in self.hash_to_id:
            new_id = len(self.state_hashes)
            self.hash_to_id[state_hash] = new_id
            self.state_hashes.append(state_hash)
        return self.hash_to_id[state_hash]

state_enumerator = StateEnumerator()

@dataclass(frozen=True)
class StateSciBench(State):
    # The initial question to solve
    puzzle: str

    # Current state towards solving the puzzle
    current_state: str

    # Steps taken towards solving the puzzle
    steps: List[str]

    # The true answer to the question
    answer: str

    # A random number associated with the state
    randomness: int
    
    # The number of steps taken so far
    step_n: int = 0

    # The current evaluated value of the state
    value: float | None = None

    # The value that the state had at its last evaluation
    values: Dict = field(default_factory=dict)
    
    # A summary of the all reflections over previous trials
    summary: str = ""

    # A list to store all reflections made during the process
    reflections: List[str] = field(default_factory=list)

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "puzzle": self.puzzle,
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
            "answer": self.answer,
            "step_n": self.step_n,
            "value": self.value,
            "values": self.values,
            "randomness": self.randomness,
            "reflections": self.reflections,
            "summary": self.summary
        }
    
    def clone(self, randomness: int=None) -> "StateSciBench":
        """
        Returns a new instance of StateSciBench with an optional new randomness value.
        """
        return StateSciBench(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            answer=self.answer,
            step_n=self.step_n,
            value=self.value,
            values=self.values,
            randomness=randomness or self.randomness,
            reflections=self.reflections,
            summary=self.summary
        )
    
    def get_seed(self) -> int:
        """
        Returns the randomness value associated with the state.
        """
        return self.randomness
    
    def __hash__(self) -> int:
        """
        Returns a hash of the current state.
        """
        return hash(str(self.serialize()))