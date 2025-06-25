from dataclasses import dataclass, field
from typing import List, Optional

from ...typedefs import State

@dataclass(frozen=True)
class StateHumanEval(State):
    # The initial code to complete
    puzzle: str

    # Current completion
    current_state: str

    # Steps taken towards solution
    steps: List[str]

    # Entry point for testing the code
    entry_point: str

    # The tests to run against the code
    test: str

    # A random number associated with the state
    randomness: int
    
    summary: str

    # A list to store all reflections made during the process
    reflections: List[str] = field(default_factory=list)

    # Value for this state. None means the value has not been computed yet.
    value: float | None = None
    

    def serialize(self) -> dict:
        """
        Returns a dictionary representation of the state.
        """
        return {
            "current_state": self.current_state,
            "steps": " -> ".join(self.steps),
            "reflections": self.reflections,
            "value": self.value,
            "summary": self.summary,
        }
    
    def clone(self, randomness: int=None) -> "StateHumanEval":
        """
        Returns a new instance of StateHumanEval with an optional new randomness value.
        """
        return StateHumanEval(
            puzzle=self.puzzle,
            current_state=self.current_state,
            steps=self.steps,
            entry_point=self.entry_point,
            test=self.test,
            randomness=randomness or self.randomness,
            value=self.value,
            summary=self.summary,
            reflections=self.reflections.copy()
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




'''


{
    'puzzle': 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n', 
    'current_state': 'from typing import List\n\n\ndef has_close_elements(numbers: List[float], threshold: float) -> bool:\n    """ Check if in given list of numbers, are any two numbers closer to each other than\n    given threshold.\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\n    False\n    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)\n    True\n    """\n', 
    'steps': '', 
    'entry_point': 'has_close_elements', 
    'test': "\n\nMETADATA = {\n    'author': 'jt',\n    'dataset': 'test'\n}\n\n\ndef check(candidate):\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.3) == True\n    assert candidate([1.0, 2.0, 3.9, 4.0, 5.0, 2.2], 0.05) == False\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.95) == True\n    assert candidate([1.0, 2.0, 5.9, 4.0, 5.0], 0.8) == False\n    assert candidate([1.0, 2.0, 3.0, 4.0, 5.0, 2.0], 0.1) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 1.0) == True\n    assert candidate([1.1, 2.2, 3.1, 4.1, 5.1], 0.5) == False\n\n", 
    'randomness': None
}



'''