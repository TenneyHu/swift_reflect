from dataclasses import dataclass


@dataclass(frozen=True)
class MathState:
    question: str
    reasoning_chain: str
    reference_solution: str

    randomness: int
