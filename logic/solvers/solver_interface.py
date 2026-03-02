"""
Solver Interface
================
Defines the abstract base class and common data structures for all solvers.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple, TypedDict, Union, Dict
from dataclasses import dataclass, field


# Move/Edge type: ((r1, c1), (r2, c2))
# We use this alias for clarity in type hints.
Edge = Tuple[Tuple[int, int], Tuple[int, int]]


class HintPayload(TypedDict):
    """
    Standard dictionary format for hints returned by generate_hint().
    """
    move: Optional[Edge]
    strategy: str
    explanation: str


@dataclass
class MoveExplanation:
    """
    Structured metadata explaining a solver's decision.
    Used for the Cognitive Visualization Layer.
    """
    mode: str  # "Greedy", "Divide & Conquer", "Dynamic Programming"
    scope: str  # "Local", "Regional", "Global"
    decision_summary: str
    highlight_cells: List[Tuple[int, int]] = field(default_factory=list)
    highlight_edges: List[Edge] = field(default_factory=list)
    highlight_region: Optional[Tuple[int, int, int, int]] = None  # (r_min, c_min, r_max, c_max)
    reasoning_data: Dict[str, Any] = field(default_factory=dict)


class AbstractSolver(ABC):
    """
    Interface that all solvers must implement.
    """

    @abstractmethod
    def solve(self, board: Any = None) -> Optional[Edge]:
        """
        Return the single best move (edge) for the current board state.
        If no move is found, return None.

        This method is for pure algorithmic use (no explanations needed).
        """
        pass

    @abstractmethod
    def decide_move(self) -> Tuple[List[Tuple[Edge, int]], Optional[Edge]]:
        """
        Return a list of candidate moves with confidence scores, and the single best move.
        Format: ([(edge, confidence), ...], best_edge)

        Used by the UI to simulate "thinking" or show options.
        """
        pass

    @abstractmethod
    def generate_hint(self, board: Any = None) -> HintPayload:
        """
        Return a hint for the human player.
        Must not mutate the board state.
        """
        pass

    @abstractmethod
    def explain_last_move(self) -> str:
        """
        Return a text explanation of the last move made by this solver.
        """
        pass

    def get_last_move_explanation(self) -> Optional[MoveExplanation]:
        """
        Return the structured explanation for the last move.
        Default implementation returns None for backward compatibility.
        """
        return getattr(self, "_last_move_metadata", None)
