"""
Solver Interface
================
Abstract interface for solver strategies.

This is a refactor-only addition to support future extensibility.
Existing gameplay and hint behavior must remain unchanged.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, TypedDict, NotRequired


class HintPayload(TypedDict):
    """
    Standardized hint payload returned by solver strategies.

    - move: edge/move information (duck-typed; UI/game decides structure)
    - explanation: human-readable explanation (optional but recommended)
    - strategy: strategy name (e.g. "Greedy")
    """

    move: Any
    explanation: NotRequired[str]
    strategy: str


class AbstractSolver(ABC):
    """
    Strategy interface for solver engines.

    Notes:
    - `board` is intentionally duck-typed to avoid coupling to UI/GameState/Graph.
    - Implementations must not mutate the board unless explicitly documented.
    """

    @abstractmethod
    def solve(self, board: Any):
        """
        Return a move (or solution) for the given board.
        Concrete meaning depends on the strategy and integration point.
        """

    @abstractmethod
    def generate_hint(self, board: Any) -> HintPayload:
        """
        Return a hint payload for the given board.

        CRITICAL:
        - Do NOT change the underlying hint/solving logic.
        - Only extend the output format to include explanation + strategy.
        """

    @abstractmethod
    def explain_last_move(self) -> str:
        """Return a human-readable explanation of the last move decision."""

