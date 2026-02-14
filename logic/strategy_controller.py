"""
Strategy Controller
===================
Hierarchical orchestration layer for CPU move selection.

Rules:
- Keep solver implementations pure.
- In DP mode: call ONLY the DP solver (no fallback chain).
- In D&C mode: D&C solver -> Greedy fallback.
- In Greedy mode: Greedy solver only.
- Re-evaluate full priority order on every CPU turn.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.greedy_solver import GreedySolver


class StrategyController:
    def __init__(self, game_state: Any, mode: str):
        self.game_state = game_state
        self.selected_mode = self._normalize_mode(mode)

        self.dp_solver = DynamicProgrammingSolver(game_state)
        self.advanced_dp_solver = AdvancedDPSolver(game_state)
        self.dnc_solver = DivideConquerSolver(game_state)
        self.greedy_solver = GreedySolver(game_state)

    def get_next_cpu_move(self) -> Tuple[Optional[Any], str]:
        """
        Route CPU move to the correct solver based on selected mode.

        DP mode: calls ONLY the DP solver. No fallback chain.
                 DP is guaranteed to never return None.
        D&C mode: D&C solver -> Greedy fallback.
        Greedy mode: Greedy solver only.

        Returns:
            (move, source_label)
        """
        if self.selected_mode == "dp":
            # DP mode: ONLY DP solver, no fallback.
            # DP solver is guaranteed to return a move (never None).
            move = self._try_solver(self.dp_solver)
            if move is not None:
                return move, "DP"
            # This should never happen, but handle gracefully
            return None, "No moves available"

        if self.selected_mode == "dnc":
            # D&C mode: D&C -> Greedy fallback
            move = self._try_solver(self.dnc_solver)
            if move is not None:
                return move, "D&C"
            move = self._try_solver(self.greedy_solver)
            if move is not None:
                return move, "Greedy (Fallback)"
            return None, "No moves available"

        # Greedy mode
        move = self._try_solver(self.greedy_solver)
        if move is not None:
            return move, "Greedy"
        return None, "No moves available"

    def get_solver_for_source(self, source: str) -> Optional[Any]:
        """Return the solver instance that produced the move."""
        if source.startswith("DP"):
            if source.startswith("DP (Advanced)"):
                return self.advanced_dp_solver
            return self.dp_solver
        if source.startswith("D&C"):
            return self.dnc_solver
        if source.startswith("Greedy"):
            return self.greedy_solver
        return None

    def get_fallback_message(self, source: str) -> Optional[str]:
        """
        Return a user-facing fallback message, or None if no fallback occurred.

        DP mode: NEVER returns a fallback message (no fallback exists).
        """
        if self.selected_mode == "dp":
            # No fallback in DP mode
            return None

        if self.selected_mode == "dnc":
            if source == "Greedy (Fallback)":
                return "No D&C move available. Switched to Greedy for this move."

        return None

    def _try_solver(self, solver: Any) -> Optional[Any]:
        """
        Attempt to get a move from a solver.
        Uses decide_move() if available, otherwise solve().
        """
        if solver is None:
            return None

        try:
            if hasattr(solver, "decide_move"):
                _, move = solver.decide_move()
                return move

            if hasattr(solver, "solve"):
                return solver.solve()
        except Exception:
            return None

        return None

    def _normalize_mode(self, mode: str) -> str:
        """Normalize mode string to internal representation."""
        if mode in ("dynamic_programming", "dp"):
            return "dp"
        if mode in ("divide_conquer", "divide_and_conquer", "dnc"):
            return "dnc"
        return "greedy"
