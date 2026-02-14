"""
Strategy Controller
===================
Hierarchical orchestration layer for CPU move selection.

Rules:
- Keep solver implementations pure.
<<<<<<< feature/loop-solver-improvements
- In DP mode: call ONLY the DP solver (no fallback chain).
- In D&C mode: D&C solver -> Greedy fallback.
- In Greedy mode: Greedy solver only.
=======
- Apply fallback only at controller level.
>>>>>>> main
- Re-evaluate full priority order on every CPU turn.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.greedy_solver import GreedySolver

<<<<<<< feature/loop-solver-improvements

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
=======
Move = Tuple[Tuple[int, int], Tuple[int, int]]


class StrategyController:
    def __init__(self, game_state: Any, selected_mode: str):
        self.game_state = game_state
        self.selected_mode = self._normalize_mode(selected_mode)

        # Reuse currently attached solver instance when compatible.
        cpu_solver = getattr(game_state, "cpu", None)
        self.dp_solver = cpu_solver if isinstance(cpu_solver, DynamicProgrammingSolver) else DynamicProgrammingSolver(game_state)
        self.advanced_dp_solver = cpu_solver if isinstance(cpu_solver, AdvancedDPSolver) else AdvancedDPSolver(game_state)
        self.dnc_solver = cpu_solver if isinstance(cpu_solver, DivideConquerSolver) else DivideConquerSolver(game_state)
        self.greedy_solver = cpu_solver if isinstance(cpu_solver, GreedySolver) else GreedySolver(game_state)

    def get_next_cpu_move(self) -> Tuple[Optional[Move], str]:
        if self.selected_mode == "dp":
            move = self._try_solver(self.dp_solver)
            if move is not None:
                return move, "DP"

            # Still within DP family: advanced DP decomposition before any non-DP fallback.
            move = self._try_solver(self.advanced_dp_solver)
            if move is not None:
                return move, "DP (Advanced)"

            move = self._try_solver(self.dnc_solver)
            if move is not None:
                return move, "D&C (Fallback)"

            move = self._try_solver(self.greedy_solver)
            if move is not None:
                return move, "Greedy (Final Fallback)"

            return None, "No moves available"

        if self.selected_mode == "dnc":
            move = self._try_solver(self.dnc_solver)
            if move is not None:
                return move, "D&C"

            move = self._try_solver(self.greedy_solver)
            if move is not None:
                return move, "Greedy (Fallback)"

            return None, "No moves available"

        move = self._try_solver(self.greedy_solver)
        if move is not None:
            return move, "Greedy"

        return None, "No moves available"

    def get_solver_for_source(self, source: str):
>>>>>>> main
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
<<<<<<< feature/loop-solver-improvements
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
=======
        if self.selected_mode == "dp":
            if source == "D&C (Fallback)":
                return "No DP move available. Switched to D&C for this move."
            if source == "Greedy (Final Fallback)":
                return "No DP or D&C move available. Switched to Greedy for this move."

        if self.selected_mode == "dnc" and source == "Greedy (Fallback)":
            return "No D&C move available. Switched to Greedy for this move."

        return None

    def _try_solver(self, solver: Any) -> Optional[Move]:
>>>>>>> main
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
<<<<<<< feature/loop-solver-improvements
        """Normalize mode string to internal representation."""
=======
>>>>>>> main
        if mode in ("dynamic_programming", "dp"):
            return "dp"
        if mode in ("divide_conquer", "divide_and_conquer", "dnc"):
            return "dnc"
        return "greedy"
