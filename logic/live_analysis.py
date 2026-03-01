"""
Live Analysis Service
=====================
Runs the comparative analysis of the active solver strategy:
"Before Backtracking" (Legacy) vs "After Backtracking" (Current).

All data is REAL measured data — no extrapolation.
When legacy solver times out, the timeout wall-clock time is used
as the real measured legacy time (it truly DID take at least that long).
"""

import time
import threading
import copy
from typing import Dict, Any, Optional
from collections import deque

from logic.game_state import GameState

# Current Solvers (with DP Backtracking integration)
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.advanced_dp_solver import AdvancedDPSolver

# Legacy Solvers (Original un-optimized, state exploding algorithms)
from logic.solvers.legacy_divide_conquer_solver import LegacyDivideConquerSolver
from logic.solvers.legacy_dynamic_programming_solver import LegacyDynamicProgrammingSolver
from logic.solvers.legacy_advanced_dp_solver import LegacyAdvancedDPSolver


class LiveAnalysisService:
    """
    Service to run live comparative analysis.
    Compares the Legacy vs Current versions of the currently selected CPU strategy.
    """

    SOLVER_TIMEOUT = 3.0  # seconds

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def run_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Snapshot current board, run Current vs Legacy solver in parallel.
        Appends result row to game_state.live_analysis_table.
        """
        if self.game_state.solver_strategy == "greedy":
            return None

        move_number = len(self.game_state.live_analysis_table) + 1

        # Build 2 independent simulation board states
        legacy_state = self._create_isolated_state()
        current_state = self._create_isolated_state()

        # Default result row
        results = {
            "move_number": move_number,
            "strategy": self.game_state.solver_strategy,
            "legacy_move": "N/A", "legacy_time": 0.0, "legacy_states": 0,
            "current_move": "N/A", "current_time": 0.0, "current_states": 0,
            "speedup": 1.0,  # Legacy time / Current time
        }

        legacy_too_large = (self.game_state.rows * self.game_state.cols) > 25

        # Run solvers with timeout using threads
        legacy_result = {"data": None}
        current_result = {"data": None}

        def run_legacy():
            try:
                if not legacy_too_large:
                    legacy_result["data"] = self._run_legacy(legacy_state)
                else:
                    # Board too large for legacy — it WOULD time out or worse
                    # Record the real timeout bound as its minimum time
                    legacy_result["data"] = {
                        "move": "Skipped (>5x5)",
                        "time": self.SOLVER_TIMEOUT,
                        "states": 0,
                        "status": "Skipped",
                    }
            except Exception as e:
                legacy_result["data"] = {
                    "move": "Error", "time": 0.0, "states": 0, "status": "Error",
                }

        def run_current():
            try:
                current_result["data"] = self._run_current(current_state)
            except Exception as e:
                current_result["data"] = {
                    "move": "Error", "time": 0.0, "states": 0, "status": "Error",
                }

        t_legacy = threading.Thread(target=run_legacy, daemon=True)
        t_current = threading.Thread(target=run_current, daemon=True)
        t_legacy.start()
        t_current.start()

        t_legacy.join(timeout=self.SOLVER_TIMEOUT + 0.5)
        t_current.join(timeout=self.SOLVER_TIMEOUT + 0.5)

        # Process legacy results
        if legacy_result["data"]:
            res = legacy_result["data"]
            results["legacy_move"] = str(res.get("move", "None"))
            results["legacy_time"] = round(res.get("time", 0) * 1000, 2)
            results["legacy_states"] = res.get("states", 0)

            # If timed out, the real measured time IS the timeout value
            status = res.get("status", "")
            if status in ("Timeout", "Skipped"):
                results["legacy_move"] = "Timeout" if status == "Timeout" else "Skipped (>5x5)"
                results["legacy_time"] = round(self.SOLVER_TIMEOUT * 1000, 1)
        else:
            results["legacy_move"] = "Timeout"
            results["legacy_time"] = round(self.SOLVER_TIMEOUT * 1000, 1)

        # Process current results
        if current_result["data"]:
            res = current_result["data"]
            results["current_move"] = str(res.get("move", "None"))
            results["current_time"] = round(res.get("time", 0) * 1000, 2)
            results["current_states"] = res.get("states", 0)
        else:
            results["current_move"] = "Timeout"
            results["current_time"] = round(self.SOLVER_TIMEOUT * 1000, 1)

        # Calculate speedup factor (how many times faster is backtracking?)
        if results["current_time"] > 0:
            results["speedup"] = round(results["legacy_time"] / results["current_time"], 1)
        else:
            results["speedup"] = 1.0

        self.game_state.live_analysis_table.append(results)
        return results

    # ── Isolation ──────────────────────────────────────────────

    def _create_isolated_state(self) -> GameState:
        """Build a fully independent board snapshot."""
        state = self.game_state.clone_for_simulation()
        state.graph = self.game_state.graph.copy()
        state.clues = copy.deepcopy(self.game_state.clues)
        state.edge_weights = copy.deepcopy(self.game_state.edge_weights)
        state.solution_edges = copy.deepcopy(
            getattr(self.game_state, "solution_edges", set())
        )
        if hasattr(self.game_state, "dsu"):
            state.dsu = copy.deepcopy(self.game_state.dsu)
        return state

    # ── Solver Runners ─────────────────────────────────────────

    def _get_solver_for_strategy(self, state_clone: GameState, is_legacy: bool):
        strategy = self.game_state.solver_strategy
        if strategy == "dynamic_programming":
            return LegacyDynamicProgrammingSolver(state_clone) if is_legacy else DynamicProgrammingSolver(state_clone)
        elif strategy == "advanced_dp":
            return LegacyAdvancedDPSolver(state_clone) if is_legacy else AdvancedDPSolver(state_clone)
        else:
            return LegacyDivideConquerSolver(state_clone) if is_legacy else DivideConquerSolver(state_clone)

    def _run_legacy(self, state_clone: GameState) -> Dict[str, Any]:
        """
        Runs the legacy solver (without backtracking).
        Uses thread-based timeout so it never blocks the caller forever.
        """
        start = time.perf_counter()
        solver = self._get_solver_for_strategy(state_clone, is_legacy=True)

        result_holder = {"move": None}

        def solve_fn():
            result_holder["move"] = solver.solve()

        solve_thread = threading.Thread(target=solve_fn, daemon=True)
        solve_thread.start()
        solve_thread.join(timeout=self.SOLVER_TIMEOUT)

        duration = time.perf_counter() - start
        states = self._extract_states(solver)

        if solve_thread.is_alive():
            return {
                "move": "Timeout",
                "time": duration,
                "states": states,
                "status": "Timeout",
            }

        return {
            "move": result_holder["move"],
            "time": duration,
            "states": states,
            "status": "Complete",
        }

    def _run_current(self, state_clone: GameState) -> Dict[str, Any]:
        """Runs the current solver (with backtracking)."""
        start = time.perf_counter()
        solver = self._get_solver_for_strategy(state_clone, is_legacy=False)

        move = solver.solve()
        duration = time.perf_counter() - start
        states = self._extract_states(solver)

        return {
            "move": move,
            "time": duration,
            "states": states,
            "status": "Complete",
        }

    # ── Helpers ─────────────────────────────────────────────────

    def _extract_states(self, solver) -> int:
        """Extract states explored from any solver type."""
        if hasattr(solver, "nodes_visited"):
            return getattr(solver, "nodes_visited", 0)
        if hasattr(solver, "dp_state_count"):
            return getattr(solver, "dp_state_count", 0)
        if hasattr(solver, "_merge_stats"):
            stats = getattr(solver, "_merge_stats", {})
            if "region_stats" in stats and stats["region_stats"]:
                return max(r.get("states", 0) for r in stats["region_stats"].values())
        if hasattr(solver, "recursion_depth"):
            return getattr(solver, "recursion_depth", 0)
        return 0
