"""
Live Analysis Service
=====================
Runs the comparative analysis of the active solver strategy:
"Before Backtracking" (Legacy) vs "After Backtracking" (Current).
"""

import time
import concurrent.futures
import copy
from typing import Dict, Any, Optional

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

    # Timeout per solver (seconds). 3s allows D&C/DP to finish on Hard 5x5.
    SOLVER_TIMEOUT = 3.0

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def run_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Snapshot current board, run Current vs Legacy solver in parallel.
        Appends result row to game_state.live_analysis_table.
        """
        if self.game_state.solver_strategy == "greedy":
            return None # Greedy is not compared since backtracking was not integrated into it

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
        }

        legacy_too_large = (self.game_state.rows * self.game_state.cols) > 25

        # Run solvers in parallel with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            if not legacy_too_large:
                future_legacy = executor.submit(self._run_legacy, legacy_state)
            else:
                future_legacy = None
                
            future_current = executor.submit(self._run_current, current_state)

            futures_map = {"current": future_current}
            if future_legacy:
                futures_map["legacy"] = future_legacy

            for key, future in futures_map.items():
                try:
                    res = future.result(timeout=self.SOLVER_TIMEOUT)
                    if res:
                        results[f"{key}_move"] = str(res.get("move", "None"))
                        results[f"{key}_time"] = round(res.get("time", 0) * 1000, 2)
                        results[f"{key}_states"] = res.get("states", 0)
                except concurrent.futures.TimeoutError:
                    results[f"{key}_move"] = "Timeout"
                    results[f"{key}_time"] = round(self.SOLVER_TIMEOUT * 1000, 1)
                    results[f"{key}_states"] = "N/A"
                except Exception as e:
                    results[f"{key}_move"] = "Error"
                    results[f"{key}_time"] = 0.0
                    results[f"{key}_states"] = "Err"
                    print(f"[LiveAnalysis] Error in {key}: {e}")

        if legacy_too_large:
            results["legacy_move"] = "Skipped"
            results["legacy_time"] = 0.0
            results["legacy_states"] = "N/A (>5x5)"

        self.game_state.live_analysis_table.append(results)
        return results

    # ── Isolation ──────────────────────────────────────────────

    def _create_isolated_state(self) -> GameState:
        """
        Build a fully independent board snapshot.
        """
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
        Runs the exact same solver but without Backtracking improvements.
        """
        start = time.perf_counter()
        solver = self._get_solver_for_strategy(state_clone, is_legacy=True)
        
        move = solver.solve()
        duration = time.perf_counter() - start

        states_explored = 0
        if self.game_state.solver_strategy == "dynamic_programming":
             states_explored = getattr(solver, "dp_state_count", 0)
        elif self.game_state.solver_strategy == "advanced_dp":
             stats = getattr(solver, "_merge_stats", {})
             if "region_stats" in stats and stats["region_stats"]:
                  states_explored = max(r.get("states", 0) for r in stats["region_stats"].values())
        else:
             states_explored = getattr(solver, "recursion_depth", 0)

        return {
            "move": move,
            "time": duration,
            "states": states_explored,
        }

    def _run_current(self, state_clone: GameState) -> Dict[str, Any]:
        """
        Runs the optimized backtracking integrated solver.
        """
        start = time.perf_counter()
        solver = self._get_solver_for_strategy(state_clone, is_legacy=False)

        move = solver.solve()
        duration = time.perf_counter() - start

        states_explored = 0
        if self.game_state.solver_strategy == "dynamic_programming":
             states_explored = getattr(solver, "dp_state_count", 0)
        elif self.game_state.solver_strategy == "advanced_dp":
             stats = getattr(solver, "_merge_stats", {})
             if "region_stats" in stats and stats["region_stats"]:
                  states_explored = max(r.get("states", 0) for r in stats["region_stats"].values())
        else:
             states_explored = getattr(solver, "recursion_depth", 0)

        return {
            "move": move,
            "time": duration,
            "states": states_explored,
        }

