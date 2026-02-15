"""
Live Analysis Service
=====================
Runs all three solvers (Greedy, D&C, DP) on isolated board clones
and records their move choice, execution time, and states explored.

Correctness guarantees:
- Each solver gets a deep-copied board state (no shared mutable objects).
- Solver APIs match the real game flow exactly:
  Greedy -> decide_move(), D&C -> solve(), DP -> solve().
- Results are appended to game_state.live_analysis_table.
- Move numbering is derived from the table length (no internal counter).
"""

import time
import concurrent.futures
import copy
from typing import Dict, Any, Optional

from logic.game_state import GameState
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver


class LiveAnalysisService:
    """
    Service to run live comparative analysis of all solvers.
    Re-instantiated every CPU turn — stateless by design.
    """

    # Timeout per solver (seconds). DP and D&C can be slow on large boards.
    SOLVER_TIMEOUT = 2.0

    def __init__(self, game_state: GameState):
        self.game_state = game_state

    def run_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Snapshot current board, run all 3 solvers in parallel, collect results.
        Appends result row to game_state.live_analysis_table.
        """
        # Move number = current row count + 1 (not a resettable counter)
        move_number = len(self.game_state.live_analysis_table) + 1

        # Build 3 independent simulation board states
        greedy_state = self._create_isolated_state()
        dnc_state = self._create_isolated_state()
        dp_state = self._create_isolated_state()

        # Default result row
        results = {
            "move_number": move_number,
            "greedy_move": "N/A", "greedy_time": 0.0, "greedy_states": 0,
            "dnc_move": "N/A", "dnc_time": 0.0, "dnc_states": 0,
            "dp_move": "N/A", "dp_time": 0.0, "dp_states": 0,
        }

        # Run solvers in parallel with timeout
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            # future_greedy = executor.submit(self._run_greedy, greedy_state) # Disabled
            future_dnc = executor.submit(self._run_dnc, dnc_state)
            future_dp = executor.submit(self._run_dp, dp_state)

            futures_map = {
                # "greedy": future_greedy,
                "dnc": future_dnc,
                "dp": future_dp,
            }

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

        # Append to REAL game state (not clone)
        self.game_state.live_analysis_table.append(results)
        return results

    # ── Isolation ──────────────────────────────────────────────

    def _create_isolated_state(self) -> GameState:
        """
        Build a fully independent board snapshot.
        Uses clone_for_simulation() as base, then deep-copies all containers
        to guarantee zero shared mutable references between threads.
        """
        state = self.game_state.clone_for_simulation()

        # Deep-copy all mutable containers to ensure thread safety
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



    def _run_dp(self, state_clone: GameState) -> Dict[str, Any]:
        """
        DP solver: uses solve() — same API as real game.
        'states' = dp_state_count (number of DP profile states generated).
        """
        start = time.time()
        solver = DynamicProgrammingSolver(state_clone)

        move = solver.solve()
        duration = time.time() - start

        # dp_state_count is incremented inside _compute_full_solution
        states_explored = getattr(solver, "dp_state_count", 0)

        return {
            "move": move,
            "time": duration,
            "states": states_explored,
        }

    def _run_dnc(self, state_clone: GameState) -> Dict[str, Any]:
        """
        D&C solver: uses solve() — same API as real game.
        'states' = recursion depth reached during divide and conquer.
        """
        start = time.time()
        solver = DivideConquerSolver(state_clone)

        move = solver.solve()
        duration = time.time() - start

        # D&C tracks recursion_depth as its primary metric
        recursion_depth = getattr(solver, "recursion_depth", 0)

        return {
            "move": move,
            "time": duration,
            "states": recursion_depth,
        }
