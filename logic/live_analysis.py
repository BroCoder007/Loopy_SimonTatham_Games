
import time
import threading
import concurrent.futures
from typing import Dict, Any, List, Optional
from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver

class LiveAnalysisService:
    """
    Service to run live comparative analysis of solvers in the background.
    """
    
    def __init__(self, game_state: GameState):
        self.game_state = game_state
        self.move_counter = 0

    def run_analysis(self) -> Optional[Dict[str, Any]]:
        """
        Runs the analysis for the current board state.
        Returns a dictionary with the analysis results.
        Safe to call from UI thread (spawns internal threads/futures).
        """
        # 1. Clone State (Fast, In-Memory)
        cloned_state = self.game_state.clone_for_simulation()
        self.move_counter += 1
        current_move_num = self.move_counter
        
        # 2. Define Simulation Tasks Wrapper methods
        def run_greedy_task():
             return self._run_greedy(cloned_state)
             
        def run_dp_task():
             return self._run_dp(cloned_state)
             
        def run_dnc_task():
             return self._run_dnc(cloned_state)

        # 3. Execute in Parallel with Timeout
        results = {
            "move_number": current_move_num,
            "greedy_move": "N/A", "greedy_time": 0.0, "greedy_states": 0,
            "dnc_move": "N/A", "dnc_time": 0.0, "dnc_states": 0,
            "dp_move": "N/A", "dp_time": 0.0, "dp_states": 0,
        }
        
        # Using ThreadPoolExecutor because solvers are CPU-intensive but we want to fail gracefully on timeout.
        # Python threads don't truly parallelize CPU work due to GIL, but they allow us to coordinate timeouts.
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_greedy = executor.submit(run_greedy_task)
            future_dnc = executor.submit(run_dnc_task)
            future_dp = executor.submit(run_dp_task)
            
            futures_map = {
                "greedy": future_greedy,
                "dnc": future_dnc,
                "dp": future_dp,
            }
            
            for key, future in futures_map.items():
                try:
                    # STRICT TIMEOUT: 0.25 seconds total wait per solver
                    # Note: They start roughly at same time, so this is ~0.25s wall clock for all.
                    res = future.result(timeout=0.25)
                    
                    if res:
                        results[f"{key}_move"] = str(res.get("move", "None"))
                        results[f"{key}_time"] = round(res.get("time", 0) * 1000, 2)
                        results[f"{key}_states"] = res.get("states", 0)
                        
                except concurrent.futures.TimeoutError:
                    results[f"{key}_move"] = "Timeout"
                    results[f"{key}_time"] = 250.0  # Maxed out visually
                    results[f"{key}_states"] = "N/A"
                except Exception as e:
                    results[f"{key}_move"] = "Error"
                    results[f"{key}_time"] = 0.0
                    results[f"{key}_states"] = "Err"
                    print(f"[LiveAnalysis] Error in {key}: {e}")

        # Append to history in the MAIN game state (not the clone)
        self.game_state.live_analysis_table.append(results)
        return results

    def _run_greedy(self, state_clone: GameState):
        start = time.time()
        # Instantiate fresh solver on the clone
        solver = GreedySolver(state_clone)
        
        # Run logic
        # decide_move returns (candidates, best_move)
        candidates, best_move = solver.decide_move()
        
        duration = time.time() - start
        states_explored = len(candidates) if candidates else 0
        
        return {
            "move": best_move,
            "time": duration,
            "states": states_explored
        }

    def _run_dp(self, state_clone: GameState):
        start = time.time()
        solver = DynamicProgrammingSolver(state_clone)
        
        # Force computation
        # solve() -> triggers _compute_full_solution internally
        move = solver.solve()
        
        duration = time.time() - start
        
        # Access internal metrics
        states_explored = getattr(solver, "dp_state_count", 0)
        
        return {
            "move": move,
            "time": duration,
            "states": states_explored
        }

    def _run_dnc(self, state_clone: GameState):
        start = time.time()
        solver = DivideConquerSolver(state_clone)
        
        # Force computation
        move = solver.solve()
        
        duration = time.time() - start
        
        # D&C solver does not currently expose exact state counts.
        total_states = 0

        return {
            "move": move,
            "time": duration,
            "states": total_states
        }
