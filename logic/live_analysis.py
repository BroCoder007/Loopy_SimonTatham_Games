
import time
import threading
import concurrent.futures
import copy
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
        # 1. Build independent simulation states (one per thread/solver)
        greedy_state = self._create_isolated_simulation_state()
        dnc_state = self._create_isolated_simulation_state()
        dp_state = self._create_isolated_simulation_state()

        # Debug safety assertions: no shared simulation graph objects.
        assert id(greedy_state.graph) != id(dnc_state.graph)
        assert id(greedy_state.graph) != id(dp_state.graph)
        assert id(dnc_state.graph) != id(dp_state.graph)

        # Additional mutable-container isolation checks.
        assert id(greedy_state.graph.edges) != id(dnc_state.graph.edges)
        assert id(greedy_state.graph.edges) != id(dp_state.graph.edges)
        assert id(dnc_state.graph.edges) != id(dp_state.graph.edges)
        assert id(greedy_state.graph.vertices) != id(dnc_state.graph.vertices)
        assert id(greedy_state.graph.vertices) != id(dp_state.graph.vertices)
        assert id(dnc_state.graph.vertices) != id(dp_state.graph.vertices)
        assert id(greedy_state.clues) != id(dnc_state.clues)
        assert id(greedy_state.clues) != id(dp_state.clues)
        assert id(dnc_state.clues) != id(dp_state.clues)
        assert id(greedy_state.solution_edges) != id(dnc_state.solution_edges)
        assert id(greedy_state.solution_edges) != id(dp_state.solution_edges)
        assert id(dnc_state.solution_edges) != id(dp_state.solution_edges)
        self.move_counter += 1
        current_move_num = self.move_counter
        
        # 2. Define Simulation Tasks Wrapper methods
        def run_greedy_task():
             return self._run_greedy(greedy_state)
             
        def run_dp_task():
             return self._run_dp(dp_state)
             
        def run_dnc_task():
             return self._run_dnc(dnc_state)

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

    def _create_isolated_simulation_state(self) -> GameState:
        """
        Build a thread-local simulation state with no shared mutable references.
        """
        state = self.game_state.clone_for_simulation()

        # Ensure graph is deeply isolated and adjacency is consistent with edges.
        state.graph = self.game_state.graph.copy()

        # Ensure no shared mutable containers from source state.
        state.clues = copy.deepcopy(self.game_state.clues)
        state.edge_weights = copy.deepcopy(self.game_state.edge_weights)
        state.solution_edges = copy.deepcopy(getattr(self.game_state, "solution_edges", set()))

        # If DSU/cache-like structures exist, isolate them as well.
        if hasattr(self.game_state, "dsu"):
            state.dsu = copy.deepcopy(self.game_state.dsu)

        return state

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
