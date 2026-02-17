
import sys
import os
import time
import csv
import argparse
import concurrent.futures
from typing import Dict, Any, List

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver

def run_single_game(game_id: int, rows: int, cols: int, difficulty: str) -> Dict[str, Any]:
    """
    Runs a single game instance with all 3 solvers on isolated clones.
    """
    # 1. Generate a Base Game
    base_state = GameState(rows=rows, cols=cols, difficulty=difficulty, game_mode="vs_cpu")
    # Force generation (normally happens in __init__ but let's be sure)
    if not base_state.clues:
        base_state._generate_clues()

    # 2. Prepare Results Container
    base_state.cpu = None # Detach default solver
    
    # Disable DP debug logging for performance
    import logic.solvers.dynamic_programming_solver
    logic.solvers.dynamic_programming_solver.DEBUG_MODE = False

    result = {
        "game_id": game_id,
        "difficulty": difficulty,
        "size": f"{rows}x{cols}",
        "greedy_solved": False, "greedy_time": 0.0, "greedy_moves": 0,
        "dnc_solved": False, "dnc_time": 0.0, "dnc_moves": 0,
        "dp_solved": False, "dp_time": 0.0, "dp_moves": 0,
    }

    # 3. Define Solver Runner
    def run_solver(solver_class, state_clone):
        solver = solver_class(state_clone)
        start_time = time.time()
        moves_count = 0
        solved = False
        
        try:
            # Run until solved or stuck
            max_moves = (rows * cols + rows + cols) * 2 # Safety limit
            while moves_count < max_moves:
                if isinstance(solver, GreedySolver):
                    # Greedy uses decide_move or make_move
                    candidates, best_move = solver.decide_move()
                    move = best_move
                else:
                    # D&C and DP use solve()
                    move = solver.solve()

                if move is None:
                    break
                
                # Force turn to CPU to bypass checking (since we are simulating a full-solve)
                state_clone.turn = "Player 2 (CPU)"
                
                # Apply move
                u, v = move
                if not state_clone.make_move(u, v, is_cpu=True):
                    # print(f"  {solver_class.__name__} move rejected by GameState: {move}") # Optional debug
                    break
                moves_count += 1
                
                if state_clone.game_over:
                    solved = (state_clone.winner == "Player 2 (CPU)" or "CPU" in str(state_clone.winner))
                    break
                    
        except Exception as e:
            print(f"Error in game {game_id} with {solver_class.__name__}: {e}")
            
        return solved, time.time() - start_time, moves_count

    # 4. Run Solvers (Sequential or Parallel - Sequential is safer for benchmarking consistency)
    
    # Greedy
    greedy_state = base_state.clone_for_simulation()
    # Need to re-init solver logic wrapper if needed, but direct solver class usage is better
    res_greedy = run_solver(GreedySolver, greedy_state)
    result["greedy_solved"], result["greedy_time"], result["greedy_moves"] = res_greedy

    # D&C
    dnc_state = base_state.clone_for_simulation()
    res_dnc = run_solver(DivideConquerSolver, dnc_state)
    result["dnc_solved"], result["dnc_time"], result["dnc_moves"] = res_dnc

    # DP
    dp_state = base_state.clone_for_simulation()
    res_dp = run_solver(DynamicProgrammingSolver, dp_state)
    result["dp_solved"], result["dp_time"], result["dp_moves"] = res_dp

    return result

def main():
    parser = argparse.ArgumentParser(description="Benchmark Loopy Solvers")
    parser.add_argument("--games", type=int, default=10, help="Number of games to run")
    parser.add_argument("--rows", type=int, default=5, help="Rows")
    parser.add_argument("--cols", type=int, default=5, help="Cols")
    parser.add_argument("--difficulty", type=str, default="Medium", help="Difficulty")
    parser.add_argument("--output", type=str, default="benchmark_results.csv", help="Output CSV file")
    
    args = parser.parse_args()
    
    print(f"Starting Benchmark: {args.games} games, {args.rows}x{args.cols}, {args.difficulty}")
    
    results = []
    failed_greedy = 0
    
    # Run games
    for i in range(args.games):
        print(f"Running Game {i+1}/{args.games}...", end="\r")
        res = run_single_game(i+1, args.rows, args.cols, args.difficulty)
        results.append(res)
        
        if not res["greedy_solved"] and (res["dnc_solved"] or res["dp_solved"]):
            failed_greedy += 1
            
    print(f"\nBenchmark Complete!")
    print(f"Games where Greedy failed but Advanced Solvers succeeded: {failed_greedy}/{args.games}")

    # Save to CSV
    keys = results[0].keys()
    with open(args.output, "w", newline="") as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(results)
        
    print(f"Results saved to {args.output}")

    # Print Summary Table
    print("\nSummary Statistics:")
    print(f"{'Solver':<10} | {'Success Rate':<12} | {'Avg Time (s)':<12} | {'Avg Moves':<10}")
    print("-" * 50)
    
    for solver in ["greedy", "dnc", "dp"]:
        solved_count = sum(1 for r in results if r[f"{solver}_solved"])
        avg_time = sum(r[f"{solver}_time"] for r in results) / len(results)
        avg_moves = sum(r[f"{solver}_moves"] for r in results) / len(results)
        success_rate = (solved_count / len(results)) * 100
        
        print(f"{solver.upper():<10} | {success_rate:>11.1f}% | {avg_time:>12.4f} | {avg_moves:>10.1f}")

if __name__ == "__main__":
    main()
