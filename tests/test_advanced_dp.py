
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.advanced_dp_solver import AdvancedDPSolver

def test_advanced_dp():
    print("Initializing GameState (4x4, Medium)...")
    # Use a small grid for deterministic testing
    game = GameState(rows=4, cols=4, difficulty="Medium", game_mode="vs_cpu", solver_strategy="advanced_dp")
    
    print(f"Solver Strategy: {game.cpu.__class__.__name__}")
    assert isinstance(game.cpu, AdvancedDPSolver)
    
    print("Generating Solution...")
    # Force computation
    game.cpu._compute_full_solution()
    
    if not game.cpu._final_solution_edges:
        print("FAIL: No solution found.")
        # Print clues for debugging
        print("Clues:", game.clues)
        return
        
    print(f"PASS: Solution found with {len(game.cpu._final_solution_edges)} edges.")
    print("Solution Edges:", game.cpu._final_solution_edges)
    
    # Verify Hint Generation
    print("Testing Hint Generation...")
    hint = game.cpu.generate_hint()
    print("Hint:", hint)
    
    if hint['move']:
        print("PASS: Hint generated.")
    else:
        print("WARNING: No hint generated (Puzzle might be solved or solver stuck).")

if __name__ == "__main__":
    with open("test_output.log", "w", encoding="utf-8") as f:
        sys.stdout = f
        test_advanced_dp()

