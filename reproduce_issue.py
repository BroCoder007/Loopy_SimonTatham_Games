
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.advanced_dp_solver import AdvancedDPSolver

def reproduce():
    print("Initializing 5x5 GameState (Medium)...")
    # Use standard 5x5 grid which seems to fail
    game = GameState(rows=5, cols=5, difficulty="Medium", game_mode="vs_cpu", solver_strategy="advanced_dp")
    
    print(f"Solver: {game.cpu.__class__.__name__}")
    
    print("Clues generated:")
    for k, v in game.clues.items():
        print(f"  {k}: {v}")
        
    print("\nCalling decide_move()...")
    candidates, best_move = game.cpu.decide_move()
    
    if best_move:
        print(f"PASS: CPU decided move {best_move}")
    else:
        print("FAIL: CPU returned no move.")

if __name__ == "__main__":
    with open("reproduce_log.txt", "w", encoding="utf-8") as f:
        sys.stdout = f
        reproduce()
