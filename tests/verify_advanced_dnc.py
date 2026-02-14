import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.divide_conquer_solver import DivideConquerSolver

def test_corner_3():
    print("\n--- Test: Corner 3 (Base Heuristic) ---")
    game = GameState(rows=5, cols=5, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_and_conquer")
    game.clues = {(0,0): 3}
    
    # Solve
    move = game.cpu.solve(game)
    print(f"Move: {move}")
    print(f"Last Explanation: {game.cpu.explain_last_move()}")
    
    if "Corner 3" in game.cpu.explain_last_move():
        print("PASS: Corner 3 logic triggered.")
    else:
        print("WARN: Corner 3 logic NOT triggered (backup path or generic logic?).")

def test_adjacent_3s():
    print("\n--- Test: Adjacent 3s (3-3) ---")
    game = GameState(rows=5, cols=5, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_and_conquer")
    # (2,2) and (2,3) are 3s.
    game.clues = {(2,2): 3, (2,3): 3}
    
    move = game.cpu.solve(game)
    print(f"Move: {move}")
    print(f"Last Explanation: {game.cpu.explain_last_move()}")
    
    if "Adjacent 3s" in game.cpu.explain_last_move():
        print("PASS: Adjacent 3s logic triggered.")
    else:
        print("WARN: Adjacent 3s logic NOT triggered.")

if __name__ == "__main__":
    test_corner_3()
    test_adjacent_3s()
