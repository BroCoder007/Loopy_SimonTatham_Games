import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.divide_conquer_solver import DivideConquerSolver

def test_explanation_backup():
    print("\n--- Test Explanation: Backup Path ---")
    game = GameState(rows=5, cols=5, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_conquer")
    game.clues = {} # Empty, forces backup path
    
    # Simulate UI call flow
    move = game.cpu.solve(game)
    game.make_move(move[0], move[1], is_cpu=True)
    game.cpu.register_move(move) # UI calls this
    
    info = game.last_cpu_move_info
    print(f"Strategy: {info['strategy']}")
    print(f"Explanation: {info['explanation']}")
    
    if "Divide & Conquer" in info['strategy']:
        print("PASS: Strategy detected correctly.")
    else:
        print("FAIL: Strategy is NOT Divide & Conquer.")

def test_explanation_dnc():
    print("\n--- Test Explanation: D&C Move ---")
    game = GameState(rows=3, cols=3, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_conquer")
    game.clues = {(0,0): 3} # Forces D&C move
    
    # Simulate UI call flow
    move = game.cpu.solve(game)
    game.make_move(move[0], move[1], is_cpu=True)
    game.cpu.register_move(move) # UI calls this
    
    info = game.last_cpu_move_info
    print(f"Strategy: {info['strategy']}")
    print(f"Explanation: {info['explanation']}")
    
    if info['strategy'] == "Divide & Conquer":
        print("PASS: Strategy detected correctly.")
    else:
        print(f"FAIL: Strategy is {info['strategy']}")

if __name__ == "__main__":
    test_explanation_backup()
    test_explanation_dnc()
