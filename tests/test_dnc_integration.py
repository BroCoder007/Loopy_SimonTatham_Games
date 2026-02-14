import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.greedy_solver import GreedySolver

def test_strategy_instantiation():
    print("\n--- Test 1: Strategy Instantiation ---")
    
    # 1. Test "divide_and_conquer" key (matching UI)
    game = GameState(rows=3, cols=3, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_and_conquer")
    print(f"Strategy 'divide_and_conquer' -> CPU is {type(game.cpu).__name__}")
    
    if isinstance(game.cpu, DivideConquerSolver):
        print("PASS: Correctly instantiated DivideConquerSolver.")
    else:
        print("FAIL: Expected DivideConquerSolver.")

    # 2. Test default/greedy
    game_greedy = GameState(rows=3, cols=3, difficulty="Custom", game_mode="vs_cpu", solver_strategy="greedy")
    print(f"Strategy 'greedy' -> CPU is {type(game_greedy.cpu).__name__}")
    
    if isinstance(game_greedy.cpu, GreedySolver):
        print("PASS: Correctly instantiated GreedySolver.")
    else:
        print("FAIL: Expected GreedySolver.")

def test_decide_move_interface():
    print("\n--- Test 2: decide_move Interface ---")
    game = GameState(rows=3, cols=3, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_and_conquer")
    # Force a move
    game.clues = {(0,0): 3}
    
    try:
        candidates, best_move = game.cpu.decide_move()
        print(f"decide_move returned: candidates={candidates}, best_move={best_move}")
        
        if best_move:
            print("PASS: decide_move successfully returned a move.")
        else:
             print("WARN: decide_move returned None (might be valid if no move found).")
             
    except Exception as e:
        print(f"FAIL: decide_move crashed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_strategy_instantiation()
    test_decide_move_interface()
