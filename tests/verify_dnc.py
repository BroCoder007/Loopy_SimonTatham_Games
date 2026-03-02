import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.divide_conquer_solver import DivideConquerSolver

def test_base_case():
    print("\n--- Test 1: Base Case (3x3) ---")
    # 3x3 Grid
    # Clue '3' in corner (0,0) -> Forces edges ((0,0)-(0,1)) and ((0,0)-(1,0))
    game = GameState(rows=3, cols=3, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_conquer")
    game.clues = {(0,0): 3}
    
    solver = game.cpu
    if not isinstance(solver, DivideConquerSolver):
        print("FAIL: Solver is not DivideConquerSolver")
        return

    move = solver.solve(game)
    print(f"Move found: {move}")
    print(f"Explanation: {solver.explain_last_move()}")
    
    if move:
        print("PASS: Base case move found.")
    else:
        print("FAIL: No move found for corner 3.")

def test_backup_path():
    print("\n--- Test 2: Backup to Greedy ---")
    # Empty 5x5 grid. D&C might not find anything deterministic without constraints.
    game = GameState(rows=5, cols=5, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_conquer")
    game.clues = {} # No clues
    
    solver = game.cpu
    move = solver.solve(game)
    print(f"Move found: {move}")
    print(f"Explanation: {solver.explain_last_move()}")
    
    if "Backup" in solver.explain_last_move() or "Greedy" in solver.explain_last_move():
        print("PASS: Backup path triggered.")
    else:
        print("WARN: Backup path not explicitly mentioned (or D&C found a move?).")

def test_merge_logic():
    print("\n--- Test 3: Merge Logic ---")
    # 4x4 Grid.
    # We want to force a boundary edge.
    # Split is likely at row 2, col 2.
    # Let's put a '3' at (1,1) (in TL quadrant) and '3' at (1,2) (in TR quadrant).
    # The edge ((1,1)-(1,2)) is on the vertical boundary between TL and TR?
    # TL: (0,0) to (1,1). TR: (0,2) to (1,3).
    # Logic split: 4 cols -> mid=2.
    # TL: cols 0,1. TR: cols 2,3.
    # Vertical boundary is between col 1 and 2.
    # Edge ((1,1), (1,2)) crosses it.
    
    game = GameState(rows=4, cols=4, difficulty="Custom", game_mode="vs_cpu", solver_strategy="divide_conquer")
    # Clue at (1,1) is 3. Clue at (1,2) is 3.
    # Adjacent 3s usually force the shared edge and parallel edges.
    game.clues = {(1,1): 3, (1,2): 3}
    
    solver = game.cpu
    move = solver.solve(game)
    print(f"Move found: {move}")
    print(f"Explanation: {solver.explain_last_move()}")
    
    if move:
         print("PASS: Merge logic found a move (or base case did).")
    else:
         print("FAIL: No move found for adjacent 3s.")

if __name__ == "__main__":
    try:
        test_base_case()
        test_backup_path()
        test_merge_logic()
    except Exception as e:
        print(f"ERROR: {e}")
        import traceback
        traceback.print_exc()
