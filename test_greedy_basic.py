from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver
import time

def test_greedy_basics():
    # Create a 3x3 grid with a single 0-clue
    print("Test 1: Single 0-clue")
    gs = GameState(3, 3, "Hard", "vs_cpu", "greedy")
    # Manually set clues: (1,1) is 0
    gs.clues = {(1, 1): 0}
    gs.solution_edges = set() # Doesn't matter for solver
    
    solver = GreedySolver(gs)
    move = solver.solve()
    print(f"Move found: {move}")
    if move:
        print("SUCCESS: Greedy found a move on 0-clue")
    else:
        print("FAILURE: Greedy missed 0-clue")

    # Create a 3x3 grid with a single 3-clue
    print("\nTest 2: Single 3-clue")
    gs2 = GameState(3, 3, "Hard", "vs_cpu", "greedy")
    gs2.clues = {(1, 1): 3}
    
    solver2 = GreedySolver(gs2)
    move2 = solver2.solve()
    print(f"Move found: {move2}")
    if move2:
        print("SUCCESS: Greedy found a move on 3-clue")
    else:
        print("FAILURE: Greedy missed 3-clue")

test_greedy_basics()
