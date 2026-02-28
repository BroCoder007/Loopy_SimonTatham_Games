import sys
from logic.game_state import GameState
from logic.solvers.dp_backtracking_solver import DPBacktrackingSolver

for _ in range(5):
    game = GameState(rows=3, cols=3, game_mode="expert", generator_type="dp")
    print(f"Clues: {game.clues}")
    
    solver = DPBacktrackingSolver(game)
    res = solver.solve(ignore_current_edges=True)
    print(f"Solve success: {res['success']}")
    if not res['success']:
        print("FAIL!")
