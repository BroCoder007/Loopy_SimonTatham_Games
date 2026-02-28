from logic.game_state import GameState
from logic.live_analysis import LiveAnalysisService
import time

print("Setting up state...")
game = GameState(rows=3, cols=3, game_mode="expert", generator_type="dp")
game.solver_strategy = "divide_conquer"
game._normalize_solver_strategy("divide_conquer")

analysis = LiveAnalysisService(game)

# Simulate User Move
edges = list(game.graph.edges)
if game.solution_edges:
    move = list(game.solution_edges)[0]
    game.make_move(move[0], move[1], is_cpu=False)
    
print("Running live analysis...")
res = analysis.run_analysis()
print("Analysis Output:")
for k, v in res.items():
    print(f"  {k}: {v}")

print("Testing DP strategy...")
game.solver_strategy = "dynamic_programming"
res2 = analysis.run_analysis()
print("Analysis Output:")
for k, v in res2.items():
    print(f"  {k}: {v}")
    
print("Testing Advanced DP strategy...")
game.solver_strategy = "advanced_dp"
res3 = analysis.run_analysis()
print("Analysis Output:")
for k, v in res3.items():
    print(f"  {k}: {v}")
    
print("Success.")
