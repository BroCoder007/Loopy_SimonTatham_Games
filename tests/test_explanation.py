
import unittest
from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver

class TestExplanation(unittest.TestCase):
    def test_greedy_explanation_tracking(self):
        # 1. Setup
        game = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="greedy")
        solver = GreedySolver(game)
        
        # Ensure CPU has some move to make.
        # Initialize board weights to predictable values or just rely on default.
        
        # 2. Force CPU Move
        move = solver.make_move()
        
        # 3. Verify
        print(f"CPU Move: {move}")
        print(f"Last Move Info: {game.last_cpu_move_info}")
        
        self.assertIsNotNone(game.last_cpu_move_info, "last_cpu_move_info should be populated")
        self.assertEqual(game.last_cpu_move_info["move"], move)
        self.assertEqual(game.last_cpu_move_info["strategy"], "Greedy")
        
        explanation = game.last_cpu_move_info["explanation"]
        self.assertTrue(explanation.startswith("Using Greedy Strategy:"), "Explanation should start with correct prefix")
        self.assertTrue(len(explanation) > 25, "Explanation should contain reasoning text")

if __name__ == "__main__":
    unittest.main()
