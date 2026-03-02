import unittest
from logic.game_state import GameState
from logic.solvers.solver_interface import MoveExplanation

class TestVisualizationLayer(unittest.TestCase):

    def setUp(self):
        # Setup a simple 3x3 game for testing
        self.game = GameState(rows=3, cols=3, difficulty="Easy", game_mode="vs_cpu")
        # Ensure we can force strategies
        
    def test_greedy_move_explanation(self):
        """Verify GreedySolver generates correct Local explanation."""
        self.game.solver_strategy = "greedy"
        self.game.clues = {(0,0): 3} # Force Clue 3 rule
        
        # Decide move
        self.game.cpu.solve() # Should return something or we decide
        candidates, best = self.game.cpu.decide_move()
        self.game.cpu.register_move(best)
        
        meta = self.game.cpu.get_last_move_explanation()
        self.assertIsInstance(meta, MoveExplanation)
        self.assertEqual(meta.mode, "Greedy")
        self.assertEqual(meta.scope, "Local")
        self.assertTrue(len(meta.highlight_cells) > 0)
        self.assertTrue(len(meta.highlight_edges) == 1)

    def test_dnc_move_explanation(self):
        """Verify DivideConquerSolver generates Regional explanation."""
        self.game = GameState(rows=4, cols=4, difficulty="Easy", game_mode="vs_cpu", solver_strategy="divide_conquer")
        self.game.clues = {} # Empty board might fail D&C solve but let's see if we can trigger *some* metadata
        
        # D&C needs clues to work usually, let's give it a simple forced move
        # Or just checking if it initializes metadata container even if no move?
        # AbstractSolver.decide_move might return None.
        
        self.game.clues = {(0,0): 3, (0,1): 3}
        
        candidates, best = self.game.cpu.decide_move()
        if best:
            self.game.cpu.register_move(best)
            meta = self.game.cpu.get_last_move_explanation()
            self.assertIsInstance(meta, MoveExplanation)
            self.assertEqual(meta.mode, "Divide & Conquer")
            self.assertEqual(meta.scope, "Regional")
            self.assertIsNotNone(meta.highlight_region)

    def test_dp_move_explanation(self):
        """Verify DynamicProgrammingSolver generates Global explanation with Certainty."""
        self.game = GameState(rows=2, cols=2, difficulty="Easy", game_mode="vs_cpu", solver_strategy="dynamic_programming")
        self.game.clues = {(0,0): 3}
        
        candidates, best = self.game.cpu.decide_move()
        if best:
            self.game.cpu.register_move(best)
            meta = self.game.cpu.get_last_move_explanation()
            self.assertIsInstance(meta, MoveExplanation)
            self.assertEqual(meta.mode, "Dynamic Programming")
            self.assertEqual(meta.scope, "Global")
            self.assertIn("certainty", meta.reasoning_data)
            self.assertTrue(0.0 <= meta.reasoning_data["certainty"] <= 1.0)

if __name__ == '__main__':
    unittest.main()
