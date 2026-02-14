
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver

class TestDynamicProgrammingSolver(unittest.TestCase):
    def setUp(self):
        # Default 3x3 game
        self.game = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        self.solver = self.game.cpu

    def test_instantiation(self):
        self.assertIsInstance(self.solver, DynamicProgrammingSolver)

    def test_3x3_hint_finding(self):
        # 3x3 with random clues (default init)
        # Should either return a hint OR explain multiple/no solutions
        hint = self.solver.generate_hint()
        self.assertIn("strategy", hint)
        self.assertEqual(hint["strategy"], "Dynamic Programming (State Compression)")
        
        if hint["move"]:
            self.assertIsInstance(hint["move"], tuple) # Edge tuple
            self.assertEqual(len(hint["move"]), 2)
        else:
            # If no move, explanation should be present
            self.assertTrue(hint["explanation"])

    def test_impossible_2x2_all_3s(self):
        # 2x2 board with all 3s is unsolvable (requires 2 loops or loose ends)
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game.clues = {
            (0,0): 3, (0,1): 3,
            (1,0): 3, (1,1): 3
        }
        hint = game.cpu.generate_hint()
        self.assertIsNone(hint["move"])
        self.assertIn("No valid solutions", hint["explanation"])

    def test_solve_method(self):
        move = self.solver.solve()
        if move:
            self.assertIsInstance(move, tuple)
            self.assertEqual(len(move), 2)
        else:
            # If unsolvable generated board, None is valid
            pass

    def test_decide_move(self):
        # Determine move and candidates
        candidates, best_move = self.solver.decide_move()
        
        self.assertIsInstance(candidates, list)
        if best_move:
            self.assertIsInstance(best_move, tuple)
            # Candidates should contain best_move
            self.assertTrue(any(c[0] == best_move for c in candidates))
            self.assertEqual(len(candidates), 1) # DP only returns best move
        else:
            self.assertIsNone(best_move)
            self.assertEqual(len(candidates), 0)

if __name__ == '__main__':
    unittest.main()
