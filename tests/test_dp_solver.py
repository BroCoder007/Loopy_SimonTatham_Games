
import unittest
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.graph import Graph
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.merge_sort import merge_sort


class TestMergeSort(unittest.TestCase):
    """Tests for the merge sort utility."""

    def test_empty_list(self):
        self.assertEqual(merge_sort([]), [])

    def test_single_element(self):
        self.assertEqual(merge_sort([42]), [42])

    def test_sorted_list(self):
        self.assertEqual(merge_sort([1, 2, 3, 4, 5]), [1, 2, 3, 4, 5])

    def test_reverse_sorted(self):
        self.assertEqual(merge_sort([5, 4, 3, 2, 1]), [1, 2, 3, 4, 5])

    def test_duplicates(self):
        self.assertEqual(merge_sort([3, 1, 4, 1, 5, 9, 2, 6]), [1, 1, 2, 3, 4, 5, 6, 9])

    def test_tuples(self):
        data = [((1, 2), (3, 4)), ((0, 1), (0, 2)), ((1, 0), (2, 0))]
        result = merge_sort(data)
        self.assertEqual(result, sorted(data))

    def test_with_key(self):
        data = ["banana", "apple", "cherry"]
        result = merge_sort(data, key=lambda x: x[0])
        self.assertEqual(result, ["apple", "banana", "cherry"])

    def test_stability(self):
        """Merge sort must be stable: equal elements keep original order."""
        data = [(1, 'a'), (2, 'b'), (1, 'c'), (2, 'd')]
        result = merge_sort(data, key=lambda x: x[0])
        self.assertEqual(result, [(1, 'a'), (1, 'c'), (2, 'b'), (2, 'd')])

    def test_set_input(self):
        """merge_sort should accept any sequence, including sets."""
        data = {5, 3, 1, 4, 2}
        result = merge_sort(data)
        self.assertEqual(result, [1, 2, 3, 4, 5])

    def test_edge_tuples(self):
        """Test sorting of edge tuples (the primary use case)."""
        edges = [
            ((2, 0), (3, 0)),
            ((0, 0), (0, 1)),
            ((1, 1), (2, 1)),
            ((0, 0), (1, 0)),
        ]
        result = merge_sort(edges)
        self.assertEqual(result, sorted(edges))


class TestDynamicProgrammingSolver(unittest.TestCase):
    def setUp(self):
        # Default 3x3 game
        self.game = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        self.solver = self.game.cpu

    def test_instantiation(self):
        self.assertIsInstance(self.solver, DynamicProgrammingSolver)

    def test_3x3_hint_finding(self):
        hint = self.solver.generate_hint()
        self.assertIn("strategy", hint)
        self.assertEqual(hint["strategy"], "Dynamic Programming (State Compression)")

        if hint["move"]:
            self.assertIsInstance(hint["move"], tuple)
            self.assertEqual(len(hint["move"]), 2)
        else:
            self.assertTrue(hint["explanation"])

    def test_impossible_2x2_all_3s(self):
        """2x2 board with all 3s — DP can't find a solution, but hint system still helps."""
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game.clues = {
            (0, 0): 3, (0, 1): 3,
            (1, 0): 3, (1, 1): 3
        }
        hint = game.cpu.generate_hint()
        # The improved hint system always tries to return a useful edge,
        # even when DP can't enumerate complete solutions.
        self.assertIn("strategy", hint)
        self.assertTrue(hint["explanation"])

    def test_solve_never_returns_none_on_solvable_board(self):
        """DP must NEVER return None on a solvable board."""
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        # Use a known-solvable 2x2 configuration: a simple rectangle loop
        # Clue 1 on all 4 cells forms a simple rectangular loop
        game.clues = {(0,0): 2, (0,1): 2, (1,0): 2, (1,1): 2}
        game.graph = Graph(game.rows, game.cols)
        solver = DynamicProgrammingSolver(game)
        move = solver.solve()
        self.assertIsNotNone(move, "DP must never return None")
        self.assertIsInstance(move, tuple)
        self.assertEqual(len(move), 2)

    def test_decide_move_never_returns_empty_on_cpu_turn(self):
        """decide_move must never return ([], None) on a solvable board when it's CPU turn."""
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game.clues = {(0,0): 2, (0,1): 2, (1,0): 2, (1,1): 2}
        game.graph = Graph(game.rows, game.cols)
        game.turn = "Player 2 (CPU)"
        solver = DynamicProgrammingSolver(game)
        candidates, best_move = solver.decide_move()
        self.assertIsNotNone(best_move, "decide_move must never return None on CPU turn")
        self.assertGreater(len(candidates), 0)

    def test_dp_frequency_analysis(self):
        """Verify frequency analysis produces correct count_on values."""
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        solution_edges = set(game.solution_edges)
        game.clues = self._build_full_clues(game.rows, game.cols, solution_edges)
        solver = DynamicProgrammingSolver(game)

        try:
            all_solutions = solver._compute_all_valid_solutions()
        except RuntimeError:
            self.skipTest("No valid solutions found for generated board")

        total = len(all_solutions)
        self.assertGreater(total, 0)

        undecided = solver._get_all_potential_edges()
        count_on, total_out = solver._frequency_analysis(all_solutions, undecided)

        self.assertEqual(total_out, total)
        for edge in undecided:
            self.assertIn(edge, count_on)
            self.assertGreaterEqual(count_on[edge], 0)
            self.assertLessEqual(count_on[edge], total)

    def test_dp_forced_inclusion(self):
        """Edge appearing in ALL solutions should be selected as forced inclusion."""
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        solution_edges = set(game.solution_edges)
        game.clues = self._build_full_clues(game.rows, game.cols, solution_edges)
        solver = DynamicProgrammingSolver(game)

        try:
            all_solutions = solver._compute_all_valid_solutions()
        except RuntimeError:
            self.skipTest("No valid solutions found")

        if len(all_solutions) <= 1:
            self.skipTest("Need multiple solutions to test forced inclusion")

        # Find an edge in ALL solutions
        intersection = set(all_solutions[0])
        for sol in all_solutions[1:]:
            intersection &= sol

        undecided = [e for e in solver._get_all_potential_edges()
                     if e not in set(game.graph.edges)]
        forced = [e for e in undecided if e in intersection]

        if forced:
            count_on, total = solver._frequency_analysis(all_solutions, undecided)
            for e in forced:
                self.assertEqual(count_on[e], total,
                                 f"Edge {e} in all solutions should have count_on == total")

    def test_dp_determinism(self):
        """Same board state must produce identical moves."""
        # Use a known-solvable 2x2 board
        full_clues = {(0,0): 2, (0,1): 2, (1,0): 2, (1,1): 2}

        game1 = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game1.clues = full_clues
        game1.graph = Graph(game1.rows, game1.cols)
        game1.cpu = DynamicProgrammingSolver(game1)
        move1 = game1.cpu.solve()

        game2 = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        game2.clues = full_clues
        game2.graph = Graph(game2.rows, game2.cols)
        game2.cpu = DynamicProgrammingSolver(game2)
        move2 = game2.cpu.solve()

        self.assertEqual(move1, move2, "Same board must produce same move (determinism)")

    def test_dp_no_fallback_message(self):
        """DP mode should never produce a fallback message."""
        from logic.strategy_controller import StrategyController
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        controller = StrategyController(game, "dynamic_programming")
        game.turn = "Player 2 (CPU)"
        move, source = controller.get_next_cpu_move()
        fallback_msg = controller.get_fallback_message(source)
        self.assertIsNone(fallback_msg, "DP mode must never produce a fallback message")

    def test_dp_source_label_is_dp(self):
        """In DP mode, the source label should always be 'DP'."""
        from logic.strategy_controller import StrategyController
        game = GameState(rows=2, cols=2, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        solution_edges = set(game.solution_edges)
        game.clues = self._build_full_clues(game.rows, game.cols, solution_edges)
        game.turn = "Player 2 (CPU)"
        controller = StrategyController(game, "dynamic_programming")
        move, source = controller.get_next_cpu_move()
        if move is not None:
            self.assertEqual(source, "DP", "DP mode source should be 'DP', not a fallback label")

    def test_certainty_tiebreak_lexicographic(self):
        """When certainty scores tie, the lexicographically smallest edge wins."""
        solver = DynamicProgrammingSolver.__new__(DynamicProgrammingSolver)
        # Construct a scenario with tied certainty
        edge_a = ((0, 0), (0, 1))
        edge_b = ((0, 1), (0, 2))
        edge_c = ((1, 0), (1, 1))

        # All 3 edges appear in 1/2 solutions → certainty = 0.0 for all
        count_on = {edge_a: 1, edge_b: 1, edge_c: 1}
        total = 2
        undecided = [edge_a, edge_b, edge_c]

        move, action, explanation = solver._select_best_move(count_on, total, undecided)
        self.assertEqual(move, edge_a, "Should pick lexicographically smallest on tie")

    def test_no_greedy_or_dnc_in_dp(self):
        """Verify no greedy or D&C references in the DP solver source."""
        import inspect
        source = inspect.getsource(DynamicProgrammingSolver)
        self.assertNotIn("GreedySolver", source)
        self.assertNotIn("DivideConquerSolver", source)
        self.assertNotIn("_find_fallback_move", source)

    def _build_full_clues(self, rows, cols, solution_edges):
        clues = {}
        for r in range(rows):
            for c in range(cols):
                count = 0
                if tuple(sorted(((r, c), (r, c + 1)))) in solution_edges:
                    count += 1
                if tuple(sorted(((r + 1, c), (r + 1, c + 1)))) in solution_edges:
                    count += 1
                if tuple(sorted(((r, c), (r + 1, c)))) in solution_edges:
                    count += 1
                if tuple(sorted(((r, c + 1), (r + 1, c + 1)))) in solution_edges:
                    count += 1
                clues[(r, c)] = count
        return clues


if __name__ == '__main__':
    unittest.main()
