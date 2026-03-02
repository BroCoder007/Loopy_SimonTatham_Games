
import unittest
from unittest.mock import MagicMock
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from logic.solvers.super_solver import ParallelDnCSolver

class MockGraph:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.edges = set()
    
    def get_clue(self, r, c):
        return None
    
    def get_neighbors(self, r, c):
        return []
        
    def get_edge_status(self, edge_index):
        return -1 # Unknown
        
    def set_edge_status(self, edge_index, status):
        return False
        
    def is_valid(self):
        return True

class TestParallelDnCSolver(unittest.TestCase):
    def test_initialization(self):
        solver = ParallelDnCSolver(max_workers=2)
        self.assertIsNotNone(solver.executor)
        self.assertEqual(len(solver.separator_memo), 0)
        
    def test_solve_small_grid_base_case(self):
        # 4x4 grid -> 16 cells <= 36, should hit base case
        graph = MockGraph(4, 4)
        solver = ParallelDnCSolver()
        
        # Mock the base solver's solve_small_grid to return a dummy result
        solver.base_solver.solve_small_grid = MagicMock(return_value=[1, 2, 3])
        
        result = solver.solve(graph)
        
        # Should call base case
        # Note: solve calls _solve_recursive which calls base_solver.solve_small_grid
        solver.base_solver.solve_small_grid.assert_called()
        self.assertEqual(result, [1, 2, 3])

    def test_solve_large_grid_split(self):
        # 10x10 grid -> 100 cells > 36, should split
        graph = MockGraph(10, 10)
        solver = ParallelDnCSolver(max_workers=2)
        
        # Mock base solver to avoid actual computation
        solver.base_solver.solve_small_grid = MagicMock(return_value=[])
        
        # Track splits by mocking on_split
        solver.on_split = MagicMock()
        
        result = solver.solve(graph)
        
        # Should have split at least once
        # logic: 10x10 -> split -> 5x10 or 10x5 -> ... -> small enough
        self.assertTrue(solver.on_split.called)
        self.assertIsNotNone(result)

if __name__ == '__main__':
    unittest.main()
