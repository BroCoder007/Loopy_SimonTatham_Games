import unittest
import random
from logic.solvers.solver_utils import ZobristHasher, ConstraintPropagator

class MockGrid:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.clues = {}  # (r, c) -> int
        self.edges = {}  # edge_index (int) -> status (0, 1, -1)
        self.neighbors = {} # (r, c) -> list of edge_indices
        
        # Initialize all edges to -1 (UNKNOWN)
        # For simplicity, let's say edge indices are just 0, 1, 2...
        self.num_edges = 0
        for r in range(rows):
            for c in range(cols):
                # Assign 4 edges to each cell for testing
                # In a real grid, edges are shared. Here we simulate shared edges manually if needed 
                # or just treat them as unique for rule testing.
                # To test propagation properly, we need SHARED edges.
                pass
        
        # simplified edge model: 
        # vertical edges: (r, c, 'v')
        # horizontal edges: (r, c, 'h')
        # Map these to integers 0..N
        self.edge_map = {}
        self.edge_list = []
        
        count = 0
        # Horizontal
        for r in range(rows + 1):
            for c in range(cols):
                tag = (r, c, 'h')
                self.edge_map[tag] = count
                self.edge_list.append(tag)
                self.edges[count] = -1
                count += 1
        # Vertical
        for r in range(rows):
            for c in range(cols + 1):
                tag = (r, c, 'v')
                self.edge_map[tag] = count
                self.edge_list.append(tag)
                self.edges[count] = -1
                count += 1
                
        self.num_total_edges = count

    def get_clue(self, r, c):
        return self.clues.get((r, c))

    def get_neighbors(self, r, c):
        # Return indices of 4 surrounding edges
        # Top: (r, c, 'h'), Bottom: (r+1, c, 'h')
        # Left: (r, c, 'v'), Right: (r, c+1, 'v')
        candidates = [
            (r, c, 'h'), (r+1, c, 'h'),
            (r, c, 'v'), (r, c+1, 'v')
        ]
        indices = []
        for tag in candidates:
            if tag in self.edge_map:
                indices.append(self.edge_map[tag])
        return indices

    def get_edge_status(self, edge_index):
        return self.edges.get(edge_index, -1)

    def set_edge_status(self, edge_index, status):
        if self.edges[edge_index] == status:
            return False
        self.edges[edge_index] = status
        return True

    def is_valid(self):
        # Simplified validity check
        # A real check would look for loops or crossing lines logic
        # For our unit test, we assume valid unless we manually break it
        return True

class TestSolverUtils(unittest.TestCase):
    def test_zobrist_hasher(self):
        num_edges = 100
        hasher = ZobristHasher(num_edges)
        
        # Test Determinism
        active_edges = [1, 5, 10, 50]
        hash1 = hasher.compute_hash(active_edges)
        hash2 = hasher.compute_hash(active_edges)
        self.assertEqual(hash1, hash2, "Hash should be deterministic")
        
        # Test Sensitivity
        active_edges_2 = [1, 5, 10, 51] # One bit different
        hash3 = hasher.compute_hash(active_edges_2)
        self.assertNotEqual(hash1, hash3, "Different states should hash differently")
        
        # Test Empty
        hash_empty = hasher.compute_hash([])
        self.assertEqual(hash_empty, 0, "Empty set should hash to 0")

    def test_constraint_propagation_rule_1(self):
        # Rule 1: Clue 0 -> All neighbors OFF
        grid = MockGrid(3, 3)
        grid.clues[(1, 1)] = 0 # Center cell has 0
        
        valid, changes = ConstraintPropagator.propagate(grid)
        
        self.assertTrue(valid)
        
        # Check neighbors of (1, 1)
        neighbors = grid.get_neighbors(1, 1)
        for idx in neighbors:
            self.assertEqual(grid.get_edge_status(idx), 0, f"Edge {idx} should be OFF (0)")
            self.assertIn(idx, changes)

    def test_constraint_propagation_rule_3(self):
        # Rule 3: Clue 4 -> All neighbors ON (Must Fill)
        grid = MockGrid(3, 3)
        grid.clues[(1, 1)] = 4
        
        valid, changes = ConstraintPropagator.propagate(grid)
        self.assertTrue(valid)
        
        neighbors = grid.get_neighbors(1, 1)
        for idx in neighbors:
            self.assertEqual(grid.get_edge_status(idx), 1, f"Edge {idx} should be ON (1)")

    def test_constraint_propagation_contradiction(self):
        # Contradiction: Clue 4 but one edge is already OFF
        grid = MockGrid(3, 3)
        grid.clues[(1, 1)] = 4
        
        # Manually set one edge to OFF
        neighbors = grid.get_neighbors(1, 1)
        grid.set_edge_status(neighbors[0], 0) 
        
        valid, changes = ConstraintPropagator.propagate(grid)
        self.assertFalse(valid, "Should detect contradiction")

if __name__ == '__main__':
    unittest.main()
