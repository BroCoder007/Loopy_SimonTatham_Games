"""
Divide & Conquer Puzzle Generator
=================================
Generates Slitherlink puzzles using a structural decomposition approach.

Algorithm:
1. Spatial Decomposition: Recursively divide grid into quadrants.
2. Base Case: Generate small valid loop segments in 2x2 or 3x3 regions.
3. Merge: Stitch regions together by connecting boundary edges to form a single global loop.
4. Validation: Use AdvancedDPSolver to ensure uniqueness and validity.
"""

import random
from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict
from logic.graph import Graph
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.validators import is_valid_move, check_win_condition

class DivideAndConquerPuzzleGenerator:
    def __init__(self, rows: int, cols: int, difficulty: str = "Medium"):
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        self.graph = Graph(rows, cols)
        self.solution_edges: Set[Tuple[Tuple[int, int], Tuple[int, int]]] = set()
        self.clues: Dict[Tuple[int, int], int] = {}
        
        # Difficulty parameters
        self.complexity_target = {
            "Easy": 0.4,   # Simple loops
            "Medium": 0.6, # Moderate winding
            "Hard": 0.8    # Complex, dense loops
        }.get(difficulty, 0.5)

    def generate(self) -> Tuple[Dict[Tuple[int, int], int], Set]:
        """
        Main entry point. Returns (clues, solution_edges).
        """
        # 1. Generate Loop Structure
        success = False
        attempts = 0
        max_attempts = 10
        
        while not success and attempts < max_attempts:
            attempts += 1
            print(f"[Generator] Attempt {attempts} of {max_attempts}...")
            
            # Clear previous state
            self.solution_edges.clear()
            self.clues.clear()
            
            # Recursive Generation
            edges = self._generate_region(0, self.rows, 0, self.cols)
            
            # Global Loop Check
            if self._validate_global_loop(edges):
                self.solution_edges = edges
                
                # 2. Assign Numbers
                self._assign_clues()
                
                # 3. Solver Validation (Uniqueness)
                # We need a temporary GameState-like object or mock for the solver
                if self._verify_uniqueness():
                    success = True
                    print("[Generator] Puzzle Generation Successful!")
                else:
                    print("[Generator] Puzzle not unique. Retrying...")
            else:
                 print("[Generator] Invalid loop structure (disconnected or crossing). Retrying...")

        if not success:
            print("[Generator] DBG: Failed to generate valid puzzle after max attempts. Backup to simple loop.")
            self._generate_backup_simple()

        return self.clues, self.solution_edges

    def _generate_region(self, r_min, r_max, c_min, c_max, depth=0) -> Set[Tuple]:
        """
        Recursive function to generate loop segments for a region.
        """
        rows = r_max - r_min
        cols = c_max - c_min
        
        from logic.execution_trace import log_generator_step
        log_generator_step(
            explanation=f"Generating Region {r_min}-{r_max}, {c_min}-{c_max}",
            recursion_depth=depth,
            region_id=f"{r_min}-{r_max},{c_min}-{c_max}"
        )
        
        # Base Case: Small enough to generate directly
        if rows <= 3 and cols <= 3:
            return self._generate_base_case(r_min, r_max, c_min, c_max)
            
        # Recursive Step: Divide
        # Prefer splitting longer dimension
        split_r = (r_min + r_max) // 2
        split_c = (c_min + c_max) // 2
        
        # 4 Quadrants
        # 4 Quadrants
        # Top-Left
        q1 = self._generate_region(r_min, split_r, c_min, split_c, depth + 1)
        # Top-Right
        q2 = self._generate_region(r_min, split_r, split_c, c_max, depth + 1)
        # Bottom-Left
        q3 = self._generate_region(split_r, r_max, c_min, split_c, depth + 1)
        # Bottom-Right
        q4 = self._generate_region(split_r, r_max, split_c, c_max, depth + 1)
        
        # Merge Horizontal (Top halves)
        log_generator_step(
            explanation="Merging Top Halves",
            recursion_depth=depth,
            region_id="TopMerge",
            merge_info=f"Horizontal Merge at col {split_c}"
        )
        top = self._merge_horizontal(q1, q2, r_min, split_r, split_c)
        # Merge Horizontal (Bottom halves)
        log_generator_step(
            explanation="Merging Bottom Halves",
            recursion_depth=depth,
            region_id="BottomMerge",
            merge_info=f"Horizontal Merge at col {split_c}"
        )
        bottom = self._merge_horizontal(q3, q4, split_r, r_max, split_c)
        
        # Merge Vertical (Top + Bottom)
        log_generator_step(
            explanation="Merging Top and Bottom",
            recursion_depth=depth,
            region_id="GlobalMerge",
            merge_info=f"Vertical Merge at row {split_r}"
        )
        combined = self._merge_vertical(top, bottom, split_r, c_min, c_max)
        
        return combined

    def _generate_base_case(self, r_min, r_max, c_min, c_max) -> Set[Tuple]:
        """
        Generates a random valid loop/path segment within a small box.
        For simplicity in D&C, we can just return a simple rectangle or 
        randomly bite out a corner, ensuring local validity (degree 0 or 2).
        """
        edges = set()
        
        # Simple Logic: Random Rectangle within the bounds
        # Ensure it's not touching the very edge of the region unless intend to connect?
        # Actually, for merge to work, we need some edges to touch the boundary 
        # so they can be connected to neighbors.
        
        # Let's generate a full rectangle for the region minus 1 margin?
        # Or just a simple cycle.
        
        # Valid tiny loops:
        # 1. Full 1x1 cell loop
        # 2. 2x2 loop
        # 3. L-shape
        
        # We need something that has a chance to connect.
        # Strategy: RandomWalk within region?
        # Better: Random Rectangle.
        
        h = r_max - r_min
        w = c_max - c_min
        
        if h < 1 or w < 1: return set()
        
        # Randomize start/end of rectangle
        r0 = r_min + random.randint(0, max(0, h-2))
        c0 = c_min + random.randint(0, max(0, w-2))
        
        r1 = r0 + random.randint(1, h - (r0 - r_min))
        c1 = c0 + random.randint(1, w - (c0 - c_min))
        
        # Add edges for box (r0, c0) to (r1, c1) vertices
        # Top
        for c in range(c0, c1):
            edges.add(tuple(sorted(((r0, c), (r0, c+1)))))
        # Bottom
        for c in range(c0, c1):
            edges.add(tuple(sorted(((r1, c), (r1, c+1)))))
        # Left
        for r in range(r0, r1):
            edges.add(tuple(sorted(((r, c0), (r+1, c0)))))
        # Right
        for r in range(r0, r1):
            edges.add(tuple(sorted(((r, c1), (r+1, c1)))))
            
        return edges

    def _merge_horizontal(self, left_edges: Set, right_edges: Set, r_min, r_max, split_c) -> Set:
        """
        Merges two regions along a vertical line at split_c.
        """
        combined = left_edges.union(right_edges)
        
        # Identify "Bridge Candidates"
        # Edges on the boundary line split_c
        # Left region touches split_c from left: ((r, split_c-1), (r+1, split_c-1)) ?? No.
        # Boundary is the vertical line at column split_c.
        # Edges ON boundary: ((r, split_c), (r+1, split_c))
        
        # We want to open the boundary to connect loops.
        # If we have a vertical edge ON the boundary in BOTH sets? 
        # Impossible, they are disjoint sets of cells.
        # But they share the vertical grid line split_c.
        
        # Left region has vertical edges at col split_c?
        # Yes, its Right boundary is split_c.
        # Right region has vertical edges at col split_c?
        # Yes, its Left boundary is split_c.
        
        # If both regions have a vertical edge at the same position (r, split_c)-(r+1, split_c),
        # It means they are touching.
        # We can REMOVE this shared edge to merge the two cells?
        # No, a vertical edge *separates* cells. 
        # If both have it, it's a double line? No, it's the same edge.
        
        # Wait, if I generate a loop in Left Q, it might use the rightmost boundary edges.
        # If I generate a loop in Right Q, it might use the leftmost boundary edges.
        # Since they are the SAME edges in the grid graph, we check for overlaps.
        
        # Overlapping edges = Potential Merge Points.
        # If valid loop in Left has edge E, and valid loop in Right has edge E.
        # If we keep E, we have two separate loops touching.
        # If we REMOVE E, we merge the loops! (   [][]  ->  [  ]  )
        
        # So strategy: Find common edges on the boundary. Remove them.
        # But we must ensure correct degree constraints.
        
        # 1. Find overlapping vertical edges on the split line
        # Edges are sorted tuples.
        intersection = left_edges.intersection(right_edges)
        
        # 2. Filter for only boundary edges
        merge_candidates = []
        for u, v in intersection:
             # Check if vertical and on split_c
             if u[1] == split_c and v[1] == split_c: # Vertical column check
                 merge_candidates.append((u, v))
                 
        # 3. Randomly select some to remove (Merge)
        # Removing a common edge merges the faces.
        # Ideally we remove enough to make it one component locally?
        
        to_remove = set()
        for edge in merge_candidates:
            if random.random() < self.complexity_target:
                to_remove.add(edge)
                
        # 4. Result
        final_edges = combined - intersection # Remove shared ones initially (merging loops)
        # Wait. If we remove it, we merge.
        # If we KEEP it, it acts as a wall between them.
        # But "combined" is Union. "intersection" is the shared ones.
        # If we subtract intersection, we remove ALL shared edges.
        # This implies we ALWAYS merge where they touch.
        # Is that safe?
        # If we remove edge E, vertices u and v lose 2 degrees (1 from left loop, 1 from right loop).
        # Total degree at u was 2 (Left) + 2 (Right) = 4?
        # No.
        # In Slitherlink, a vertex has degree 0 or 2.
        # If Left Loop passes through u, degree is 2.
        # If Right Loop passes through u, degree is 2.
        # If they share edge u-v:
        #   u has edge (u, v) and another edge in Left.
        #   u has edge (u, v) and another edge in Right.
        #   Total edges at u from overlapping perspective: 
        #     Left:  A-u-v
        #     Right: B-u-v
        #     Union: A-u, B-u, u-v (shared).
        #     Degree at u is 3? No.
        #     Wait.
        #     Left Loop:  path ... -> A -> u -> v -> ...
        #     Right Loop: path ... -> B -> u -> v -> ...
        #     If we superimpose: u has edges (u,A), (u,B), (u,v). Degree 3. Invalid.
        #     We need degree 2.
        #     So if we merge, we MUST remove (u,v).
        #     Then u connects to A and B. Degree 2. Valid!
        
        # CONCLUSION:
        # If two valid loops share an edge, we MUST remove that edge to maintain degree 2 at vertices.
        # (Assuming no other lines meet there, which is true for checking just 2 regions).
        
        # So: ALWAYS remove overlapping edges.
        final_edges = combined - intersection
        
        return final_edges

    def _merge_vertical(self, top_edges: Set, bottom_edges: Set, split_r, c_min, c_max) -> Set:
        """
        Merges two regions along a horizontal line at split_r.
        """
        combined = top_edges.union(bottom_edges)
        intersection = top_edges.intersection(bottom_edges)
        
        # Overlapping edges on horizontal split_r line
        # ((split_r, c), (split_r, c+1))
        
        # Remove all shared edges to merge loops and maintain valid degrees.
        final_edges = combined - intersection
        
        return final_edges

    def _validate_global_loop(self, edges: Set[Tuple]) -> bool:
        """
        Checks if the edges form exactly ONE single closed loop with no branches.
        """
        if not edges: return False
        
        # 1. Check Degrees (Must be 0 or 2)
        degree = defaultdict(int)
        nodes = set()
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            nodes.add(u)
            nodes.add(v)
            
        for n in nodes:
            if degree[n] != 2:
                return False # Crossing or loose end
                
        # 2. Check Connectivity (Single Component)
        # BFS/UnionFind
        if not nodes: return True # Empty is valid? No, we needed a puzzle.
        
        start_node = next(iter(nodes))
        visited = {start_node}
        queue = [start_node]
        
        # Build adjacency for BFS
        adj = defaultdict(list)
        for u, v in edges:
            adj[u].append(v)
            adj[v].append(u)
            
        while queue:
            curr = queue.pop(0)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
                    
        return len(visited) == len(nodes)

    def _assign_clues(self):
        """
        Calculate numbers (0-3) based on solution_edges.
        """
        for r in range(self.rows):
            for c in range(self.cols):
                # Count surrounding edges
                count = 0
                
                # Top: (r,c)-(r,c+1)
                if tuple(sorted(((r, c), (r, c+1)))) in self.solution_edges: count += 1
                # Bottom: (r+1,c)-(r+1,c+1)
                if tuple(sorted(((r+1, c), (r+1, c+1)))) in self.solution_edges: count += 1
                # Left: (r,c)-(r+1,c)
                if tuple(sorted(((r, c), (r+1, c)))) in self.solution_edges: count += 1
                # Right: (r,c+1)-(r+1,c+1)
                if tuple(sorted(((r, c+1), (r+1, c+1)))) in self.solution_edges: count += 1
                
                # Add clue with some probability (difficulty dependent)
                # Easy: More clues. Hard: Fewer clues.
                # But typically Slitherlink patterns are about strict logic.
                # Let's keep most clues for now to ensure unique solvability,
                # then maybe sparse them out?
                # User req: "Assign number 0-3".
                
                self.clues[(r, c)] = count
        
        # Post-process: Sparsify clues?
        # If we leave all clues, it's trivial? No, still need to deduce.
        # But to be verified by Solver, let's keep all for now (Maximum constraint),
        # or remove 0s/3s randomly?
        
        # Let's verify with ALL clues first. If unique, try removing some?
        # This is expensive.
        # Just return full clues for Version 1.

    def _verify_uniqueness(self) -> bool:
        """
        Uses DynamicProgrammingSolver to check uniqueness.
        """
        # Create a mock GameState with just graph and clues
        class MockGameState:
            def __init__(self, r, c, clues):
                self.rows = r
                self.cols = c
                self.clues = clues
                self.graph = Graph(r, c) # Empty graph
                self.game_mode = "vs_cpu" # Required for some solver checks?
                self.turn = "Player 1"

        mock_gs = MockGameState(self.rows, self.cols, self.clues)
        
        # Instantiate Solver
        solver = DynamicProgrammingSolver(mock_gs)
        
        # Check for exactly 1 solution
        # using internal _run_dp with limit=2 to detect ambiguity
        solutions = solver._run_dp(mock_gs, limit=2)
        
        if len(solutions) == 1:
            return True
        elif len(solutions) == 0:
            print("[Generator] Verification Failed: No solution found for these clues.")
        else:
            print("[Generator] Verification Failed: Multiple solutions found (Ambiguous).")
            
        return False

    def _generate_backup_simple(self):
        """
        Backup: Simple huge rectangle if D&C fails too many times.
        """
        edges = set()
        r_min, r_max = 0, self.rows
        c_min, c_max = 0, self.cols
        
        # Simple Logic: Max rectangle with margin 1
        r0 = 0
        c0 = 0
        r1 = r_max 
        c1 = c_max 
        
        for c in range(c0, c1):
            edges.add(tuple(sorted(((r0, c), (r0, c+1)))))
        for c in range(c0, c1):
            edges.add(tuple(sorted(((r1, c), (r1, c+1)))))
        for r in range(r0, r1):
            edges.add(tuple(sorted(((r, c0), (r+1, c0)))))
        for r in range(r0, r1):
            edges.add(tuple(sorted(((r, c1), (r+1, c1)))))
            
        self.solution_edges = edges
        self._assign_clues()
