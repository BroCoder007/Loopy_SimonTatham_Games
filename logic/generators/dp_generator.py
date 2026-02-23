import random

class DPGenerator:
    """
    Phase 3: DP & Backtracking Generator
    Uses Backtracking to generate algorithmic board shapes (finding a loop via DFS backtracking).
    Uses Dynamic Programming (DP) state caching to calculate the difficulty score of the board
    by measuring local path ambiguities.
    """
    def __init__(self, rows, cols, difficulty="Medium"):
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        self.edges = set()
        self.clues = {}

    def generate(self):
        # 1. Backtracking Generator: Generate a valid continuous loop
        self.edges = self._generate_loop_backtracking()
        
        # Fallback if backtracking completely fails on tight constraints
        if not self.edges:
            self.edges = self._fallback_simple_loop()

        # 2. Derive Clues from the loop
        full_clues = self._derive_clues(self.edges)
        
        # 3. DP Difficulty Analytics: drop clues until we reach a target DP score
        self.clues = self._apply_dp_difficulty(full_clues)
        
        return self.clues, self.edges

    def _generate_loop_backtracking(self):
        """
        Uses standard Backtracking to find a Hamiltonian-like cycle 
        (a single continuous loop) on a subset of the grid vertices.
        """
        # Graph of vertices
        vertices = [(r, c) for r in range(self.rows + 1) for c in range(self.cols + 1)]
        
        # Target size scale for loop
        target_size = max(4, int((self.rows * self.cols) * 0.6)) 
        
        visited_vertices = set()
        start_v = (random.randint(1, self.rows-1), random.randint(1, self.cols-1))
        visited_vertices.add(start_v)
        
        path = [start_v]
        
        def get_neighbors(v):
            r, c = v
            ns = []
            if r > 0: ns.append((r-1, c))
            if r < self.rows: ns.append((r+1, c))
            if c > 0: ns.append((r, c-1))
            if c < self.cols: ns.append((r, c+1))
            random.shuffle(ns)
            return ns

        def backtrack(current_v):
            if len(path) >= target_size:
                # Try to close the loop
                if start_v in get_neighbors(current_v):
                    if len(path) > 4: # Must be >4 to be a valid puzzle loop
                        path.append(start_v)
                        return True
            
            for neighbor in get_neighbors(current_v):
                if neighbor not in visited_vertices:
                    visited_vertices.add(neighbor)
                    path.append(neighbor)
                    
                    if backtrack(neighbor):
                        return True
                        
                    # Backtrack (Undo step)
                    path.pop()
                    visited_vertices.remove(neighbor)
                    
            return False
            
        success = backtrack(start_v)
        if not success:
            return set()
            
        # Convert vertex path to edge tuples
        loop_edges = set()
        for i in range(len(path) - 1):
            u = path[i]
            v = path[i+1]
            loop_edges.add(tuple(sorted((u, v))))
            
        return loop_edges

    def _fallback_simple_loop(self):
        """In case constraints make backtracking fail, provide a simple rectangle"""
        edges = set()
        r1, c1 = 1, 1
        r2, c2 = self.rows - 1, self.cols - 1
        for c in range(c1, c2):
            edges.add(tuple(sorted(((r1, c), (r1, c+1)))))
            edges.add(tuple(sorted(((r2, c), (r2, c+1)))))
        for r in range(r1, r2):
            edges.add(tuple(sorted(((r, c1), (r+1, c1)))))
            edges.add(tuple(sorted(((r, c2), (r+1, c2)))))
        return edges

    def _derive_clues(self, loop_edges):
        clues = {}
        for r in range(self.rows):
            for c in range(self.cols):
                count = 0
                if tuple(sorted(((r, c), (r, c+1)))) in loop_edges: count += 1
                if tuple(sorted(((r+1, c), (r+1, c+1)))) in loop_edges: count += 1
                if tuple(sorted(((r, c), (r+1, c)))) in loop_edges: count += 1
                if tuple(sorted(((r, c+1), (r+1, c+1)))) in loop_edges: count += 1
                clues[(r, c)] = count
        return clues

    def _evaluate_dp_complexity(self, current_clues):
        """
        DP Analytics: Counts 'ambiguous' local states. 
        Memoizes cell evaluations to quickly calculate a difficulty score 
        based on how many combinations of empty/filled lines are theoretically possible.
        """
        dp_cache = {}
        score = 0
        
        def count_combinations(r, c):
            if (r, c) in dp_cache:
                return dp_cache[(r, c)]
                
            clue = current_clues.get((r, c), None)
            if clue is None:
                combos = 16 # 2^4 possible combinations for 4 edges if no clue
            elif clue == 0 or clue == 4:
                combos = 1
            elif clue == 1 or clue == 3:
                combos = 4
            elif clue == 2:
                combos = 6
                
            # Add complexity multiplier if neighbors are also ambiguous
            local_multiplier = 1
            if r > 0 and (r-1, c) not in current_clues: local_multiplier += 1
            if r < self.rows-1 and (r+1, c) not in current_clues: local_multiplier += 1
            if c > 0 and (r, c-1) not in current_clues: local_multiplier += 1
            if c < self.cols-1 and (r, c+1) not in current_clues: local_multiplier += 1
            
            final_score = combos * local_multiplier
            dp_cache[(r, c)] = final_score
            return final_score

        for r in range(self.rows):
            for c in range(self.cols):
                score += count_combinations(r, c)
                
        return score

    def _apply_dp_difficulty(self, full_clues):
        """
        Uses the DP complexity evaluator to reach a target difficulty tier.
        Removes clues to monotonically increase the DP ambiguity score.
        """
        cells = list(full_clues.keys())
        random.shuffle(cells)
        
        base_score = self._evaluate_dp_complexity(full_clues)
        
        target_score = {
            "Easy": base_score * 2.5,   
            "Medium": base_score * 4.0, 
            "Hard": base_score * 6.5    
        }.get(self.difficulty, base_score * 4.0)

        current_clues = full_clues.copy()
        for cell in cells:
            val = current_clues.pop(cell)
            score = self._evaluate_dp_complexity(current_clues)
            if score > target_score:
                # Revert if it exceeds the difficulty rating
                current_clues[cell] = val
                break
                
        return current_clues
