import time

class DPBacktrackingSolver:
    """
    Phase 3: Memoized Backtracking Solver (Person 1 / Leader)
    Recursively tries to solve the board but uses a DP Hash Cache to 
    remember dead-end board states, preventing exponential time blowouts.
    """
    def __init__(self, game_state):
        self.game_state = game_state
        self.clues = game_state.clues
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.dp_cache = {} # The DP Memoization Table
        self.nodes_visited = 0
        self.all_edges = self._get_all_edges()

    def _get_all_edges(self):
        edges = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edges.append(tuple(sorted(((r, c), (r, c+1)))))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edges.append(tuple(sorted(((r, c), (r+1, c)))))
        return edges

    def solve(self, ui_callback=None, delay=0.0):
        """
        Solves the puzzle from the current board state.
        ui_callback: function passed from UI to update the screen step-by-step
        delay: float seconds to wait between steps for animation
        """
        self.dp_cache.clear()
        self.nodes_visited = 0
        
        # Work on a copy of the graph edges
        current_edges = set(self.game_state.graph.edges)
        
        start_time = time.time()
        success = self._backtrack(current_edges, 0, ui_callback, delay)
        end_time = time.time()

        return {
            "success": success,
            "edges": current_edges if success else set(),
            "nodes_visited": self.nodes_visited,
            "time_taken": end_time - start_time,
            "cache_hits": len([v for v in self.dp_cache.values() if v is False])
        }

    def _get_state_string(self, current_edges):
        """
        Serializes the current board state into a hashable string.
        """
        # Sorting ensures the exact same edges drawn in a different order are recognized as the same state.
        return str(sorted(list(current_edges)))

    def _backtrack(self, current_edges, edge_index, ui_callback, delay):
        self.nodes_visited += 1
        
        # 1. DP Memoization Check (Cache Hit)
        # O(1) return if we already evaluated this exact sub-problem mapping
        state_key = self._get_state_string(current_edges)
        if state_key in self.dp_cache:
            return self.dp_cache[state_key]

        # 2. Check Win Condition
        if self._shallow_win_check(current_edges):
            self.dp_cache[state_key] = True
            return True

        # 3. Base Case: Searched all edges
        if edge_index >= len(self.all_edges):
            self.dp_cache[state_key] = False
            return False

        edge = self.all_edges[edge_index]
        u, v = edge

        # Branch A: Try WITH the edge
        if edge not in current_edges:
            if self._is_locally_valid(current_edges, u, v):
                current_edges.add(edge)
                
                # UI Real-time playback integration
                if ui_callback:
                    ui_callback(current_edges)
                    if delay > 0:
                        time.sleep(delay)

                if self._backtrack(current_edges, edge_index + 1, ui_callback, delay):
                    self.dp_cache[state_key] = True
                    return True
                    
                # Backtrack (Revert)
                current_edges.remove(edge)
                
                if ui_callback:
                    ui_callback(current_edges)
                    if delay > 0:
                        time.sleep(delay)

        # Branch B: Try WITHOUT the edge
        if self._backtrack(current_edges, edge_index + 1, ui_callback, delay):
            self.dp_cache[state_key] = True
            return True

        # 4. DP Memoization Save (Dead End)
        # Record that navigating from this board state inevitably leads to a failure.
        self.dp_cache[state_key] = False
        return False

    def _is_locally_valid(self, edges, u, v):
        """Fast constraint and branching validation."""
        edges.add((u, v))
        degrees = {}
        for (a, b) in edges:
            degrees[a] = degrees.get(a, 0) + 1
            degrees[b] = degrees.get(b, 0) + 1
            if degrees[a] > 2 or degrees[b] > 2:
                edges.remove((u, v))
                return False
                
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.clues:
                    count = 0
                    if tuple(sorted(((r, c), (r, c+1)))) in edges: count += 1
                    if tuple(sorted(((r+1, c), (r+1, c+1)))) in edges: count += 1
                    if tuple(sorted(((r, c), (r+1, c)))) in edges: count += 1
                    if tuple(sorted(((r, c+1), (r+1, c+1)))) in edges: count += 1
                    if count > self.clues[(r, c)]:
                        edges.remove((u, v))
                        return False
        edges.remove((u, v))
        return True

    def _shallow_win_check(self, edges):
        if not edges: return False
        
        degrees = {}
        for (u, v) in edges:
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        
        for _, deg in degrees.items():
            if deg != 2:
                return False
                
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.clues:
                    count = 0
                    if tuple(sorted(((r, c), (r, c+1)))) in edges: count += 1
                    if tuple(sorted(((r+1, c), (r+1, c+1)))) in edges: count += 1
                    if tuple(sorted(((r, c), (r+1, c)))) in edges: count += 1
                    if tuple(sorted(((r, c+1), (r+1, c+1)))) in edges: count += 1
                    if count != self.clues[(r, c)]:
                        return False
        return True
