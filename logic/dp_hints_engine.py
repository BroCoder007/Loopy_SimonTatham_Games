from collections import deque
from logic.validators import is_valid_move

class DPHintsEngine:
    """
    Phase 3: DP & Backtracking Hint Engine
    1. Backtracking Validator: Checks if current lines have mathematically doomed the board.
    2. DP Shortest Path: Finds the optimal path connecting two loose endpoints without guessing.
    """
    def __init__(self, game_state):
        self.game_state = game_state
        self.graph = game_state.graph
        self.clues = game_state.clues

    def generate_hint(self):
        # 1. Backtracking Validation (Fast shallow check)
        if not self._backtrack_validate_current_state():
            return {
                "move": None, 
                "strategy": "Backtrack Validation",
                "explanation": "Fatal Error! Current lines violate the puzzle rules. Undo recent moves!",
                "is_error": True
            }

        loose_ends = self._find_loose_ends()
        
        # 2. DP Shortest Path between endpoints
        if len(loose_ends) >= 2:
            # Try to connect the first loose end to any other loose end
            start = loose_ends[0]
            for end in loose_ends[1:]:
                path_edges = self._dp_shortest_path(start, end)
                if path_edges:
                    # Give the first step of this DP shortest path
                    for edge in path_edges:
                        if edge not in self.graph.edges:
                            return {
                                "move": edge,
                                "strategy": "DP Shortest Path",
                                "explanation": f"DP found the optimal shortest path connecting segments.",
                                "is_error": False,
                                "highlight_path": path_edges
                            }

        # Fallback greedy safe move
        for edge in self._get_all_empty_edges():
            u, v = edge
            valid, _ = is_valid_move(self.graph, u, v, self.clues)
            if valid:
                return {
                    "move": edge,
                    "strategy": "Safe Edge (Fallback)",
                    "explanation": "No clear DP path found. Here is a generally safe isolated edge to start.",
                    "is_error": False
                }
                
        return {"move": None, "strategy": "None", "explanation": "No valid moves left!", "is_error": False}

    def _backtrack_validate_current_state(self):
        """
        Shallow Backtracking to check current validity against basic rules.
        """
        # Constraint check: Have we bypassed a node's required constraint?
        for r in range(self.game_state.rows):
            for c in range(self.game_state.cols):
                if (r, c) in self.clues:
                    count = 0
                    if tuple(sorted(((r, c), (r, c+1)))) in self.graph.edges: count += 1
                    if tuple(sorted(((r+1, c), (r+1, c+1)))) in self.graph.edges: count += 1
                    if tuple(sorted(((r, c), (r+1, c)))) in self.graph.edges: count += 1
                    if tuple(sorted(((r, c+1), (r+1, c+1)))) in self.graph.edges: count += 1
                    
                    if count > self.clues[(r, c)]:
                        return False # Overloaded
                        
        # Branching check: Any vertex with degree > 2 is an instant failure
        degrees = {}
        for edge in self.graph.edges:
            u, v = edge
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
            if degrees[u] > 2 or degrees[v] > 2:
                return False 
                
        return True

    def _find_loose_ends(self):
        degrees = {}
        for edge in self.graph.edges:
            u, v = edge
            degrees[u] = degrees.get(u, 0) + 1
            degrees[v] = degrees.get(v, 0) + 1
        return [v for v, degree in degrees.items() if degree == 1]

    def _get_all_empty_edges(self):
        edges = []
        for r in range(self.game_state.rows + 1):
            for c in range(self.game_state.cols):
                edges.append(tuple(sorted(((r, c), (r, c+1)))))
        for r in range(self.game_state.rows):
            for c in range(self.game_state.cols + 1):
                edges.append(tuple(sorted(((r, c), (r+1, c)))))
        return [e for e in edges if e not in self.graph.edges]

    def _dp_shortest_path(self, start, end):
        """
        Calculates the DP Table (shortest paths) using BFS to memoize the minimum
        steps required to bridge empty grid edges between Start and End constraints.
        """
        distances = {start: 0}
        previous = {}
        queue = deque([start])
        visited = {start}
        
        while queue:
            current = queue.popleft()
            if current == end:
                break
                
            r, c = current
            neighbors = []
            if r > 0: neighbors.append((r-1, c))
            if r < self.game_state.rows: neighbors.append((r+1, c))
            if c > 0: neighbors.append((r, c-1))
            if c < self.game_state.cols: neighbors.append((r, c+1))
            
            for neighbor in neighbors:
                edge = tuple(sorted((current, neighbor)))
                # Traverse only empty valid edges
                if edge not in self.graph.edges:
                    valid, _ = is_valid_move(self.graph, current, neighbor, self.clues)
                    if valid and neighbor not in visited:
                        visited.add(neighbor)
                        distances[neighbor] = distances[current] + 1
                        previous[neighbor] = current
                        queue.append(neighbor)
                        
        path_edges = []
        curr = end
        if curr in previous:
            while curr != start:
                prev = previous[curr]
                path_edges.append(tuple(sorted((prev, curr))))
                curr = prev
            return path_edges[::-1]
            
        return []
