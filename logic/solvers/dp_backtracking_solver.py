import time

class DPBacktrackingSolver:
    """
    Phase 3: Highly Optimized DP & Backtracking Solver
    Features:
    - Smart Edge Selection Heuristic (Implicit Constraint Propagation)
    - Early Cycle Detection to aggressively prune dead-end loops
    - True DP State Memoization hashing on the array of edges
    """
    def __init__(self, game_state):
        self.game_state = game_state
        self.clues = game_state.clues
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.dp_cache = {} 
        self.nodes_visited = 0
        self.cache_hits = 0
        self.all_edges = self._get_all_edges()
        
        self.cell_edges = {}
        self.edge_to_cells = {e: [] for e in self.all_edges}
        for r in range(self.rows):
            for c in range(self.cols):
                if (r, c) in self.clues:
                    edges = [
                        tuple(sorted(((r, c), (r, c+1)))),
                        tuple(sorted(((r+1, c), (r+1, c+1)))),
                        tuple(sorted(((r, c), (r+1, c)))),
                        tuple(sorted(((r, c+1), (r+1, c+1))))
                    ]
                    self.cell_edges[(r, c)] = edges
                    for e in edges:
                        self.edge_to_cells[e].append((r, c))

    def _get_all_edges(self):
        edges = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edges.append(tuple(sorted(((r, c), (r, c+1)))))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edges.append(tuple(sorted(((r, c), (r+1, c)))))
        return tuple(edges)

    def solve(self, ui_callback=None, delay=0.0):
        self.dp_cache.clear()
        self.nodes_visited = 0
        self.cache_hits = 0
        
        # Core edge state mapping (0 = Unknown, 1 = Line, -1 = Cross)
        edge_states = {e: 0 for e in self.all_edges}
        for e in self.game_state.graph.edges:
            edge_states[e] = 1
            
        start_time = time.time()
        success = self._backtrack(edge_states, ui_callback, delay)
        end_time = time.time()

        final_edges = set(e for e, st in edge_states.items() if st == 1)

        return {
            "success": success,
            "edges": final_edges if success else set(self.game_state.graph.edges),
            "nodes_visited": self.nodes_visited,
            "time_taken": end_time - start_time,
            "cache_hits": self.cache_hits
        }

    def _backtrack(self, edge_states, ui_callback, delay):
        self.nodes_visited += 1

        # O(E) precalculations for immediate DP pruning constraints
        vertex_deg = {}
        vertex_unks = {}
        for edge, state in edge_states.items():
            u, v = edge
            if state == 1:
                vertex_deg[u] = vertex_deg.get(u, 0) + 1
                vertex_deg[v] = vertex_deg.get(v, 0) + 1
            elif state == 0:
                vertex_unks[u] = vertex_unks.get(u, 0) + 1
                vertex_unks[v] = vertex_unks.get(v, 0) + 1

        cell_lines = {}
        cell_unks = {}
        for cell, clue in self.clues.items():
            lines = sum(1 for e in self.cell_edges[cell] if edge_states[e] == 1)
            unks = sum(1 for e in self.cell_edges[cell] if edge_states[e] == 0)
            cell_lines[cell] = lines
            cell_unks[cell] = unks

        # 1. Fast Validity Constraint Pruning
        if not self._is_valid(edge_states, vertex_deg, vertex_unks, cell_lines, cell_unks):
            return False

        # 2. DP Memoization Check (Subproblem caching)
        state_key = tuple(edge_states[e] for e in self.all_edges)
        if state_key in self.dp_cache:
            self.cache_hits += 1
            return self.dp_cache[state_key]

        # 3. Dynamic Node Routing via Constraint Propagation 
        unknown_edge = self._get_best_unknown_edge(edge_states, vertex_deg, vertex_unks, cell_lines, cell_unks)
        
        if unknown_edge is None:
            # Base Case: Exhausted all edges
            if self._is_single_loop_and_complete(edge_states, vertex_deg, cell_lines):
                self.dp_cache[state_key] = True
                return True
            self.dp_cache[state_key] = False
            return False

        # Multi-branch Recursion
        for guess in [1, -1]:
            edge_states[unknown_edge] = guess
            
            # Conditionally decouple UI rendering for ultra-fast time benchmarks via delay controls
            if ui_callback and delay > 0:
                ui_callback(set(e for e, st in edge_states.items() if st == 1))
                time.sleep(delay)
                
            if self._backtrack(edge_states, ui_callback, delay):
                self.dp_cache[state_key] = True
                return True
                
        # Revert 
        edge_states[unknown_edge] = 0
        self.dp_cache[state_key] = False
        return False

    def _is_valid(self, edge_states, vertex_deg, vertex_unks, cell_lines, cell_unks):
        for deg in vertex_deg.values():
            if deg > 2: return False

        for v, deg in vertex_deg.items():
            if deg == 1 and vertex_unks.get(v, 0) == 0:
                return False  # Dead un-closed branch

        for cell, clue in self.clues.items():
            lines = cell_lines[cell]
            unks = cell_unks[cell]
            if lines > clue: return False
            if lines + unks < clue: return False

        # Premature Multi-Loop Cycle Detection
        active_vertices = [v for v, deg in vertex_deg.items() if deg > 0]
        if not active_vertices: return True
            
        adj = {v: [] for v in active_vertices}
        for edge, state in edge_states.items():
            if state == 1:
                u, v = edge
                adj[u].append(v)
                adj[v].append(u)
        
        visited = set()
        cycles_found = 0
        
        for v in active_vertices:
            if v not in visited:
                q = [v]
                comp_visited = {v}
                is_cycle = True
                while q:
                    curr = q.pop(0)
                    if vertex_deg.get(curr, 0) != 2:
                        is_cycle = False
                    for nxt in adj[curr]:
                        if nxt not in comp_visited:
                            comp_visited.add(nxt)
                            q.append(nxt)
                visited.update(comp_visited)
                if is_cycle:
                    cycles_found += 1
                    
        if cycles_found > 0:
            if cycles_found > 1: return False 
            if len(visited) < len(active_vertices): return False 
            for cell, clue in self.clues.items():
                if cell_lines[cell] < clue: return False 
                    
        return True

    def _get_best_unknown_edge(self, edge_states, vertex_deg, vertex_unks, cell_lines, cell_unks):
        best_edge = None
        best_score = -1

        for edge, state in edge_states.items():
            if state == 0:
                score = 0
                u, v = edge
                du = vertex_deg.get(u, 0)
                dv = vertex_deg.get(v, 0)
                uu = vertex_unks.get(u, 0)
                uv = vertex_unks.get(v, 0)
                
                # Implicit Constraint Propagation (identifying mathematically forced lines/crosses)
                if (du == 1 and uu == 1) or (dv == 1 and uv == 1):
                    score += 10000 
                elif du == 2 or dv == 2:
                    score += 10000 
                elif (du == 0 and uu == 1) or (dv == 0 and uv == 1):
                    score += 10000 
                    
                if du == 1 or dv == 1:
                    score += 100 
                
                for cell in self.edge_to_cells[edge]:
                    clue = self.clues.get(cell)
                    if clue is not None:
                        lines = cell_lines[cell]
                        unks = cell_unks[cell]
                        
                        if lines == clue:
                            score += 10000 
                        elif lines + unks == clue:
                            score += 10000 
                        score += (4 - unks) * 10

                if score > best_score:
                    best_score = score
                    best_edge = edge
                    if best_score >= 10000:
                        return best_edge
                        
        return best_edge

    def _is_single_loop_and_complete(self, edge_states, vertex_deg, cell_lines):
        for deg in vertex_deg.values():
            if deg != 2: return False
                
        for cell, clue in self.clues.items():
            if cell_lines[cell] != clue: return False
                
        active_vertices = [v for v, deg in vertex_deg.items() if deg > 0]
        if not active_vertices: return False
            
        start_node = active_vertices[0]
        adj = {v: [] for v in active_vertices}
        for edge, state in edge_states.items():
            if state == 1:
                u, v = edge
                adj[u].append(v)
                adj[v].append(u)
                
        visited = set([start_node])
        q = [start_node]
        while q:
            curr = q.pop(0)
            for nxt in adj[curr]:
                if nxt not in visited:
                    visited.add(nxt)
                    q.append(nxt)
                    
        return len(visited) == len(active_vertices)
