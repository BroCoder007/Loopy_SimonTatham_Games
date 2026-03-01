"""
DP & Backtracking Solver
========================
Phase 3: Highly Optimized DP & Backtracking Solver

Enhancements:
- Hard timeout (default 3s) and max-state limit (default 10M)
- Clean stop via threading.Event
- Periodic metrics streaming via queue
- Incremental expansion with yield points
"""

import time
import threading
import queue
from logic.solver_worker import SolverMetrics


class DPBacktrackingSolver:
    """
    Phase 3: Highly Optimized DP & Backtracking Solver

    Now supports:
    - stop_event: threading.Event to request clean stop from outside
    - timeout: wall-clock seconds before auto-stop
    - max_states: maximum states explored before auto-stop
    - metrics_queue: optional queue.Queue to push real-time SolverMetrics
    """

    DEFAULT_TIMEOUT = 3.0
    DEFAULT_MAX_STATES = 10_000_000
    METRICS_PUSH_INTERVAL = 500    # push metrics every N nodes
    YIELD_CHECK_INTERVAL = 200     # check stop/timeout every N nodes

    def __init__(self, game_state, stop_event=None, timeout=None,
                 max_states=None, metrics_queue=None):
        self.game_state = game_state
        self.clues = game_state.clues
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.dp_cache = {}
        self.nodes_visited = 0
        self.cache_hits = 0
        self.all_edges = self._get_all_edges()

        # --- Control flags ---
        self.stop_event = stop_event or threading.Event()
        self.timeout = timeout if timeout is not None else self.DEFAULT_TIMEOUT
        self.max_states = max_states if max_states is not None else self.DEFAULT_MAX_STATES
        self.metrics_queue = metrics_queue

        # --- Status tracking ---
        self._timed_out = False
        self._stopped = False
        self._start_time = 0.0
        self._last_metrics_time = 0.0
        self._last_metrics_states = 0
        self._branch_counts = []     # track branching for metrics

        # --- Precomputed cell-edge mappings ---
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

    def solve(self, ui_callback=None, delay=0.0, ignore_current_edges=False):
        self.dp_cache.clear()
        self.nodes_visited = 0
        self.cache_hits = 0
        self._timed_out = False
        self._stopped = False
        self._branch_counts = []

        # Core edge state mapping (0 = Unknown, 1 = Line, -1 = Cross)
        edge_states = {e: 0 for e in self.all_edges}
        if not ignore_current_edges:
            for e in self.game_state.graph.edges:
                edge_states[e] = 1

        # O(E) precalculations ONCE for incremental updates
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

        self._start_time = time.perf_counter()
        self._last_metrics_time = self._start_time
        self._last_metrics_states = 0

        success = self._backtrack(
            edge_states, vertex_deg, vertex_unks,
            cell_lines, cell_unks, ui_callback, delay,
            added_line=False
        )
        end_time = time.perf_counter()

        final_edges = set(e for e, st in edge_states.items() if st == 1)

        # Determine status
        if self._timed_out or self._stopped:
            status = "Timeout"
        elif success:
            status = "Success"
        else:
            status = "NoSolution"

        return {
            "success": success,
            "status": status,
            "edges": final_edges if success else set(self.game_state.graph.edges),
            "nodes_visited": self.nodes_visited,
            "time_taken": end_time - self._start_time,
            "cache_hits": self.cache_hits,
            "timed_out": self._timed_out or self._stopped,
        }

    def _should_stop(self):
        """Check all termination conditions."""
        # Check stop event (external cancellation)
        if self.stop_event.is_set():
            self._stopped = True
            return True

        # Check state limit
        if self.nodes_visited >= self.max_states:
            self._timed_out = True
            return True

        # Check timeout (only every YIELD_CHECK_INTERVAL nodes for performance)
        if self.nodes_visited % self.YIELD_CHECK_INTERVAL == 0:
            elapsed = time.perf_counter() - self._start_time
            if elapsed >= self.timeout:
                self._timed_out = True
                return True

        return False

    def _push_metrics_if_due(self):
        """Push metrics snapshot to queue if interval elapsed."""
        if self.metrics_queue is None:
            return
        if self.nodes_visited % self.METRICS_PUSH_INTERVAL != 0:
            return

        now = time.perf_counter()
        elapsed_since_start = now - self._start_time
        elapsed_since_last = now - self._last_metrics_time
        states_delta = self.nodes_visited - self._last_metrics_states

        # Calculate time per step in this window
        time_per_step_ms = 0.0
        if states_delta > 0:
            time_per_step_ms = (elapsed_since_last * 1000.0) / states_delta

        # Calculate average branching factor from recent data
        branching = 0.0
        if self._branch_counts:
            recent = self._branch_counts[-20:]  # last 20
            branching = sum(recent) / len(recent)
            self._branch_counts = self._branch_counts[-50:]  # trim

        metrics = SolverMetrics(
            timestamp=elapsed_since_start,
            states_explored=self.nodes_visited,
            states_delta=states_delta,
            time_per_step_ms=time_per_step_ms,
            branching_factor=branching,
            interval_ms=elapsed_since_last * 1000.0,
        )

        try:
            self.metrics_queue.put_nowait(metrics)
        except queue.Full:
            pass

        self._last_metrics_time = now
        self._last_metrics_states = self.nodes_visited

    def _backtrack(self, edge_states, vertex_deg, vertex_unks,
                   cell_lines, cell_unks, ui_callback, delay, added_line):
        self.nodes_visited += 1

        # --- Periodic checks ---
        if self._should_stop():
            return False

        self._push_metrics_if_due()

        # 1. Fast Validity Constraint Pruning
        if not self._is_valid(edge_states, vertex_deg, vertex_unks,
                              cell_lines, cell_unks, added_line):
            return False

        # 2. DP Memoization Check (Subproblem caching)
        state_key = tuple(edge_states[e] for e in self.all_edges)
        if state_key in self.dp_cache:
            self.cache_hits += 1
            return self.dp_cache[state_key]

        # 3. Dynamic Node Routing via Constraint Propagation
        unknown_edge = self._get_best_unknown_edge(
            edge_states, vertex_deg, vertex_unks, cell_lines, cell_unks
        )

        if unknown_edge is None:
            # Base Case: Exhausted all edges
            if self._is_single_loop_and_complete(edge_states, vertex_deg, cell_lines):
                self.dp_cache[state_key] = True
                return True
            self.dp_cache[state_key] = False
            return False

        u, v = unknown_edge
        affected_cells = self.edge_to_cells[unknown_edge]

        # Track branching factor
        self._branch_counts.append(2)  # binary branch (line or cross)

        # Multi-branch Recursion
        for guess in [1, -1]:
            # Apply Delta
            edge_states[unknown_edge] = guess
            vertex_unks[u] -= 1
            vertex_unks[v] -= 1
            for c in affected_cells:
                if c in cell_unks:
                    cell_unks[c] -= 1

            if guess == 1:
                vertex_deg[u] = vertex_deg.get(u, 0) + 1
                vertex_deg[v] = vertex_deg.get(v, 0) + 1
                for c in affected_cells:
                    if c in cell_lines:
                        cell_lines[c] += 1

            # Conditionally decouple UI rendering via delay controls
            if ui_callback and delay > 0:
                ui_callback(set(e for e, st in edge_states.items() if st == 1))
                time.sleep(delay)

            if self._backtrack(edge_states, vertex_deg, vertex_unks,
                               cell_lines, cell_unks, ui_callback, delay,
                               added_line=(guess == 1)):
                return True

            # Revert Delta
            if guess == 1:
                vertex_deg[u] -= 1
                vertex_deg[v] -= 1
                for c in affected_cells:
                    if c in cell_lines:
                        cell_lines[c] -= 1

            vertex_unks[u] += 1
            vertex_unks[v] += 1
            for c in affected_cells:
                if c in cell_unks:
                    cell_unks[c] += 1
            edge_states[unknown_edge] = 0

        self.dp_cache[state_key] = False
        return False

    def _is_valid(self, edge_states, vertex_deg, vertex_unks,
                  cell_lines, cell_unks, added_line):
        for deg in vertex_deg.values():
            if deg > 2: return False

        for v, deg in vertex_deg.items():
            if deg == 1 and vertex_unks.get(v, 0) == 0:
                return False  # Dead un-closed branch

        for cell, clue in self.clues.items():
            lines = cell_lines.get(cell, 0)
            unks = cell_unks.get(cell, 0)
            if lines > clue: return False
            if lines + unks < clue: return False

        # Premature Multi-Loop Cycle Detection (ONLY run if we just added a line)
        if not added_line:
            return True

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
                if cell_lines.get(cell, 0) < clue: return False

        return True

    def _get_best_unknown_edge(self, edge_states, vertex_deg, vertex_unks,
                                cell_lines, cell_unks):
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

                # Implicit Constraint Propagation
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
            if deg != 0 and deg != 2: return False

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
