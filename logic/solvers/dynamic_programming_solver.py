"""
Dynamic Programming Solver
==========================
Exact deterministic row-profile DP for Slitherlink.

Core guarantees:
- No recursion
- No backtracking search
- No fallback solver calls
- No beam truncation / artificial DP state cap
- Deterministic iteration order
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from logic.solvers.solver_interface import AbstractSolver, HintPayload

Vertex = Tuple[int, int]
Edge = Tuple[Vertex, Vertex]
Profile = Tuple[int, Tuple[int, ...], bool]  # (vertical_mask, component_labels, closed_flag)
StateKey = Tuple[int, int, Tuple[int, ...], bool]  # (top_h_mask, vertical_mask, component_labels, closed_flag)
ParentRef = Tuple[StateKey, Tuple[Edge, ...]]


class DynamicProgrammingSolver(AbstractSolver):
    def __init__(self, game_state: Any):
        self.game_state = game_state
        self.rows = game_state.rows
        self.cols = game_state.cols

        self.last_explanation = ""
        self.solution_moves: List[Dict[str, Any]] = []
        self.current_move_index = 0
        self._solution_computed = False
        self._final_solution_edges: Set[Edge] = set()

        # Trace/analysis metrics
        self.dp_state_count = 0
        self.memo_hits = 0
        self.current_row = 0
        self.state_count = 0
        self.total_states_generated = 0
        self.dp_debug_logging = True

    def decide_move(self) -> Tuple[List[Tuple[Edge, int]], Optional[Edge]]:
        """
        Required by UI to simulate 'thinking' and return candidates + best move.
        DP solver uses precomputed deterministic solution moves.
        """
        is_cpu_turn = (
            self.game_state.game_mode in ["vs_cpu", "expert"]
            and self.game_state.turn == "Player 2 (CPU)"
        )

        if not is_cpu_turn:
            return [], None

        if not self._solution_computed:
            self._compute_full_solution()

        for _ in range(2):
            idx = self._find_valid_solution_move_index()
            if idx is not None:
                self.current_move_index = idx
                move_data = self.solution_moves[idx]
                move = move_data["move"]
                self.last_explanation = move_data.get("explanation", "Dynamic Programming precomputed move.")

                from logic.execution_trace import log_pure_dp_move

                log_pure_dp_move(
                    move=move,
                    explanation=self.last_explanation,
                    dp_state_count=self.dp_state_count,
                )
                return [(move, 100)], move

            self._recompute_solution()

        fallback_move, fallback_reason = self._find_fallback_move()
        if fallback_move is not None:
            self.last_explanation = fallback_reason
            from logic.execution_trace import log_pure_dp_move

            log_pure_dp_move(
                move=fallback_move,
                explanation=self.last_explanation,
                dp_state_count=self.dp_state_count,
            )
            return [(fallback_move, 90)], fallback_move

        self.last_explanation = "Dynamic Programming has completed all precomputed moves."
        return [], None

    def solve(self, board: Any = None) -> Optional[Edge]:
        """
        Returns the next move from precomputed deterministic solution.
        """
        if not self._solution_computed:
            self._compute_full_solution()

        for _ in range(2):
            idx = self._find_valid_solution_move_index()
            if idx is not None:
                self.current_move_index = idx
                move_data = self.solution_moves[idx]
                move = move_data["move"]

                from logic.execution_trace import log_pure_dp_move

                log_pure_dp_move(
                    move=move,
                    explanation=move_data.get("explanation", ""),
                    dp_state_count=self.dp_state_count,
                )
                return move

            self._recompute_solution()

        fallback_move, fallback_reason = self._find_fallback_move()
        if fallback_move is not None:
            self.last_explanation = fallback_reason
            from logic.execution_trace import log_pure_dp_move

            log_pure_dp_move(
                move=fallback_move,
                explanation=fallback_reason,
                dp_state_count=self.dp_state_count,
            )
            return fallback_move

        return None

    def generate_hint(self, board: Any = None) -> HintPayload:
        target = board if board is not None else self.game_state
        strategy_name = "Dynamic Programming (State Compression)"

        if not self._solution_computed:
            self._compute_full_solution()

        is_human_turn = (
            (self.game_state.game_mode in ["vs_cpu", "expert"] and self.game_state.turn == "Player 1 (Human)")
            or (self.game_state.game_mode not in ["vs_cpu", "expert"])
        )

        if not is_human_turn:
            return {
                "move": None,
                "strategy": strategy_name,
                "explanation": "Hints are only available during your turn.",
            }

        if not self._final_solution_edges:
            return {
                "move": None,
                "strategy": strategy_name,
                "explanation": "No valid solutions found via exact profile DP.",
            }

        current_edges = set(target.graph.edges)

        for i in range(self.current_move_index, len(self.solution_moves)):
            move_data = self.solution_moves[i]
            move = move_data["move"]

            if i == self.current_move_index:
                continue

            if move not in current_edges:
                from logic.validators import is_valid_move

                valid, _ = is_valid_move(target.graph, move[0], move[1], target.clues)
                if valid:
                    return {
                        "move": move,
                        "strategy": strategy_name,
                        "explanation": move_data.get(
                            "explanation",
                            "This edge is part of the exact DP solution path.",
                        ),
                    }

        for edge in sorted(current_edges):
            if edge not in self._final_solution_edges:
                return {
                    "move": edge,
                    "strategy": strategy_name,
                    "explanation": self._generate_edge_removal_reasoning(edge, target),
                }

        return {
            "move": None,
            "strategy": strategy_name,
            "explanation": "No specific hints available. Try any valid move.",
        }

    def explain_last_move(self) -> str:
        return self.last_explanation

    def register_move(self, move: Edge) -> None:
        """Called after a move is made to update the current move index."""
        if self.current_move_index < len(self.solution_moves):
            expected_move = self.solution_moves[self.current_move_index]["move"]
            if move == expected_move:
                self.current_move_index += 1
            else:
                self._recompute_solution()

    def _find_valid_solution_move_index(self) -> Optional[int]:
        """
        Find the first currently valid planned move in deterministic queue order.
        """
        if not self.solution_moves:
            return None

        from logic.validators import is_valid_move

        n = len(self.solution_moves)

        for idx in range(self.current_move_index, n):
            move = self.solution_moves[idx]["move"]
            valid, _ = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
            if valid:
                return idx

        for idx in range(0, self.current_move_index):
            move = self.solution_moves[idx]["move"]
            valid, _ = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
            if valid:
                return idx

        return None

    def _find_fallback_move(self) -> Tuple[Optional[Edge], str]:
        """
        Deterministic local fallback to avoid CPU stalling when queued solution moves are blocked.
        """
        from logic.validators import is_valid_move

        current_edges = set(self.game_state.graph.edges)
        target_edges = set(self._final_solution_edges)

        # 1) Prefer adding a missing target edge if currently valid.
        for edge in sorted(target_edges):
            if edge in current_edges:
                continue
            valid, _ = is_valid_move(self.game_state.graph, edge[0], edge[1], self.game_state.clues)
            if valid:
                return edge, f"Fallback: add target edge {edge}."

        # 2) Any valid add on the board.
        for edge in sorted(self._get_all_potential_edges()):
            if edge in current_edges:
                continue
            valid, _ = is_valid_move(self.game_state.graph, edge[0], edge[1], self.game_state.clues)
            if valid:
                return edge, f"Fallback: add valid edge {edge}."

        return None, "No valid fallback move found."

    def _recompute_solution(self) -> None:
        self._solution_computed = False
        self.current_move_index = 0
        self._compute_full_solution()

    def _compute_full_solution(self) -> None:
        """
        Compute a full exact solution and store deterministic moves from current board to solution.
        """
        solutions = self._run_dp(self.game_state, limit=1)
        if not solutions:
            self._final_solution_edges = set()
            self.solution_moves = []
            self._solution_computed = True
            return

        computed_solution_edges = set(solutions[0])
        self._final_solution_edges = computed_solution_edges

        current_edges = set(self.game_state.graph.edges)
        self.solution_moves = []

        # Constructive ordering: add target edges first.
        for edge in sorted(computed_solution_edges):
            if edge not in current_edges:
                self.solution_moves.append(
                    {
                        "move": edge,
                        "explanation": f"Add edge {edge} - part of the exact profile-DP solution.",
                        "dp_state_reference": "solution_edge",
                    }
                )

        # Cleanup moves second.
        for edge in sorted(current_edges):
            if edge not in computed_solution_edges:
                self.solution_moves.append(
                    {
                        "move": edge,
                        "explanation": f"Remove edge {edge} - not part of the exact profile-DP solution.",
                        "dp_state_reference": "incorrect_edge",
                    }
                )

        self._solution_computed = True

    def _get_all_potential_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edges.append(tuple(sorted(((r, c), (r, c + 1)))))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edges.append(tuple(sorted(((r, c), (r + 1, c)))))
        return edges

    def _run_dp(self, target: Any, limit: Optional[int] = 1) -> List[Set[Edge]]:
        """
        Exact deterministic row-profile DP.

        Public compatibility:
        - Returns a list of edge-sets.
        - `limit` controls output list length only.

        Internals:
        - No recursion, no search backtracking.
        - Full reachable-state exploration (no beam/state truncation).
        """
        self.dp_state_count = 0
        self.memo_hits = 0
        self.current_row = 0
        self.state_count = 0
        self.total_states_generated = 0

        rows = self.rows
        cols = self.cols
        clues = target.clues

        clue_by_row: List[Dict[int, int]] = [dict() for _ in range(rows)]
        for (r, c), val in sorted(clues.items()):
            if 0 <= r < rows and 0 <= c < cols:
                clue_by_row[r][c] = val

        current_layer: Dict[StateKey, int] = {}

        # Requested profile-level reachability table
        dp_profiles: List[Dict[Profile, bool]] = [dict() for _ in range(rows + 1)]
        for top_mask in range(1 << cols):
            initial_labels = self._initial_frontier_labels(top_mask)
            if initial_labels is None:
                continue
            state: StateKey = (top_mask, 0, initial_labels, False)
            current_layer[state] = 1
            dp_profiles[0][(0, initial_labels, False)] = True

        parent_layers: List[Dict[StateKey, ParentRef]] = [dict() for _ in range(rows + 1)]
        alt_parent_layers: List[Dict[StateKey, ParentRef]] = [dict() for _ in range(rows + 1)]

        for r in range(rows):
            self.current_row = r
            self.state_count = len(current_layer)
            self.dp_state_count += len(current_layer)
            incoming_states = len(current_layer)
            row_masks_tested = 0
            row_states_accepted = 0
            row_rejected = {
                "cell_constraint": 0,
                "vertex_degree": 0,
                "premature_loop": 0,
                "component_inconsistency": 0,
            }

            next_layer: Dict[StateKey, int] = {}
            row_clues = clue_by_row[r]
            is_last_row = r == rows - 1

            for state in sorted(current_layer):
                top_mask, vertical_mask, comp_labels, closed_flag = state
                ways_to_state = current_layer[state]

                if closed_flag and not is_last_row:
                    continue

                for bottom_mask in range(1 << cols):
                    row_masks_tested += 1
                    transition = self._transition_row(
                        row=r,
                        top_mask=top_mask,
                        incoming_vertical_mask=vertical_mask,
                        comp_labels=comp_labels,
                        closed_flag=closed_flag,
                        bottom_mask=bottom_mask,
                        row_clues=row_clues,
                        is_last_row=is_last_row,
                        rejection_counts=row_rejected,
                    )
                    if transition is None:
                        continue
                    row_states_accepted += 1

                    next_vertical_mask, next_labels, next_closed, row_edges = transition
                    profile: Profile = (next_vertical_mask, next_labels, next_closed)
                    next_key: StateKey = (bottom_mask, next_vertical_mask, next_labels, next_closed)

                    self.total_states_generated += 1
                    dp_profiles[r + 1][profile] = True

                    if next_key not in next_layer:
                        next_layer[next_key] = ways_to_state
                        parent_layers[r + 1][next_key] = (state, row_edges)
                    else:
                        self.memo_hits += 1
                        next_layer[next_key] += ways_to_state
                        if next_key not in alt_parent_layers[r + 1]:
                            alt_parent_layers[r + 1][next_key] = (state, row_edges)

            if self.dp_debug_logging:
                print(f"[DP DEBUG] Row {r}")
                print(f"  incoming_states: {incoming_states}")
                print(f"  horizontal_masks_tested: {row_masks_tested}")
                print(f"  states_accepted: {row_states_accepted}")
                print(f"  rejected_cell_constraint: {row_rejected['cell_constraint']}")
                print(f"  rejected_vertex_degree: {row_rejected['vertex_degree']}")
                print(f"  rejected_premature_loop: {row_rejected['premature_loop']}")
                print(f"  rejected_component_inconsistency: {row_rejected['component_inconsistency']}")

            current_layer = next_layer

            if not current_layer:
                self.state_count = 0
                if self.dp_debug_logging:
                    print(f"DP terminated at row {r + 1} — no valid states remain.")
                return []

        self.current_row = rows
        self.state_count = len(current_layer)

        final_states: List[Tuple[StateKey, Set[Edge]]] = []
        total_solution_count = 0

        for state in sorted(current_layer):
            top_mask, vertical_mask, comp_labels, closed_flag = state
            if vertical_mask != 0:
                continue
            if not closed_flag:
                continue
            if any(comp_labels):
                continue

            candidate_solution = self._reconstruct_solution_iterative(parent_layers, state)
            if not self._is_valid_final_solution(candidate_solution, clues):
                continue

            # Deterministic final-state selection is by lexicographic state order.
            final_states.append((state, candidate_solution))
            total_solution_count += current_layer[state]

        if not final_states:
            return []

        best_final_state, best_solution = final_states[0]

        if limit is None or limit <= 1:
            return [best_solution]

        out: List[Set[Edge]] = [best_solution]

        if total_solution_count > 1:
            second_solution = None

            for state, _ in final_states:
                if state in alt_parent_layers[rows]:
                    alt_solution = self._reconstruct_solution_iterative(
                        parent_layers,
                        state,
                        alt_parent_layers=alt_parent_layers,
                    )
                    if alt_solution != best_solution and self._is_valid_final_solution(alt_solution, clues):
                        second_solution = alt_solution
                        break

            if second_solution is None:
                for _, state_solution in final_states:
                    if state_solution != best_solution:
                        second_solution = state_solution
                        break

            if second_solution is None:
                second_solution = set(best_solution)

            out.append(second_solution)

        return out[:limit]

    def _is_valid_final_solution(self, edges: Set[Edge], clues: Dict[Tuple[int, int], int]) -> bool:
        """
        Final acceptance validation on reconstructed edge set:
        - all clues satisfied
        - no degree-1 vertices
        - exactly one connected component among active vertices only (degree > 0)
        """
        if not edges:
            return False

        for (r, c), clue in sorted(clues.items()):
            count = 0
            if tuple(sorted(((r, c), (r, c + 1)))) in edges:
                count += 1
            if tuple(sorted(((r + 1, c), (r + 1, c + 1)))) in edges:
                count += 1
            if tuple(sorted(((r, c), (r + 1, c)))) in edges:
                count += 1
            if tuple(sorted(((r, c + 1), (r + 1, c + 1)))) in edges:
                count += 1
            if count != clue:
                return False

        adjacency: Dict[Vertex, Set[Vertex]] = {}
        for u, v in edges:
            if u not in adjacency:
                adjacency[u] = set()
            if v not in adjacency:
                adjacency[v] = set()
            adjacency[u].add(v)
            adjacency[v].add(u)

        active_vertices = sorted(adjacency.keys())
        if not active_vertices:
            return False

        for vertex in active_vertices:
            degree = len(adjacency[vertex])
            if degree == 1:
                return False

        components = 0
        visited: Set[Vertex] = set()
        for start in active_vertices:
            if start in visited:
                continue
            components += 1
            stack = [start]
            visited.add(start)
            while stack:
                node = stack.pop()
                for neighbor in adjacency[node]:
                    if neighbor in visited:
                        continue
                    visited.add(neighbor)
                    stack.append(neighbor)

        return components == 1

    def _initial_frontier_labels(self, top_mask: int) -> Optional[Tuple[int, ...]]:
        """
        Build canonical odd-frontier component labels for row 0 from top boundary
        horizontal edges.
        """
        uf = _UnionFind()

        for c in range(self.cols):
            if ((top_mask >> c) & 1) == 0:
                continue
            a = c + 1
            b = c + 2
            uf.add(a)
            uf.add(b)
            uf.union(a, b)

        labels = [0] * (self.cols + 1)
        for c in range(self.cols + 1):
            left = (top_mask >> (c - 1)) & 1 if c > 0 else 0
            right = (top_mask >> c) & 1 if c < self.cols else 0
            degree = left + right

            if degree > 2:
                return None
            if degree & 1:
                vertex_id = c + 1
                uf.add(vertex_id)
                labels[c] = uf.find(vertex_id)

        return _normalize_labels(tuple(labels))

    def _transition_row(
        self,
        row: int,
        top_mask: int,
        incoming_vertical_mask: int,
        comp_labels: Tuple[int, ...],
        closed_flag: bool,
        bottom_mask: int,
        row_clues: Dict[int, int],
        is_last_row: bool,
        rejection_counts: Optional[Dict[str, int]] = None,
    ) -> Optional[Tuple[int, Tuple[int, ...], bool, Tuple[Edge, ...]]]:
        """
        Apply one deterministic row transition.

        Returns:
            (next_vertical_mask, normalized_next_labels, next_closed_flag, row_edges)
            or None if invalid.
        """

        cols = self.cols

        # Compute vertical edges into next row by parity completion at current row vertices.
        next_vertical_mask = 0
        for c in range(cols + 1):
            up = (incoming_vertical_mask >> c) & 1
            left = (top_mask >> (c - 1)) & 1 if c > 0 else 0
            right = (top_mask >> c) & 1 if c < cols else 0
            partial_degree = up + left + right

            if partial_degree > 2:
                self._count_rejection(rejection_counts, "vertex_degree")
                return None

            down = partial_degree & 1
            if down:
                next_vertical_mask |= (1 << c)

        if closed_flag:
            if top_mask != 0 or incoming_vertical_mask != 0:
                self._count_rejection(rejection_counts, "component_inconsistency")
                return None
            if bottom_mask != 0 or next_vertical_mask != 0:
                self._count_rejection(rejection_counts, "component_inconsistency")
                return None

        # Row clue exactness check.
        for c, clue in sorted(row_clues.items()):
            top = (top_mask >> c) & 1
            bottom = (bottom_mask >> c) & 1
            left = (next_vertical_mask >> c) & 1
            right = (next_vertical_mask >> (c + 1)) & 1
            if top + bottom + left + right != clue:
                self._count_rejection(rejection_counts, "cell_constraint")
                return None

        uf = _UnionFind()
        touched_ids: Set[int] = set()

        for label in comp_labels:
            if label > 0:
                uf.add(label)
                touched_ids.add(label)

        next_vertex_ids = [0] * (cols + 1)
        next_new_id = (max((x for x in comp_labels if x > 0), default=0) + 1)

        # Vertical carry: odd frontier points continue downward.
        for c in range(cols + 1):
            if ((next_vertical_mask >> c) & 1) == 0:
                continue
            source_id = comp_labels[c]
            if source_id == 0:
                self._count_rejection(rejection_counts, "component_inconsistency")
                return None
            uf.add(source_id)
            touched_ids.add(source_id)
            next_vertex_ids[c] = source_id

        # Bottom horizontal edges merge/extend components on next frontier.
        for c in range(cols):
            if ((bottom_mask >> c) & 1) == 0:
                continue

            left_id = next_vertex_ids[c]
            right_id = next_vertex_ids[c + 1]

            if left_id == 0 and right_id == 0:
                left_id = next_new_id
                right_id = next_new_id
                next_new_id += 1
                next_vertex_ids[c] = left_id
                next_vertex_ids[c + 1] = right_id
                uf.add(left_id)
                touched_ids.add(left_id)
            elif left_id == 0:
                next_vertex_ids[c] = right_id
                uf.add(right_id)
                touched_ids.add(right_id)
            elif right_id == 0:
                next_vertex_ids[c + 1] = left_id
                uf.add(left_id)
                touched_ids.add(left_id)
            else:
                uf.add(left_id)
                uf.add(right_id)
                touched_ids.add(left_id)
                touched_ids.add(right_id)
                uf.union(left_id, right_id)

        # Build next frontier odd-degree labels.
        raw_next_labels = [0] * (cols + 1)
        next_roots: Set[int] = set()

        for c in range(cols + 1):
            up = (next_vertical_mask >> c) & 1
            left = (bottom_mask >> (c - 1)) & 1 if c > 0 else 0
            right = (bottom_mask >> c) & 1 if c < cols else 0
            partial_degree = up + left + right

            if partial_degree > 2:
                self._count_rejection(rejection_counts, "vertex_degree")
                return None

            if partial_degree & 1:
                vertex_id = next_vertex_ids[c]
                if vertex_id == 0:
                    vertex_id = next_new_id
                    next_new_id += 1
                    next_vertex_ids[c] = vertex_id
                    uf.add(vertex_id)
                    touched_ids.add(vertex_id)
                root = uf.find(vertex_id)
                raw_next_labels[c] = root
                next_roots.add(root)

        active_roots = set(uf.find(x) for x in touched_ids)
        disappeared = active_roots - next_roots

        next_closed_flag = closed_flag
        if disappeared:
            if closed_flag:
                self._count_rejection(rejection_counts, "component_inconsistency")
                return None
            if len(disappeared) > 1:
                self._count_rejection(rejection_counts, "component_inconsistency")
                return None
            next_closed_flag = True

        if next_closed_flag and any(raw_next_labels):
            self._count_rejection(rejection_counts, "component_inconsistency")
            return None

        if next_closed_flag and not is_last_row:
            self._count_rejection(rejection_counts, "premature_loop")
            return None

        normalized_next_labels = _normalize_labels(tuple(raw_next_labels))

        row_edges: List[Edge] = []
        if row == 0:
            for c in range(cols):
                if (top_mask >> c) & 1:
                    row_edges.append(tuple(sorted(((0, c), (0, c + 1)))))
        for c in range(cols):
            if (bottom_mask >> c) & 1:
                row_edges.append(tuple(sorted(((row + 1, c), (row + 1, c + 1)))))
        for c in range(cols + 1):
            if (next_vertical_mask >> c) & 1:
                row_edges.append(tuple(sorted(((row, c), (row + 1, c)))))

        return next_vertical_mask, normalized_next_labels, next_closed_flag, tuple(sorted(row_edges))

    def _count_rejection(self, rejection_counts: Optional[Dict[str, int]], key: str) -> None:
        if rejection_counts is None:
            return
        if key not in rejection_counts:
            return
        rejection_counts[key] += 1

    def _reconstruct_solution_iterative(
        self,
        parent_layers: List[Dict[StateKey, ParentRef]],
        final_state: StateKey,
        alt_parent_layers: Optional[List[Dict[StateKey, ParentRef]]] = None,
    ) -> Set[Edge]:
        """
        Reconstruct one deterministic solution path iteratively (no recursion).
        """
        edges: Set[Edge] = set()
        state = final_state
        row = self.rows
        used_alt = False

        while row > 0:
            parent_ref = None
            if alt_parent_layers is not None and not used_alt and state in alt_parent_layers[row]:
                parent_ref = alt_parent_layers[row][state]
                used_alt = True
            elif state in parent_layers[row]:
                parent_ref = parent_layers[row][state]

            if parent_ref is None:
                break

            prev_state, row_edges = parent_ref
            for edge in row_edges:
                edges.add(edge)
            state = prev_state
            row -= 1

        return edges

    def _generate_edge_removal_reasoning(self, edge: Edge, target: Any) -> str:
        """
        Generate specific reasoning for why an edge should be removed.
        """
        u, v = edge

        affected_cells: List[Tuple[int, int]] = []

        if u[0] == v[0]:
            r = u[0]
            c = min(u[1], v[1])
            if r > 0:
                affected_cells.append((r - 1, c))
            if r < target.rows:
                affected_cells.append((r, c))
        else:
            c = u[1]
            r = min(u[0], v[0])
            if c > 0:
                affected_cells.append((r, c - 1))
            if c < target.cols:
                affected_cells.append((r, c))

        for cell in affected_cells:
            if cell not in target.clues:
                continue

            clue_val = target.clues[cell]
            cell_r, cell_c = cell
            current_count = 0

            edges_around = [
                tuple(sorted(((cell_r, cell_c), (cell_r, cell_c + 1)))),
                tuple(sorted(((cell_r + 1, cell_c), (cell_r + 1, cell_c + 1)))),
                tuple(sorted(((cell_r, cell_c), (cell_r + 1, cell_c)))),
                tuple(sorted(((cell_r, cell_c + 1), (cell_r + 1, cell_c + 1)))),
            ]

            for check_edge in edges_around:
                if check_edge in target.graph.edges and check_edge != edge:
                    current_count += 1

            if current_count > clue_val:
                return (
                    f"Remove this edge because cell ({cell_r}, {cell_c}) has clue {clue_val} "
                    f"but would still have {current_count} adjacent edges without it."
                )

        for node in [u, v]:
            degree = 0
            for neighbor in target.graph.get_neighbors(node):
                if tuple(sorted((node, neighbor))) in target.graph.edges:
                    degree += 1
            if degree > 2:
                return f"Remove this edge because node {node} exceeds degree 2."

        return "Remove this edge because it is not part of the exact profile-DP solution."


class _UnionFind:
    def __init__(self):
        self.parent: Dict[int, int] = {}

    def add(self, x: int) -> None:
        if x not in self.parent:
            self.parent[x] = x

    def find(self, x: int) -> int:
        if x not in self.parent:
            self.parent[x] = x
            return x

        root = x
        while self.parent[root] != root:
            root = self.parent[root]

        while self.parent[x] != x:
            parent = self.parent[x]
            self.parent[x] = root
            x = parent

        return root

    def union(self, a: int, b: int) -> int:
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return ra

        # Deterministic: smaller id is chosen as root.
        if ra < rb:
            self.parent[rb] = ra
            return ra

        self.parent[ra] = rb
        return rb


def _normalize_labels(labels: Tuple[int, ...]) -> Tuple[int, ...]:
    mapping: Dict[int, int] = {}
    out: List[int] = []
    next_id = 1

    for val in labels:
        if val == 0:
            out.append(0)
            continue
        if val not in mapping:
            mapping[val] = next_id
            next_id += 1
        out.append(mapping[val])

    return tuple(out)
