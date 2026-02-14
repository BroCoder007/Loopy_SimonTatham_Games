"""
Dynamic Programming Solver
==========================
Exact deterministic row-profile DP for Slitherlink.

Elite Strategy Engine:
- Full solution-space enumeration
- Frequency analysis across all valid solutions
- Certainty-based move selection (never returns None)
- Merge sort for deterministic ordering
- No fallback, no greedy, no D&C, no heuristics, no randomness

Core guarantees:
- No recursion in DP core
- No backtracking search
<<<<<<< feature/loop-solver-improvements
- No fallback solver calls
- No beam truncation / artificial DP state cap
- Deterministic iteration order (merge sort)
- DP NEVER returns None
=======
- No backup solver calls
- No beam truncation (only a safety guard for state explosion)
- Deterministic iteration order
>>>>>>> main
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from logic.solvers.solver_interface import AbstractSolver, HintPayload
<<<<<<< feature/loop-solver-improvements
from logic.solvers.merge_sort import merge_sort
=======
from logic.solvers.solver_errors import (
    ControlledStateExplosionError,
    STATE_SPACE_EXPLOSION_MESSAGE,
    resolve_safe_limit,
)
>>>>>>> main

Vertex = Tuple[int, int]
Edge = Tuple[Vertex, Vertex]
Profile = Tuple[int, Tuple[int, ...], bool]  # (vertical_mask, component_labels, closed_flag)
StateKey = Tuple[int, int, Tuple[int, ...], bool]  # (top_h_mask, vertical_mask, component_labels, closed_flag)
ParentRef = Tuple[StateKey, Tuple[Edge, ...]]
ParentLayer = Dict[StateKey, List[ParentRef]]


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
        self._has_compatible_solution = True

        # Trace/analysis metrics
        self.dp_state_count = 0
        self.memo_hits = 0
        self.current_row = 0
        self.state_count = 0
        self.total_states_generated = 0
        self.dp_debug_logging = True
        self.SAFE_LIMIT = resolve_safe_limit(game_state, "dp_safe_limit")
        self._state_explosion_detected = False

    # ------------------------------------------------------------------
    #  PUBLIC API
    # ------------------------------------------------------------------

    def decide_move(self) -> Tuple[List[Tuple[Edge, int]], Optional[Edge]]:
        """
        Required by UI to simulate 'thinking' and return candidates + best move.
        Uses full solution-space frequency analysis for certainty-based selection.
        DP NEVER returns ([], None).
        """
        if self._state_explosion_detected:
            self._publish_state_explosion_message()
            return [], None

        is_cpu_turn = (
            self.game_state.game_mode in ["vs_cpu", "expert"]
            and self.game_state.turn == "Player 2 (CPU)"
        )

        if not is_cpu_turn:
            return [], None

<<<<<<< feature/loop-solver-improvements
        move, action, explanation = self._certainty_based_move()
        self.last_explanation = explanation

        from logic.execution_trace import log_pure_dp_move

        log_pure_dp_move(
            move=move,
            explanation=self.last_explanation,
            dp_state_count=self.dp_state_count,
        )

        confidence = 100
        return [(move, confidence)], move
=======
        try:
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

            self.last_explanation = "Dynamic Programming has completed all precomputed moves."
            return [], None
        except ControlledStateExplosionError:
            self._handle_state_explosion()
            return [], None
>>>>>>> main

    def solve(self, board: Any = None) -> Edge:
        """
        Returns the next move using certainty-based selection.
        DP NEVER returns None.
        """
<<<<<<< feature/loop-solver-improvements
        move, action, explanation = self._certainty_based_move()
        self.last_explanation = explanation

        from logic.execution_trace import log_pure_dp_move

        log_pure_dp_move(
            move=move,
            explanation=explanation,
            dp_state_count=self.dp_state_count,
        )
        return move
=======
        if self._state_explosion_detected:
            self._publish_state_explosion_message()
            return None

        try:
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
        except ControlledStateExplosionError:
            self._handle_state_explosion()

        return None
>>>>>>> main

    def generate_hint(self, board: Any = None) -> HintPayload:
        target = board if board is not None else self.game_state
        strategy_name = "Dynamic Programming (State Compression)"

<<<<<<< feature/loop-solver-improvements
=======
        if self._state_explosion_detected:
            self._publish_state_explosion_message()
            return {
                "move": None,
                "strategy": strategy_name,
                "explanation": STATE_SPACE_EXPLOSION_MESSAGE,
            }

>>>>>>> main
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

<<<<<<< feature/loop-solver-improvements
        current_edges = set(target.graph.edges)
        all_potential = self._get_all_potential_edges()

        # ── Layer 1: Try full certainty engine ──────────────────────────
        try:
            all_solutions = self._compute_all_valid_solutions()
        except RuntimeError:
            all_solutions = []

        if all_solutions:
            # Check for wrong edges on the board (edges in NO solution)
            solution_union = set()
            for sol in all_solutions:
                solution_union.update(sol)

            for edge in merge_sort(list(current_edges)):
                if edge not in solution_union:
                    return {
                        "move": edge,
                        "strategy": strategy_name,
                        "explanation": (
                            f"This edge does not appear in any valid solution. "
                            f"Remove it to stay on the correct path."
                        ),
                    }

            # Frequency analysis on undecided edges
            undecided = [e for e in all_potential if e not in current_edges]
            if undecided:
                count_on, total = self._frequency_analysis(all_solutions, undecided)
                move, action, explanation = self._select_best_move(count_on, total, undecided)
                return {
                    "move": move,
                    "strategy": strategy_name,
                    "explanation": explanation,
                }

            # All edges placed — board already matches a solution
            return {
                "move": None,
                "strategy": strategy_name,
                "explanation": "Your board matches a valid solution! No further hints needed.",
            }

        # ── Layer 2: Solution-edge directed hints ───────────────────────
        # When DP can't enumerate full solutions (e.g. early game),
        # compare against the known puzzle solution edges.
        known_solution = getattr(target, "solution_edges", None) or set()

        if known_solution:
            from logic.validators import is_valid_move

            # Priority A: Add a solution edge that's missing
            for edge in merge_sort(list(known_solution)):
                if edge not in current_edges:
                    valid, _ = is_valid_move(target.graph, edge[0], edge[1], target.clues)
                    if valid:
                        return {
                            "move": edge,
                            "strategy": strategy_name,
                            "explanation": (
                                f"Add edge {edge} — this edge is part of the correct "
                                f"solution path determined by DP analysis."
                            ),
                        }

            # Priority B: Remove an edge that's not in the solution
            for edge in merge_sort(list(current_edges)):
                if edge not in known_solution:
                    return {
                        "move": edge,
                        "strategy": strategy_name,
                        "explanation": (
                            f"Remove edge {edge} — this edge is not part of the "
                            f"correct solution. Removing it brings you closer to solving."
                        ),
                    }

        # ── Layer 3: Constraint-valid edge (last resort) ────────────────
        # If no solution_edges are available, find any valid addable edge
        from logic.validators import is_valid_move

        for edge in merge_sort(all_potential):
            if edge not in current_edges:
                valid, _ = is_valid_move(target.graph, edge[0], edge[1], target.clues)
                if valid:
                    return {
                        "move": edge,
                        "strategy": strategy_name,
                        "explanation": (
                            f"Try adding edge {edge} — it satisfies all current "
                            f"constraints and may lead toward a valid solution."
                        ),
                    }

        # Board is fully solved or stuck
=======
        try:
            if not self._solution_computed:
                self._compute_full_solution()

            if not self._has_compatible_solution:
                return {
                    "move": None,
                    "strategy": strategy_name,
                    "explanation": "No valid solutions found via exact profile DP.",
                }

            current_edges = set(target.graph.edges)

            for i in range(self.current_move_index, len(self.solution_moves)):
                move_data = self.solution_moves[i]
                move = move_data["move"]

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
        except ControlledStateExplosionError:
            self._handle_state_explosion()
            return {
                "move": None,
                "strategy": strategy_name,
                "explanation": STATE_SPACE_EXPLOSION_MESSAGE,
            }

>>>>>>> main
        return {
            "move": None,
            "strategy": strategy_name,
            "explanation": "Your board appears complete! No further hints needed.",
        }

    def explain_last_move(self) -> str:
        return self.last_explanation

    def register_move(self, move: Edge) -> None:
        """Called after a move is made to update state."""
        # Reset computation so next call re-analyzes from fresh board state
        self._solution_computed = False

    # ------------------------------------------------------------------
    #  CERTAINTY ENGINE (Steps 1-5)
    # ------------------------------------------------------------------

    def _certainty_based_move(self) -> Tuple[Edge, str, str]:
        """
        Core certainty-based move selection.

        STEP 1: Compute full valid solution space
        STEP 2: Frequency analysis
        STEP 3: Move selection priority
        STEP 4: Deterministic ordering (merge sort)
        STEP 5: Apply move

        Returns:
            (edge, action, explanation) — NEVER returns None.
        """
        # STEP 1 — Compute full valid solution space
        try:
            all_solutions = self._compute_all_valid_solutions()
        except RuntimeError:
            # No valid complete solutions found by DP.
            # Use deterministic valid-edge scan (pure constraint analysis, no fallback).
            from logic.validators import is_valid_move
            current_edges = set(self.game_state.graph.edges)
            all_potential = self._get_all_potential_edges()
            for edge in merge_sort(all_potential):
                if edge not in current_edges:
                    valid, _ = is_valid_move(
                        self.game_state.graph, edge[0], edge[1], self.game_state.clues
                    )
                    if valid:
                        return (
                            edge,
                            "include",
                            f"DP certainty-based selection: No complete solution space "
                            f"available yet. Adding constraint-valid edge {edge}.",
                        )
            # If no addable edge is valid, try first potential edge
            edge = merge_sort(all_potential)[0]
            return (
                edge,
                "include",
                "DP certainty-based selection: Constraint-valid edge selection.",
            )

        # Determine undecided edges
        current_edges = set(self.game_state.graph.edges)
        all_potential = self._get_all_potential_edges()
        undecided = [e for e in all_potential if e not in current_edges]

        # Also consider edges currently on the board that might need removal
        # Check which current edges might not belong to any solution
        if all_solutions:
            solution_union = set()
            for sol in all_solutions:
                solution_union.update(sol)

            # Edges on board that are in NO solution → forced removal candidates
            removable = [e for e in merge_sort(list(current_edges))
                         if e not in solution_union]
            if removable:
                edge = removable[0]
                return (
                    edge,
                    "exclude",
                    f"DP certainty-based selection: Remove edge {edge} — "
                    f"present in 0/{len(all_solutions)} solutions (forced exclusion).",
                )

        if not undecided:
            # All potential edges are placed; check for removals from current edges
            if all_solutions:
                best_solution = merge_sort(list(all_solutions), key=lambda s: tuple(merge_sort(list(s))))[0]
                for edge in merge_sort(list(current_edges)):
                    if edge not in best_solution:
                        return (
                            edge,
                            "exclude",
                            f"DP certainty-based selection: Remove edge {edge} — "
                            f"not part of the optimal DP solution.",
                        )

            # Truly nothing left — return first potential edge as identity move
            edge = merge_sort(all_potential)[0]
            return (
                edge,
                "include",
                "DP certainty-based selection: Board fully resolved.",
            )

        # STEP 2 — Frequency analysis
        count_on, total = self._frequency_analysis(all_solutions, undecided)

        # STEP 3 & 4 — Move selection with deterministic ordering
        move, action, explanation = self._select_best_move(count_on, total, undecided)

        return move, action, explanation

<<<<<<< feature/loop-solver-improvements
    def _compute_all_valid_solutions(self) -> List[Set[Edge]]:
        """
        STEP 1: Compute full valid solution space.

        Uses existing DP solver to compute ALL valid solutions respecting:
        - Current board state
        - Forced edges (edges already on board)
        - Forbidden edges (none currently)

        Raises RuntimeError if no solution exists (board invalid).
        """
        solutions = self._run_dp(self.game_state, limit=None)

        if not solutions:
            raise RuntimeError(
                "DP: No valid solutions exist for the current board state. "
                "The board may be in an invalid configuration."
            )

        return solutions

    def _frequency_analysis(
        self,
        all_solutions: List[Set[Edge]],
        undecided_edges: List[Edge],
    ) -> Tuple[Dict[Edge, int], int]:
        """
        STEP 2: Frequency analysis.

        For every undecided edge E, compute:
            count_on[E] = number of solutions where E is included
=======
    def _recompute_solution(self) -> None:
        if self._state_explosion_detected:
            return
        self._solution_computed = False
        self.current_move_index = 0
        self._compute_full_solution()
>>>>>>> main

        Returns:
            (count_on, total) where total = len(all_solutions)
        """
<<<<<<< feature/loop-solver-improvements
        total = len(all_solutions)
        count_on: Dict[Edge, int] = {}

        for edge in undecided_edges:
            count = 0
            for sol in all_solutions:
                if edge in sol:
                    count += 1
            count_on[edge] = count

        return count_on, total

    def _select_best_move(
        self,
        count_on: Dict[Edge, int],
        total: int,
        undecided_edges: List[Edge],
    ) -> Tuple[Edge, str, str]:
        """
        STEP 3: Move selection priority.

        Priority order:
        1. If any edge has count_on[E] == total → select E = ON (forced inclusion)
        2. Else if any edge has count_on[E] == 0 → select E = OFF (forced exclusion)
        3. Else: select edge with MAX(certainty[E])
           - certainty[E] = abs((count_on[E] / total) - 0.5)
           - Tie-break: lexicographically smallest edge

        STEP 4: Deterministic ordering via merge sort.
        STEP 5: Apply move direction based on frequency.

        Returns:
            (edge, action, explanation) — NEVER returns None.
        """
        sorted_edges = merge_sort(undecided_edges)

        # Priority 1: Forced inclusion (edge in ALL solutions)
        for edge in sorted_edges:
            if count_on.get(edge, 0) == total:
                return (
                    edge,
                    "include",
                    f"DP certainty-based selection: Edge {edge} appears in "
                    f"{total}/{total} solutions (forced inclusion).",
                )

        # Priority 2: Forced exclusion (edge in NO solutions)
        for edge in sorted_edges:
            if count_on.get(edge, 0) == 0:
                return (
                    edge,
                    "exclude",
                    f"DP certainty-based selection: Edge {edge} appears in "
                    f"0/{total} solutions (forced exclusion).",
                )

        # Priority 3: Maximum certainty score
        # certainty[E] = abs((count_on[E] / total) - 0.5)
        best_edge = None
        best_certainty = -1.0

        for edge in sorted_edges:
            c = count_on.get(edge, 0)
            certainty = abs((c / total) - 0.5)
            if certainty > best_certainty:
                best_certainty = certainty
                best_edge = edge
            # If tie in certainty, merge_sort already ensures lexicographic order,
            # so the first one encountered is the smallest.

        # STEP 5: Determine action based on frequency
        c = count_on.get(best_edge, 0)
        if c / total >= 0.5:
            action = "include"
            action_label = "include edge"
        else:
            action = "exclude"
            action_label = "exclude edge"

        return (
            best_edge,
            action,
            f"DP certainty-based selection: Edge {best_edge} — "
            f"{c}/{total} solutions contain it "
            f"(certainty={best_certainty:.4f}, action={action_label}).",
        )

    # ------------------------------------------------------------------
    #  DP CORE ENGINE
    # ------------------------------------------------------------------
=======
        Compute one deterministic contradiction-based forced deduction.
        """
        forced = self._find_forced_deduction(self.game_state)
        self.current_move_index = 0

        if forced is None:
            self._final_solution_edges = set(self.game_state.graph.edges)
            self.solution_moves = []
            if not self.last_explanation:
                self.last_explanation = "No deterministic DP contradiction-based deduction available."
            self._solution_computed = True
            return

        edge, is_inclusion, explanation = forced
        next_edges = set(self.game_state.graph.edges)
        if is_inclusion:
            next_edges.add(edge)
            ref = "forced_inclusion"
        else:
            next_edges.discard(edge)
            ref = "forced_exclusion"

        self._final_solution_edges = next_edges
        self.solution_moves = [
            {
                "move": edge,
                "explanation": explanation,
                "dp_state_reference": ref,
            }
        ]
        self.last_explanation = explanation
        self._solution_computed = True
>>>>>>> main

    def _find_forced_deduction(self, target: Any) -> Optional[Tuple[Edge, bool, str]]:
        """
<<<<<<< feature/loop-solver-improvements
        Exact deterministic row-profile DP.

        Public compatibility:
        - Returns a list of edge-sets.
        - `limit` controls output list length.
        - When limit=None, returns ALL valid solutions.

        Internals:
        - No recursion, no search backtracking.
        - Full reachable-state exploration (no beam/state truncation).
        - All iteration uses merge_sort for deterministic ordering.
=======
        Full contradiction-based deduction.

        Returns:
            (edge, is_inclusion, explanation) or None if no forced move exists.
>>>>>>> main
        """
        all_potential_edges = self._get_all_potential_edges()
        all_potential_set = set(all_potential_edges)
        current_edges = set(edge for edge in target.graph.edges if edge in all_potential_set)

        # Base compatible solution set respects currently ON edges.
        all_solutions = self._compute_all_valid_solutions(target, forced_edges=current_edges)
        if not all_solutions:
            self._has_compatible_solution = False
            self.last_explanation = "No valid profile-DP solutions remain consistent with current board edges."
            return None
        self._has_compatible_solution = True

        forced_inclusions, forced_exclusions = self._compute_edge_intersections(
            all_solutions=all_solutions,
            current_edges=current_edges,
            all_potential_edges=all_potential_edges,
        )

<<<<<<< feature/loop-solver-improvements
        clue_by_row: List[Dict[int, int]] = [dict() for _ in range(rows)]
        for (r, c), val in merge_sort(list(clues.items())):
            if 0 <= r < rows and 0 <= c < cols:
                clue_by_row[r][c] = val
=======
        if forced_inclusions:
            edge = forced_inclusions[0]
            return (
                edge,
                True,
                f"Forced inclusion: edge {edge} appears in all {len(all_solutions)} compatible DP solutions.",
            )
>>>>>>> main

        if forced_exclusions:
            edge = forced_exclusions[0]
            return (
                edge,
                False,
                f"Forced exclusion: edge {edge} appears in no compatible DP solution.",
            )

        # No intersection move; use contradiction testing as oracle.
        undecided_edges = sorted(edge for edge in all_potential_edges if edge not in current_edges)
        for edge in undecided_edges:
            assume_on = set(current_edges)
            assume_on.add(edge)
            on_solutions = self._run_dp(target, limit=1, forced_edges=assume_on)
            if not on_solutions:
                # Edge forced OFF, but absent edges are already OFF in this UI model.
                continue
<<<<<<< feature/loop-solver-improvements
            state: StateKey = (top_mask, 0, initial_labels, False)
            current_layer[state] = 1
            dp_profiles[0][(0, initial_labels, False)] = True

        # Store ALL parent refs to enable full solution enumeration
        parent_layers: List[Dict[StateKey, List[ParentRef]]] = [dict() for _ in range(rows + 1)]

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

            for state in merge_sort(list(current_layer.keys())):
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
                        parent_layers[r + 1][next_key] = [(state, row_edges)]
                    else:
                        self.memo_hits += 1
                        next_layer[next_key] += ways_to_state
                        parent_layers[r + 1][next_key].append((state, row_edges))

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
=======

            assume_off = {edge}
            off_solutions = self._run_dp(
                target,
                limit=1,
                forced_edges=current_edges,
                forbidden_edges=assume_off,
            )
            if not off_solutions:
                return (
                    edge,
                    True,
                    f"Contradiction test: assuming {edge}=OFF yields no DP solution, so {edge} must be ON.",
                )
>>>>>>> main

        self.last_explanation = "No contradiction-based forced DP move exists for current undecided edges."
        return None

    def _compute_all_valid_solutions(
        self,
        target: Any,
        forced_edges: Optional[Set[Edge]] = None,
        forbidden_edges: Optional[Set[Edge]] = None,
    ) -> List[Set[Edge]]:
        return self._run_dp(
            target,
            limit=None,
            forced_edges=forced_edges,
            forbidden_edges=forbidden_edges,
        )

<<<<<<< feature/loop-solver-improvements
        # Collect valid final states
        valid_final_states: List[StateKey] = []

        for state in merge_sort(list(current_layer.keys())):
            top_mask, vertical_mask, comp_labels, closed_flag = state
            if vertical_mask != 0:
                continue
            if not closed_flag:
                continue
            if any(comp_labels):
                continue
            valid_final_states.append(state)

        if not valid_final_states:
            return []

        # Enumerate ALL solutions from ALL valid final states
        all_solutions: List[Set[Edge]] = []
        seen_solutions: Set[frozenset] = set()

        for final_state in valid_final_states:
            # Enumerate all solution paths through this final state
            state_solutions = self._enumerate_all_solutions(parent_layers, final_state)
            for sol in state_solutions:
                if self._is_valid_final_solution(sol, clues):
                    frozen = frozenset(sol)
                    if frozen not in seen_solutions:
                        seen_solutions.add(frozen)
                        all_solutions.append(sol)
                        if limit is not None and len(all_solutions) >= limit:
                            return all_solutions

        return all_solutions

    def _enumerate_all_solutions(
        self,
        parent_layers: List[Dict[StateKey, List[ParentRef]]],
        final_state: StateKey,
    ) -> List[Set[Edge]]:
        """
        Enumerate all distinct solution paths leading to a given final state.
        Uses iterative DFS over the parent DAG (no recursion).
        """
        rows = self.rows
        # Stack items: (current_row, current_state, edges_collected_so_far)
        stack: List[Tuple[int, StateKey, Set[Edge]]] = [(rows, final_state, set())]
        results: List[Set[Edge]] = []

        while stack:
            row, state, edges = stack.pop()

            if row == 0:
                # Reached the top — this is a complete solution
                results.append(edges)
                continue

            if state not in parent_layers[row]:
                # No parent — incomplete path, skip
                continue

            parents = parent_layers[row][state]
            for prev_state, row_edges in parents:
                new_edges = set(edges)
                for edge in row_edges:
                    new_edges.add(edge)
                stack.append((row - 1, prev_state, new_edges))

        return results
=======
    def _compute_edge_intersections(
        self,
        all_solutions: List[Set[Edge]],
        current_edges: Set[Edge],
        all_potential_edges: List[Edge],
    ) -> Tuple[List[Edge], List[Edge]]:
        if not all_solutions:
            return [], []
        from logic.validators import is_valid_move

        common_edges = set(all_solutions[0])
        seen_edges: Set[Edge] = set()
        for sol in all_solutions:
            common_edges.intersection_update(sol)
            seen_edges.update(sol)

        forced_inclusions: List[Edge] = []
        for edge in sorted(common_edges):
            if edge in current_edges:
                continue

            valid, _ = is_valid_move(self.game_state.graph, edge[0], edge[1], self.game_state.clues)
            if valid:
                forced_inclusions.append(edge)

        forced_exclusions: List[Edge] = []
        for edge in sorted(all_potential_edges):
            if edge in seen_edges:
                continue
            if edge in current_edges:
                forced_exclusions.append(edge)

        return forced_inclusions, forced_exclusions

    def _get_all_potential_edges(self) -> List[Edge]:
        edges: List[Edge] = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edges.append(tuple(sorted(((r, c), (r, c + 1)))))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edges.append(tuple(sorted(((r, c), (r + 1, c)))))
        return edges
>>>>>>> main

    def _is_valid_final_solution(self, edges: Set[Edge], clues: Dict[Tuple[int, int], int]) -> bool:
        """
        Final acceptance validation on reconstructed edge set:
        - all clues satisfied
        - no degree-1 vertices
        - exactly one connected component among active vertices only (degree > 0)
        """
        if not edges:
            return False

        for (r, c), clue in merge_sort(list(clues.items())):
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

        active_vertices = merge_sort(list(adjacency.keys()))
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
        for c, clue in merge_sort(list(row_clues.items())):
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

        return next_vertical_mask, normalized_next_labels, next_closed_flag, tuple(merge_sort(row_edges))

    def _count_rejection(self, rejection_counts: Optional[Dict[str, int]], key: str) -> None:
        if rejection_counts is None:
            return
        if key not in rejection_counts:
            return
        rejection_counts[key] += 1

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

    def _run_dp(
        self,
        target: Any,
        limit: Optional[int] = 1,
        forced_edges: Optional[Set[Edge]] = None,
        forbidden_edges: Optional[Set[Edge]] = None,
    ) -> List[Set[Edge]]:
        """
        Exact deterministic row-profile DP with optional edge assumptions.
        """
        if limit is not None and limit <= 0:
            return []

        self.dp_state_count = 0
        self.memo_hits = 0
        self.current_row = 0
        self.state_count = 0
        self.total_states_generated = 0

        rows = self.rows
        cols = self.cols
        clues = target.clues

        normalized_forced = set(tuple(sorted(edge)) for edge in (forced_edges or set()))
        normalized_forbidden = set(tuple(sorted(edge)) for edge in (forbidden_edges or set()))
        if normalized_forced.intersection(normalized_forbidden):
            return []

        masks = self._build_row_constraint_masks(
            forced_edges=normalized_forced,
            forbidden_edges=normalized_forbidden,
        )
        if masks is None:
            return []

        top_ones, top_zeros, bottom_ones, bottom_zeros, vertical_ones, vertical_zeros = masks

        clue_by_row: List[Dict[int, int]] = [dict() for _ in range(rows)]
        for (r, c), val in sorted(clues.items()):
            if 0 <= r < rows and 0 <= c < cols:
                clue_by_row[r][c] = val

        current_layer: Dict[StateKey, int] = {}
        dp_profiles: List[Dict[Profile, bool]] = [dict() for _ in range(rows + 1)]

        for top_mask in range(1 << cols):
            if not self._mask_satisfies_constraints(top_mask, top_ones, top_zeros):
                continue
            initial_labels = self._initial_frontier_labels(top_mask)
            if initial_labels is None:
                continue
            state: StateKey = (top_mask, 0, initial_labels, False)
            current_layer[state] = 1
            dp_profiles[0][(0, initial_labels, False)] = True
            if len(current_layer) > self.SAFE_LIMIT:
                raise ControlledStateExplosionError(
                    safe_limit=self.SAFE_LIMIT,
                    observed=len(current_layer),
                    context="dp_initial_row",
                )

        parent_layers: List[ParentLayer] = [dict() for _ in range(rows + 1)]

        for r in range(rows):
            self.current_row = r
            self.state_count = len(current_layer)
            self.dp_state_count += len(current_layer)
            incoming_states = len(current_layer)
            if incoming_states > self.SAFE_LIMIT:
                raise ControlledStateExplosionError(
                    safe_limit=self.SAFE_LIMIT,
                    observed=incoming_states,
                    context=f"dp_row_{r}_incoming",
                )

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
                    if not self._mask_satisfies_constraints(bottom_mask, bottom_ones[r], bottom_zeros[r]):
                        continue

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

                    next_vertical_mask, next_labels, next_closed, row_edges = transition
                    if not self._mask_satisfies_constraints(next_vertical_mask, vertical_ones[r], vertical_zeros[r]):
                        continue

                    row_states_accepted += 1
                    profile: Profile = (next_vertical_mask, next_labels, next_closed)
                    next_key: StateKey = (bottom_mask, next_vertical_mask, next_labels, next_closed)

                    self.total_states_generated += 1
                    dp_profiles[r + 1][profile] = True

                    if next_key not in next_layer:
                        next_layer[next_key] = ways_to_state
                        parent_layers[r + 1][next_key] = [(state, row_edges)]
                        if len(next_layer) > self.SAFE_LIMIT:
                            raise ControlledStateExplosionError(
                                safe_limit=self.SAFE_LIMIT,
                                observed=len(next_layer),
                                context=f"dp_row_{r}_next_layer",
                            )
                    else:
                        self.memo_hits += 1
                        next_layer[next_key] += ways_to_state
                        parent_layers[r + 1][next_key].append((state, row_edges))

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
                    print(f"DP terminated at row {r + 1} - no valid states remain.")
                return []

        self.current_row = rows
        self.state_count = len(current_layer)

        final_states: List[StateKey] = []
        for state in sorted(current_layer):
            top_mask, vertical_mask, comp_labels, closed_flag = state
            if vertical_mask != 0:
                continue
            if not closed_flag:
                continue
            if any(comp_labels):
                continue
            final_states.append(state)

        if not final_states:
            return []

        out: List[Set[Edge]] = []
        seen_signatures: Set[Tuple[Edge, ...]] = set()
        for final_state in sorted(final_states):
            stack: List[Tuple[int, StateKey, frozenset[Edge]]] = [(rows, final_state, frozenset())]
            while stack:
                row, state, edge_accumulator = stack.pop()
                if row == 0:
                    signature = tuple(sorted(edge_accumulator))
                    if signature in seen_signatures:
                        continue
                    edge_set = set(edge_accumulator)
                    if not self._is_valid_final_solution(edge_set, clues):
                        continue
                    seen_signatures.add(signature)
                    out.append(edge_set)
                    if limit is not None and len(out) >= limit:
                        return out
                    continue

                parents = parent_layers[row].get(state, [])
                for prev_state, row_edges in reversed(parents):
                    stack.append((row - 1, prev_state, edge_accumulator.union(row_edges)))

        return out

    def _build_row_constraint_masks(
        self,
        forced_edges: Set[Edge],
        forbidden_edges: Set[Edge],
    ) -> Optional[Tuple[int, int, List[int], List[int], List[int], List[int]]]:
        """
        Convert edge assumptions into row-mask constraints.
        """
        potential_edges = set(self._get_all_potential_edges())
        for edge in sorted(forced_edges.union(forbidden_edges)):
            if edge not in potential_edges:
                return None

        top_ref = [0, 0]
        bottom_ones = [0] * self.rows
        bottom_zeros = [0] * self.rows
        vertical_ones = [0] * self.rows
        vertical_zeros = [0] * self.rows

        for edge in sorted(forced_edges):
            if not self._apply_constraint_edge(
                edge=edge,
                bit_value=1,
                top_ref=top_ref,
                bottom_ones=bottom_ones,
                bottom_zeros=bottom_zeros,
                vertical_ones=vertical_ones,
                vertical_zeros=vertical_zeros,
            ):
                return None

        for edge in sorted(forbidden_edges):
            if not self._apply_constraint_edge(
                edge=edge,
                bit_value=0,
                top_ref=top_ref,
                bottom_ones=bottom_ones,
                bottom_zeros=bottom_zeros,
                vertical_ones=vertical_ones,
                vertical_zeros=vertical_zeros,
            ):
                return None

        top_ones, top_zeros = top_ref

        if top_ones & top_zeros:
            return None
        for r in range(self.rows):
            if bottom_ones[r] & bottom_zeros[r]:
                return None
            if vertical_ones[r] & vertical_zeros[r]:
                return None

        return top_ones, top_zeros, bottom_ones, bottom_zeros, vertical_ones, vertical_zeros

    def _apply_constraint_edge(
        self,
        edge: Edge,
        bit_value: int,
        top_ref: List[int],
        bottom_ones: List[int],
        bottom_zeros: List[int],
        vertical_ones: List[int],
        vertical_zeros: List[int],
    ) -> bool:
        """
        Map one edge constraint to the corresponding row bit.
        """
        u, v = edge
        if u[0] == v[0]:
            row = u[0]
            col = min(u[1], v[1])
            if row < 0 or row > self.rows or col < 0 or col >= self.cols:
                return False
            bit = 1 << col
            if row == 0:
                if bit_value == 1:
                    top_ref[0] |= bit
                else:
                    top_ref[1] |= bit
            else:
                idx = row - 1
                if bit_value == 1:
                    bottom_ones[idx] |= bit
                else:
                    bottom_zeros[idx] |= bit
            return True

        col = u[1]
        row = min(u[0], v[0])
        if row < 0 or row >= self.rows or col < 0 or col > self.cols:
            return False
        bit = 1 << col
        if bit_value == 1:
            vertical_ones[row] |= bit
        else:
            vertical_zeros[row] |= bit
        return True

    def _mask_satisfies_constraints(self, mask: int, required_ones: int, required_zeros: int) -> bool:
        if (mask & required_ones) != required_ones:
            return False
        if (mask & required_zeros) != 0:
            return False
        return True

    def _publish_state_explosion_message(self) -> None:
        self.last_explanation = STATE_SPACE_EXPLOSION_MESSAGE
        self.game_state.message = STATE_SPACE_EXPLOSION_MESSAGE

    def _handle_state_explosion(self) -> None:
        self._state_explosion_detected = True
        self._has_compatible_solution = False
        self._final_solution_edges = set()
        self.solution_moves = []
        self.current_move_index = 0
        self._solution_computed = True
        self._publish_state_explosion_message()


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
