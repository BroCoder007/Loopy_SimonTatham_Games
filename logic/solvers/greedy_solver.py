"""
Greedy Solver
=============
Pure local rule-based propagation engine.

Design constraints:
- No recursion
- No backtracking
- No global search
- No DP logic
- No randomness
- No cross-calls to other solvers
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from logic.solvers.solver_interface import AbstractSolver, HintPayload

# Move/Edge type: ((r1, c1), (r2, c2))
Move = Tuple[Tuple[int, int], Tuple[int, int]]
RuleProposal = Tuple[Move, bool, str, str]  # (edge, desired_on, rule_name, explanation)


class GreedySolver(AbstractSolver):
    """
    Local deterministic rule engine.

    Core loop:
        while changes_occur:
            apply deterministic local rules in fixed order
    """

    def __init__(self, game_state: Any):
        self.game_state = game_state
        self._last_explanation: str = ""
        self._pending_explanations: Dict[Move, str] = {}
        self._all_edges: List[Move] = self._enumerate_all_edges()
        self._last_partial_state: Dict[str, Any] = {
            "on_edges": set(),
            "off_edges": set(),
            "unknown_edges": set(),
        }

    # ---- AbstractSolver API -------------------------------------------------
    def solve(self, board: Any = None):
        _, best_move = self.decide_move()
        return best_move

    def generate_hint(self, board: Any = None) -> HintPayload:
        # Hint preview must be strategy-local and must not mutate real board state.
        # Use a one-iteration local rule pass on cloned internal state.
        state: Dict[str, Set[Move]] = {
            "on": set(self.game_state.graph.edges),
            "off": set(),
        }
        current_on = set(state["on"])

        for rule_fn in (
            self._rule_clue_zero,
            self._rule_clue_three,
            self._rule_cell_excess,
            self._rule_cell_deficit,
            self._rule_vertex_degree,
            self._rule_two_edge_vertex_completion,
            self._rule_forced_edge,
        ):
            proposals = rule_fn(state)
            for proposal in proposals:
                if not self._apply_proposal(state, proposal):
                    continue

                move, explanation = self._proposal_to_hint(proposal, current_on)
                if move is not None:
                    return {
                        "move": move,
                        "explanation": explanation,
                        "strategy": "Greedy",
                    }

        return {
            "move": None,
            "explanation": "No local greedy deduction available.",
            "strategy": "Greedy",
        }

    def explain_last_move(self) -> str:
        return self._last_explanation

    # ---- Public integration API --------------------------------------------
    def decide_move(self) -> Tuple[List[Tuple[Move, int]], Optional[Move]]:
        state, action_log = self._propagate_rules()
        self._last_partial_state = self._build_partial_state(state)

        candidates, reason_by_move = self._extract_actionable_moves(state, action_log)
        self._pending_explanations = reason_by_move

        best_move = candidates[0][0] if candidates else None
        if best_move:
            self._last_explanation = reason_by_move.get(
                best_move,
                "Using Greedy Strategy: Local rule propagation selected this edge.",
            )
        else:
            self._last_explanation = "Using Greedy Strategy: No local rule applies; the board is locally stable."

        return candidates, best_move

    def make_move(self):
        _, best_move = self.decide_move()
        self.register_move(best_move)
        return best_move

    def register_move(self, move):
        if move is None:
            self.game_state.last_cpu_move_info = {
                "move": None,
                "explanation": "Using Greedy Strategy: No local rule applies; no move selected.",
                "strategy": "Greedy",
            }
            self._last_explanation = self.game_state.last_cpu_move_info["explanation"]
            return

        explanation = self._pending_explanations.get(
            move,
            "Using Greedy Strategy: Local rule propagation selected this edge.",
        )

        if not explanation.startswith("Using Greedy Strategy:"):
            explanation = f"Using Greedy Strategy: {explanation}"

        self.game_state.last_cpu_move_info = {
            "move": move,
            "explanation": explanation,
            "strategy": "Greedy",
        }
        self._last_explanation = explanation

        # Keep trace integration for UI analysis.
        try:
            from logic.execution_trace import log_greedy_move

            log_greedy_move(move, explanation)
        except Exception:
            pass

    def get_partial_board_state(self) -> Dict[str, Any]:
        """
        Return the last propagated partial board state.
        """
        return {
            "on_edges": set(self._last_partial_state["on_edges"]),
            "off_edges": set(self._last_partial_state["off_edges"]),
            "unknown_edges": set(self._last_partial_state["unknown_edges"]),
        }

    # ---- Rule Engine --------------------------------------------------------
    def _propagate_rules(self) -> Tuple[Dict[str, Set[Move]], List[RuleProposal]]:
        state: Dict[str, Set[Move]] = {
            "on": set(self.game_state.graph.edges),
            "off": set(),
        }
        action_log: List[RuleProposal] = []

        changes_occur = True
        while changes_occur:
            changes_occur = False

            for rule_fn in (
                self._rule_clue_zero,
                self._rule_clue_three,
                self._rule_cell_excess,
                self._rule_cell_deficit,
                self._rule_vertex_degree,
                self._rule_two_edge_vertex_completion,
                self._rule_forced_edge,
            ):
                proposals = rule_fn(state)
                for proposal in proposals:
                    if self._apply_proposal(state, proposal):
                        action_log.append(proposal)
                        changes_occur = True

        return state, action_log

    def _rule_clue_zero(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for cell, clue in self._iter_clues():
            if clue != 0:
                continue
            for edge in self._cell_edges(cell):
                proposals.append(
                    (
                        edge,
                        False,
                        "Clue 0 Rule",
                        f"Using Greedy Strategy: Clue 0 Rule at cell {cell}; edge {edge} must be OFF.",
                    )
                )
        return proposals

    def _rule_clue_three(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for cell, clue in self._iter_clues():
            if clue != 3:
                continue
            for edge in self._cell_edges(cell):
                proposals.append(
                    (
                        edge,
                        True,
                        "Clue 3 Rule",
                        f"Using Greedy Strategy: Clue 3 Rule at cell {cell}; edge {edge} must be ON.",
                    )
                )
        return proposals

    def _rule_cell_excess(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for cell, clue in self._iter_clues():
            on_count = self._count_on_edges_around_cell(cell, state)
            if on_count != clue:
                continue

            for edge in self._cell_edges(cell):
                if edge in state["on"] or edge in state["off"]:
                    continue
                proposals.append(
                    (
                        edge,
                        False,
                        "Cell Excess Rule",
                        f"Using Greedy Strategy: Cell Excess Rule at cell {cell}; remaining edge {edge} is OFF.",
                    )
                )
        return proposals

    def _rule_cell_deficit(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for cell, clue in self._iter_clues():
            edges = self._cell_edges(cell)
            on_count = self._count_on_edges_around_cell(cell, state)
            unknown_edges = [e for e in edges if e not in state["on"] and e not in state["off"]]
            needed = clue - on_count

            if needed <= 0:
                continue
            if len(unknown_edges) != needed:
                continue

            for edge in unknown_edges:
                proposals.append(
                    (
                        edge,
                        True,
                        "Cell Deficit Rule",
                        f"Using Greedy Strategy: Cell Deficit Rule at cell {cell}; edge {edge} must be ON.",
                    )
                )
        return proposals

    def _rule_vertex_degree(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for vertex in self.game_state.graph.vertices:
            degree = self._vertex_degree(vertex, state)
            if degree != 2:
                continue

            for edge in self._incident_edges(vertex):
                if edge in state["on"] or edge in state["off"]:
                    continue
                proposals.append(
                    (
                        edge,
                        False,
                        "Vertex Degree Rule",
                        f"Using Greedy Strategy: Vertex Degree Rule at vertex {vertex}; edge {edge} must be OFF.",
                    )
                )
        return proposals

    def _rule_two_edge_vertex_completion(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for vertex in self.game_state.graph.vertices:
            degree = self._vertex_degree(vertex, state)
            if degree != 1:
                continue

            undecided = [
                edge
                for edge in self._incident_edges(vertex)
                if edge not in state["on"] and edge not in state["off"]
            ]
            if len(undecided) != 1:
                continue

            edge = undecided[0]
            proposals.append(
                (
                    edge,
                    True,
                    "Two-Edge Vertex Completion",
                    f"Using Greedy Strategy: Two-Edge Vertex Completion at vertex {vertex}; edge {edge} must be ON.",
                )
            )
        return proposals

    def _rule_forced_edge(self, state: Dict[str, Set[Move]]) -> List[RuleProposal]:
        proposals: List[RuleProposal] = []
        for cell, clue in self._iter_clues():
            edges = self._cell_edges(cell)
            on_count = self._count_on_edges_around_cell(cell, state)
            unknown_edges = [e for e in edges if e not in state["on"] and e not in state["off"]]

            # If removing an edge would violate clue -> edge must be ON.
            for edge in edges:
                if edge in state["off"]:
                    continue

                on_without = on_count - (1 if edge in state["on"] else 0)
                unknown_without = len(unknown_edges) - (1 if edge in unknown_edges else 0)
                max_possible_without_edge = on_without + unknown_without

                if max_possible_without_edge < clue:
                    proposals.append(
                        (
                            edge,
                            True,
                            "Forced Edge Rule",
                            f"Using Greedy Strategy: Forced Edge Rule at cell {cell}; edge {edge} must be ON.",
                        )
                    )

            # If adding an edge would exceed clue -> edge must be OFF.
            if on_count > clue:
                for edge in edges:
                    if edge in state["on"]:
                        proposals.append(
                            (
                                edge,
                                False,
                                "Forced Edge Rule",
                                f"Using Greedy Strategy: Forced Edge Rule at cell {cell}; edge {edge} must be OFF.",
                            )
                        )

            if on_count + 1 > clue:
                for edge in unknown_edges:
                    proposals.append(
                        (
                            edge,
                            False,
                            "Forced Edge Rule",
                            f"Using Greedy Strategy: Forced Edge Rule at cell {cell}; edge {edge} must be OFF.",
                        )
                    )

        return proposals

    def _apply_proposal(self, state: Dict[str, Set[Move]], proposal: RuleProposal) -> bool:
        edge, desired_on, _, _ = proposal

        if desired_on:
            if edge in state["on"] or edge in state["off"]:
                return False
            if not self._can_set_on(edge, state):
                return False
            state["on"].add(edge)
            return True

        # desired OFF
        if edge in state["off"] and edge not in state["on"]:
            return False
        if edge in state["on"]:
            state["on"].remove(edge)
        state["off"].add(edge)
        return True

    # ---- Candidate extraction ----------------------------------------------
    def _extract_actionable_moves(
        self,
        state: Dict[str, Set[Move]],
        action_log: List[RuleProposal],
    ) -> Tuple[List[Tuple[Move, int]], Dict[Move, str]]:
        current_on = set(self.game_state.graph.edges)
        seen: Set[Move] = set()
        candidates: List[Tuple[Move, int]] = []
        reason_by_move: Dict[Move, str] = {}

        for edge, _, _, explanation in action_log:
            if edge in seen:
                continue

            final_on = edge in state["on"]
            should_add = final_on and edge not in current_on
            should_remove = (not final_on) and edge in current_on

            if not (should_add or should_remove):
                continue

            seen.add(edge)
            candidates.append((edge, 1))
            reason_by_move[edge] = explanation

        return candidates, reason_by_move

    def _proposal_to_hint(self, proposal: RuleProposal, current_on: Set[Move]) -> Tuple[Optional[Move], str]:
        edge, desired_on, rule_name, _ = proposal
        if desired_on:
            if edge in current_on:
                return None, ""
            return edge, self._format_hint_explanation(rule_name, edge, desired_on)

        # desired OFF
        if edge in current_on:
            return edge, self._format_hint_explanation(rule_name, edge, desired_on)
        return None, ""

    def _format_hint_explanation(self, rule_name: str, edge: Move, desired_on: bool) -> str:
        cells = self._adjacent_cells(edge)
        if rule_name == "Clue 3 Rule" and cells:
            for cell in cells:
                if cell in self.game_state.clues and self.game_state.clues[cell] == 3:
                    return f"Greedy Rule: Cell {cell} has clue 3 -> all remaining edges must be added."
        if rule_name == "Clue 0 Rule" and cells:
            for cell in cells:
                if cell in self.game_state.clues and self.game_state.clues[cell] == 0:
                    return f"Greedy Rule: Cell {cell} has clue 0 -> adjacent edges must be removed."
        if rule_name == "Two-Edge Vertex Completion":
            u, v = edge
            if self._vertex_degree(u, {"on": set(self.game_state.graph.edges), "off": set()}) == 1:
                return f"Greedy Rule: Vertex {u} has only one possible edge remaining -> forced edge."
            if self._vertex_degree(v, {"on": set(self.game_state.graph.edges), "off": set()}) == 1:
                return f"Greedy Rule: Vertex {v} has only one possible edge remaining -> forced edge."

        action = "added" if desired_on else "removed"
        return f"Greedy Rule: {rule_name} -> edge {edge} is forced and should be {action}."

    def _build_partial_state(self, state: Dict[str, Set[Move]]) -> Dict[str, Any]:
        on_edges = set(state["on"])
        off_edges = set(state["off"])
        unknown_edges = set(self._all_edges) - on_edges - off_edges
        return {
            "on_edges": on_edges,
            "off_edges": off_edges,
            "unknown_edges": unknown_edges,
        }

    # ---- Local geometry helpers --------------------------------------------
    def _enumerate_all_edges(self) -> List[Move]:
        edges: List[Move] = []
        rows = self.game_state.rows
        cols = self.game_state.cols

        for r in range(rows + 1):
            for c in range(cols):
                edges.append(self._norm_edge((r, c), (r, c + 1)))

        for r in range(rows):
            for c in range(cols + 1):
                edges.append(self._norm_edge((r, c), (r + 1, c)))

        return edges

    def _iter_clues(self):
        clues = self.game_state.clues
        for r in range(self.game_state.rows):
            for c in range(self.game_state.cols):
                cell = (r, c)
                if cell in clues:
                    yield cell, clues[cell]

    def _cell_edges(self, cell: Tuple[int, int]) -> List[Move]:
        r, c = cell
        return [
            self._norm_edge((r, c), (r, c + 1)),
            self._norm_edge((r + 1, c), (r + 1, c + 1)),
            self._norm_edge((r, c), (r + 1, c)),
            self._norm_edge((r, c + 1), (r + 1, c + 1)),
        ]

    def _incident_edges(self, vertex: Tuple[int, int]) -> List[Move]:
        r, c = vertex
        rows = self.game_state.rows
        cols = self.game_state.cols
        edges: List[Move] = []

        if r > 0:
            edges.append(self._norm_edge((r - 1, c), (r, c)))
        if r < rows:
            edges.append(self._norm_edge((r, c), (r + 1, c)))
        if c > 0:
            edges.append(self._norm_edge((r, c - 1), (r, c)))
        if c < cols:
            edges.append(self._norm_edge((r, c), (r, c + 1)))

        return edges

    def _adjacent_cells(self, edge: Move) -> List[Tuple[int, int]]:
        (r1, c1), (r2, c2) = edge
        cells: List[Tuple[int, int]] = []

        if r1 == r2:
            c_min = min(c1, c2)
            if r1 > 0:
                cells.append((r1 - 1, c_min))
            if r1 < self.game_state.rows:
                cells.append((r1, c_min))
        else:
            r_min = min(r1, r2)
            c = c1
            if c > 0:
                cells.append((r_min, c - 1))
            if c < self.game_state.cols:
                cells.append((r_min, c))

        return cells

    def _vertex_degree(self, vertex: Tuple[int, int], state: Dict[str, Set[Move]]) -> int:
        degree = 0
        for edge in self._incident_edges(vertex):
            if edge in state["on"]:
                degree += 1
        return degree

    def _count_on_edges_around_cell(self, cell: Tuple[int, int], state: Dict[str, Set[Move]]) -> int:
        count = 0
        for edge in self._cell_edges(cell):
            if edge in state["on"]:
                count += 1
        return count

    def _can_set_on(self, edge: Move, state: Dict[str, Set[Move]]) -> bool:
        if edge in state["off"]:
            return False
        if edge in state["on"]:
            return True

        u, v = edge
        if self._vertex_degree(u, state) >= 2:
            return False
        if self._vertex_degree(v, state) >= 2:
            return False

        clues = self.game_state.clues
        for cell in self._adjacent_cells(edge):
            if cell in clues:
                clue = clues[cell]
                if self._count_on_edges_around_cell(cell, state) + 1 > clue:
                    return False

        # Local small loop prevention:
        # If this edge closes a loop and an adjacent clue cell would still be unmet,
        # reject the edge.
        if self._would_create_closed_loop(edge, state) and self._has_unmet_adjacent_clue_after_add(edge, state):
            return False

        return True

    def _would_create_closed_loop(self, edge: Move, state: Dict[str, Set[Move]]) -> bool:
        """
        Non-recursive local chain walk:
        in a degree<=2 partial graph, if u can already reach v by unique chain,
        adding edge (u, v) closes a loop.
        """
        u, v = edge
        if self._vertex_degree(u, state) != 1 or self._vertex_degree(v, state) != 1:
            return False

        current = u
        prev: Optional[Tuple[int, int]] = None
        visited: Set[Tuple[int, int]] = {u}
        max_steps = len(state["on"]) + 2
        steps = 0

        while steps < max_steps:
            if current == v:
                return True

            neighbors = self._on_neighbors(current, state)
            next_nodes = [node for node in neighbors if node != prev]
            if not next_nodes:
                return False

            nxt = next_nodes[0]
            prev = current
            current = nxt
            if current in visited and current != v:
                return False
            visited.add(current)
            steps += 1

        return False

    def _has_unmet_adjacent_clue_after_add(self, edge: Move, state: Dict[str, Set[Move]]) -> bool:
        clues = self.game_state.clues
        for cell in self._adjacent_cells(edge):
            if cell not in clues:
                continue
            clue = clues[cell]
            projected_on = self._count_on_edges_around_cell(cell, state) + 1
            if projected_on != clue:
                return True
        return False

    def _on_neighbors(self, vertex: Tuple[int, int], state: Dict[str, Set[Move]]) -> List[Tuple[int, int]]:
        neighbors: List[Tuple[int, int]] = []
        for edge in self._incident_edges(vertex):
            if edge not in state["on"]:
                continue
            a, b = edge
            neighbors.append(b if a == vertex else a)
        return neighbors

    def _norm_edge(self, u: Tuple[int, int], v: Tuple[int, int]) -> Move:
        return tuple(sorted((u, v)))
