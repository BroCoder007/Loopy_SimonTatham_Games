"""
Constraint Propagation + Backtracking Solver
=============================================
A CSP (Constraint Satisfaction Problem) solver for Slitherlink puzzles.

Algorithm Overview:
1. Constraint Propagation: Deduce forced edges (ON/OFF) using:
   - Clue constraints: Each numbered cell dictates how many surrounding edges are part of the loop.
   - Vertex degree constraints: Each vertex must have degree 0 or 2 in the final loop.
   Propagation repeats until no more deductions can be made.

2. Backtracking: When propagation stalls (no forced moves), pick the most constrained
   undecided edge, guess ON/OFF, propagate, and recurse. Backtrack on contradiction.

Time Complexity: O(2^E) worst case, but propagation prunes the search space massively.
Space Complexity: O(E) for edge state tracking.

DAA Concepts: Constraint Satisfaction, Backtracking, Pruning, Arc Consistency.
"""

from typing import Any, Dict, List, Optional, Set, Tuple
from logic.solvers.solver_interface import AbstractSolver, Edge, HintPayload, MoveExplanation
from logic.execution_trace import log_execution_step


# Edge states
UNKNOWN = 0
ON = 1
OFF = 2


class ConstraintPropagator:
    """
    Pure deduction engine for Slitherlink constraints.
    
    Tracks every edge in the grid as UNKNOWN, ON, or OFF.
    Applies clue constraints and vertex degree constraints repeatedly 
    until no more forced deductions exist (fixed-point iteration).
    
    This is analogous to Arc Consistency (AC-3) in CSP literature.
    """

    def __init__(self, rows: int, cols: int, clues: Dict[Tuple[int, int], int],
                 current_edges: Optional[Set] = None):
        self.rows = rows
        self.cols = cols
        self.clues = clues

        # Build the complete set of all possible edges on the grid
        self.all_edges: List[Edge] = []
        self._build_all_edges()

        # Edge state map: edge -> UNKNOWN / ON / OFF
        self.edge_state: Dict[Edge, int] = {e: UNKNOWN for e in self.all_edges}

        # Precompute which edges surround each clue cell
        self.cell_edges: Dict[Tuple[int, int], List[Edge]] = {}
        self._precompute_cell_edges()

        # Precompute which edges touch each vertex
        self.vertex_edges: Dict[Tuple[int, int], List[Edge]] = {}
        self._precompute_vertex_edges()

        # Track deduction reasons for hint explanations
        self.deduction_log: List[Dict[str, Any]] = []

        # Apply currently placed edges from the game board
        if current_edges:
            for edge in current_edges:
                canonical = tuple(sorted(edge))
                if canonical in self.edge_state:
                    self.edge_state[canonical] = ON

    def _build_all_edges(self):
        """Enumerate all horizontal and vertical edges on the grid."""
        # Horizontal edges: between (r, c) and (r, c+1)
        for r in range(self.rows + 1):
            for c in range(self.cols):
                self.all_edges.append(tuple(sorted(((r, c), (r, c + 1)))))
        # Vertical edges: between (r, c) and (r+1, c)
        for r in range(self.rows):
            for c in range(self.cols + 1):
                self.all_edges.append(tuple(sorted(((r, c), (r + 1, c)))))

    def _precompute_cell_edges(self):
        """For each cell (r, c), store its 4 surrounding edges."""
        for r in range(self.rows):
            for c in range(self.cols):
                top = tuple(sorted(((r, c), (r, c + 1))))
                bottom = tuple(sorted(((r + 1, c), (r + 1, c + 1))))
                left = tuple(sorted(((r, c), (r + 1, c))))
                right = tuple(sorted(((r, c + 1), (r + 1, c + 1))))
                self.cell_edges[(r, c)] = [top, bottom, left, right]

    def _precompute_vertex_edges(self):
        """For each vertex (r, c), store all edges touching it."""
        for r in range(self.rows + 1):
            for c in range(self.cols + 1):
                edges = []
                # Up
                if r > 0:
                    edges.append(tuple(sorted(((r - 1, c), (r, c)))))
                # Down
                if r < self.rows:
                    edges.append(tuple(sorted(((r, c), (r + 1, c)))))
                # Left
                if c > 0:
                    edges.append(tuple(sorted(((r, c - 1), (r, c)))))
                # Right
                if c < self.cols:
                    edges.append(tuple(sorted(((r, c), (r, c + 1)))))
                self.vertex_edges[(r, c)] = edges

    def propagate(self) -> bool:
        """
        Run constraint propagation until no more deductions can be made.
        
        Returns:
            True if propagation completed without contradiction.
            False if a contradiction was detected (unsolvable from current state).
            
        Algorithm:
            Repeat until fixed-point (no changes in a full pass):
                1. For each clue cell, check if edges can be forced.
                2. For each vertex, check degree constraints.
            If any constraint is violated, return False (contradiction).
        """
        changed = True
        while changed:
            changed = False

            # --- Clue Constraints ---
            for cell, clue_val in self.clues.items():
                if cell not in self.cell_edges:
                    continue

                edges = self.cell_edges[cell]
                on_count = sum(1 for e in edges if self.edge_state[e] == ON)
                off_count = sum(1 for e in edges if self.edge_state[e] == OFF)
                unknown_edges = [e for e in edges if self.edge_state[e] == UNKNOWN]
                unknown_count = len(unknown_edges)

                # Contradiction: more edges ON than clue allows
                if on_count > clue_val:
                    return False

                # Contradiction: not enough undecided edges left to reach clue
                if on_count + unknown_count < clue_val:
                    return False

                # Rule 1: All remaining unknowns must be ON
                # (we need exactly unknown_count more edges, and that's all that's left)
                if clue_val - on_count == unknown_count and unknown_count > 0:
                    for e in unknown_edges:
                        self.edge_state[e] = ON
                        self.deduction_log.append({
                            "edge": e,
                            "state": "ON",
                            "reason": f"Cell {cell} needs {clue_val - on_count} more edge(s) "
                                      f"and only {unknown_count} unknown remain → forced ON"
                        })
                        changed = True

                # Rule 2: Clue is satisfied, force all remaining unknowns OFF
                if on_count == clue_val and unknown_count > 0:
                    for e in unknown_edges:
                        self.edge_state[e] = OFF
                        self.deduction_log.append({
                            "edge": e,
                            "state": "OFF",
                            "reason": f"Cell {cell} already has {clue_val} edge(s) "
                                      f"(clue satisfied) → remaining forced OFF"
                        })
                        changed = True

            # --- Vertex Degree Constraints ---
            for vertex, edges in self.vertex_edges.items():
                # Filter to edges that actually exist in our edge set
                valid_edges = [e for e in edges if e in self.edge_state]
                on_count = sum(1 for e in valid_edges if self.edge_state[e] == ON)
                unknown_edges = [e for e in valid_edges if self.edge_state[e] == UNKNOWN]
                unknown_count = len(unknown_edges)

                # Contradiction: vertex already has degree > 2
                if on_count > 2:
                    return False

                # Contradiction: vertex has degree 1 and no unknowns left
                # (can't complete the path through this vertex)
                if on_count == 1 and unknown_count == 0:
                    return False

                # Rule 3: Vertex at degree 2 → force all remaining unknowns OFF
                if on_count == 2 and unknown_count > 0:
                    for e in unknown_edges:
                        self.edge_state[e] = OFF
                        self.deduction_log.append({
                            "edge": e,
                            "state": "OFF",
                            "reason": f"Vertex {vertex} already has degree 2 "
                                      f"→ no more edges allowed, forced OFF"
                        })
                        changed = True

                # Rule 4: Vertex at degree 1 with exactly 1 unknown → force it ON
                # (the loop must continue through this vertex)
                if on_count == 1 and unknown_count == 1:
                    e = unknown_edges[0]
                    self.edge_state[e] = ON
                    self.deduction_log.append({
                        "edge": e,
                        "state": "ON",
                        "reason": f"Vertex {vertex} has degree 1 and only 1 unknown edge "
                                  f"→ loop must continue, forced ON"
                    })
                    changed = True

        return True  # No contradiction

    def get_forced_edges(self) -> List[Dict[str, Any]]:
        """Return all edges that were deduced as ON or OFF (not from the input)."""
        return list(self.deduction_log)

    def get_unknown_edges(self) -> List[Edge]:
        """Return edges still in UNKNOWN state."""
        return [e for e, state in self.edge_state.items() if state == UNKNOWN]

    def is_solved(self) -> bool:
        """Check if no UNKNOWN edges remain and no contradictions exist."""
        return all(state != UNKNOWN for state in self.edge_state.values())

    def copy(self) -> "ConstraintPropagator":
        """Create a deep copy for backtracking branches."""
        new = ConstraintPropagator.__new__(ConstraintPropagator)
        new.rows = self.rows
        new.cols = self.cols
        new.clues = self.clues  # Immutable during solve
        new.all_edges = self.all_edges  # Immutable
        new.cell_edges = self.cell_edges  # Immutable
        new.vertex_edges = self.vertex_edges  # Immutable
        new.edge_state = dict(self.edge_state)  # Mutable copy
        new.deduction_log = list(self.deduction_log)  # Mutable copy
        return new

    def get_most_constrained_edge(self) -> Optional[Edge]:
        """
        Pick the unknown edge that is most constrained (adjacent to the most 
        clue cells or vertices with high degree). This heuristic (MRV — Minimum
        Remaining Values) reduces backtracking branches.
        """
        unknown = self.get_unknown_edges()
        if not unknown:
            return None

        def constraint_score(edge: Edge) -> int:
            """Higher score = more constrained = should be tried first."""
            score = 0
            u, v = edge

            # Score from adjacent clue cells
            for cell, edges in self.cell_edges.items():
                if edge in edges and cell in self.clues:
                    cell_unknown = sum(1 for e in edges if self.edge_state[e] == UNKNOWN)
                    # Fewer unknowns around a clue = more constrained
                    score += (5 - cell_unknown)

            # Score from vertex constraints
            for vertex in [u, v]:
                if vertex in self.vertex_edges:
                    v_edges = self.vertex_edges[vertex]
                    on_count = sum(1 for e in v_edges 
                                   if e in self.edge_state and self.edge_state[e] == ON)
                    if on_count > 0:
                        score += on_count * 3  # Edges near active paths are more constrained

            return score

        return max(unknown, key=constraint_score)


class ConstraintBacktrackingSolver(AbstractSolver):
    """
    Solver that combines Constraint Propagation with Recursive Backtracking.
    
    Implements the AbstractSolver interface so it can plug directly into
    the existing strategy controller and game state infrastructure.
    
    Algorithm:
        1. Propagate constraints (deduce all forced edges).
        2. If solved → return solution.
        3. If contradiction → backtrack (return failure).
        4. Pick most constrained unknown edge (MRV heuristic).
        5. Branch: try ON, then OFF. Recurse on each.
        6. If both branches fail → backtrack.
    
    Complexity:
        Worst case: O(2^E) where E = number of edges.
        Practical: Much faster due to propagation pruning.
    """

    def __init__(self, game_state: Any):
        self.game_state = game_state
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.clues = game_state.clues

        # Tracking stats
        self.nodes_explored = 0
        self.max_recursion_depth = 0
        self.propagation_deductions = 0

        # For hint/explanation output
        self._last_hint = None
        self._last_explanation = ""
        self._last_move_metadata = None

        # Limits to prevent runaway on large boards
        self.max_nodes = 500_000
        self.max_depth = 100

    def solve(self, board: Any = None) -> Optional[Edge]:
        """
        Find the next best move by running constraint propagation + backtracking.
        
        Returns the first forced/deduced edge that is not already on the board,
        or None if no moves can be found.
        """
        self.nodes_explored = 0
        self.max_recursion_depth = 0

        current_edges = self.game_state.graph.edges
        propagator = ConstraintPropagator(
            self.rows, self.cols, self.clues, current_edges
        )

        # Run propagation
        valid = propagator.propagate()
        if not valid:
            self._last_explanation = "Board state has a contradiction — no solution possible."
            return None

        # Check if propagation found any new ON edges
        for entry in propagator.deduction_log:
            edge = entry["edge"]
            if entry["state"] == "ON" and edge not in current_edges:
                self._last_explanation = entry["reason"]
                self.propagation_deductions += 1

                # Log to execution trace
                log_execution_step(
                    strategy_name="Constraint Propagation",
                    move=edge,
                    explanation=entry["reason"],
                    recursion_depth=0
                )
                return edge

        # Propagation didn't find a new ON edge — try backtracking
        solution_state = self._backtrack_solve(propagator, depth=0)
        if solution_state is not None:
            # Find the first edge in the solution that isn't already placed
            for edge, state in solution_state.edge_state.items():
                if state == ON and edge not in current_edges:
                    self._last_explanation = (
                        f"Backtracking search (explored {self.nodes_explored} nodes, "
                        f"max depth {self.max_recursion_depth}) found this edge is required."
                    )
                    log_execution_step(
                        strategy_name="Constraint Backtracking",
                        move=edge,
                        explanation=self._last_explanation,
                        recursion_depth=self.max_recursion_depth,
                        dp_state_count=self.nodes_explored
                    )
                    return edge

        self._last_explanation = "No valid moves found by constraint propagation + backtracking."
        return None

    def _backtrack_solve(self, propagator: ConstraintPropagator, 
                         depth: int) -> Optional[ConstraintPropagator]:
        """
        Recursive backtracking with constraint propagation at each node.
        
        Args:
            propagator: Current constraint state
            depth: Current recursion depth
            
        Returns:
            A solved ConstraintPropagator if solution exists, else None.
        """
        self.nodes_explored += 1
        self.max_recursion_depth = max(self.max_recursion_depth, depth)

        # Safety limits
        if self.nodes_explored > self.max_nodes or depth > self.max_depth:
            return None

        # Base case: all edges decided
        if propagator.is_solved():
            return propagator

        # Pick the most constrained unknown edge (MRV heuristic)
        edge = propagator.get_most_constrained_edge()
        if edge is None:
            return propagator  # All decided

        # Branch 1: Try setting edge to ON
        branch_on = propagator.copy()
        branch_on.edge_state[edge] = ON
        branch_on.deduction_log.append({
            "edge": edge, "state": "ON",
            "reason": f"Backtracking guess at depth {depth}: edge {edge} → ON"
        })
        if branch_on.propagate():
            result = self._backtrack_solve(branch_on, depth + 1)
            if result is not None:
                return result

        # Branch 2: Try setting edge to OFF
        branch_off = propagator.copy()
        branch_off.edge_state[edge] = OFF
        branch_off.deduction_log.append({
            "edge": edge, "state": "OFF",
            "reason": f"Backtracking guess at depth {depth}: edge {edge} → OFF"
        })
        if branch_off.propagate():
            result = self._backtrack_solve(branch_off, depth + 1)
            if result is not None:
                return result

        # Both branches failed — backtrack
        return None

    def decide_move(self) -> Tuple[List[Tuple[Edge, int]], Optional[Edge]]:
        """
        Return candidate moves with confidence scores and the best move.
        
        Forced moves (from propagation) get confidence 100.
        Backtracking-found moves get confidence 80.
        """
        current_edges = self.game_state.graph.edges
        propagator = ConstraintPropagator(
            self.rows, self.cols, self.clues, current_edges
        )

        valid = propagator.propagate()
        if not valid:
            return [], None

        candidates = []
        best_move = None

        # Collect all forced ON edges
        for entry in propagator.deduction_log:
            edge = entry["edge"]
            if entry["state"] == "ON" and edge not in current_edges:
                candidates.append((edge, 100))  # Max confidence for forced moves
                if best_move is None:
                    best_move = edge

        if best_move:
            self._last_explanation = propagator.deduction_log[0]["reason"] if propagator.deduction_log else ""
            return candidates, best_move

        # No forced moves — try backtracking for one move
        best_move = self.solve()
        if best_move:
            candidates.append((best_move, 80))

        return candidates, best_move

    def generate_hint(self, board: Any = None) -> HintPayload:
        """
        Generate a human-friendly hint with explanation of WHY the edge is forced.
        """
        current_edges = self.game_state.graph.edges
        propagator = ConstraintPropagator(
            self.rows, self.cols, self.clues, current_edges
        )

        valid = propagator.propagate()

        if not valid:
            return {
                "move": None,
                "strategy": "Constraint Propagation",
                "explanation": "⚠ Board contradiction detected! Your current edges "
                               "violate puzzle constraints. Try undoing recent moves.",
                "is_error": True
            }

        # Find a forced ON edge
        for entry in propagator.deduction_log:
            edge = entry["edge"]
            if entry["state"] == "ON" and edge not in current_edges:
                self._last_hint = entry
                return {
                    "move": edge,
                    "strategy": "Constraint Propagation",
                    "explanation": f"🔍 {entry['reason']}",
                    "is_error": False
                }

        # No propagation hints — try backtracking
        self.nodes_explored = 0
        self.max_recursion_depth = 0
        move = self.solve()
        if move:
            return {
                "move": move,
                "strategy": "Constraint Backtracking",
                "explanation": f"🧠 Backtracking search (explored {self.nodes_explored} states) "
                               f"determined this edge must be part of the loop.",
                "is_error": False
            }

        return {
            "move": None,
            "strategy": "Constraint Propagation",
            "explanation": "No hints available — the solver could not find a forced move.",
            "is_error": False
        }

    def explain_last_move(self) -> str:
        """Return text explanation of the last move made by this solver."""
        return self._last_explanation or "No move explanation available."

    def get_last_move_explanation(self) -> Optional[MoveExplanation]:
        """Return structured explanation for the last move."""
        if not self._last_explanation:
            return None
        return MoveExplanation(
            mode="Constraint Propagation + Backtracking",
            scope="Global",
            decision_summary=self._last_explanation,
            reasoning_data={
                "nodes_explored": self.nodes_explored,
                "max_depth": self.max_recursion_depth,
                "propagation_deductions": self.propagation_deductions
            }
        )
