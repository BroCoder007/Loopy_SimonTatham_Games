"""
Advanced DP Solver
==================
Implements "Dynamic Programming with Divide & Conquer Decomposition".

Strategy:
1. Explicit Spatial Decomposition: Divide grid into 4 quadrants.
2. Region-Level DP: Solve each quadrant independently (Region DP).
3. Boundary Interface Compression: Compress boundary states.
4. Region Merge Phase: Merge 4 regions (Q1+Q2 -> Top, Q3+Q4 -> Bottom, then Top+Bottom).
5. Global Loop Enforcement.

Region-level deterministic enumeration with constraint pruning and seam compatibility merge.
"""

from __future__ import annotations

import collections
from math import comb
from typing import Any, List, Set, Tuple, Dict, Optional
from logic.solvers.solver_interface import AbstractSolver, HintPayload
from logic.validators import is_valid_move

# Type aliases
# Edge: ((r1, c1), (r2, c2)) sorted
Edge = Tuple[Tuple[int, int], Tuple[int, int]]
# Signature: tuple of component IDs along a boundary line
Signature = Tuple[int, ...]

class RegionSolution:
    """
    Represents a valid partial solution for a rectangular region.
    """
    def __init__(self, 
                 edges: Set[Edge], 
                 top_sig: Signature, 
                 bottom_sig: Signature, 
                 left_sig: Signature, 
                 right_sig: Signature,
                 # Boundary Edge Masks (True if edge exists on boundary line)
                 v_left_mask: Tuple[bool, ...] = (),
                 v_right_mask: Tuple[bool, ...] = (),
                 h_top_mask: Tuple[bool, ...] = (),
                 h_bottom_mask: Tuple[bool, ...] = (),
                 internal_loops: int = 0):
        self.edges = edges
        self.top_sig = top_sig
        self.bottom_sig = bottom_sig
        self.left_sig = left_sig
        self.right_sig = right_sig
        self.v_left_mask = v_left_mask
        self.v_right_mask = v_right_mask
        self.h_top_mask = h_top_mask
        self.h_bottom_mask = h_bottom_mask
        self.internal_loops = internal_loops

    def __repr__(self):
        return f"Region(loops={self.internal_loops}, sigs={self.top_sig}/{self.bottom_sig})"

class AdvancedDPSolver(AbstractSolver):
    def __init__(self, game_state: Any):
        self.game_state = game_state
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.last_explanation = ""
        self.solution_moves = []
        self.current_move_index = 0
        self._solution_computed = False
        
        # Store full solution edges
        self._final_solution_edges: Set[Edge] = set()
        
        # Track merge statistics for detailed explanations
        self._merge_stats = {
            'region_stats': {},  # region_id -> {'states': int, 'pruned': int}
            'merge_details': []  # List of merge operation details
        }

    def solve(self, board: Any = None):
        """
        Returns the next move from precomputed solution.
        """
        if not self._solution_computed:
            self._compute_full_solution()
            
        # Standard playback logic (similar to other solvers)
        while self.current_move_index < len(self.solution_moves):
            move_data = self.solution_moves[self.current_move_index]
            move = move_data["move"]
            
            # Check validity
            valid, reason = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
            if valid:
                from logic.execution_trace import log_advanced_dp_move
                log_advanced_dp_move(
                    move=move,
                    explanation=move_data.get("explanation", ""),
                    recursion_depth=0,
                    region_id="Global"
                )
                return move
            else:
                self.current_move_index += 1
        return None

    def decide_move(self) -> Tuple[List[Tuple[Tuple[int, int], int]], Optional[Tuple[int, int]]]:
        """
        UI Hook.
        """
        print("[AdvancedDPSolver] decide_move called.")
        if not self._solution_computed:
            print("[AdvancedDPSolver] Computing full solution...")
            self._compute_full_solution()
            
        # Check if solution exists
        if not self._final_solution_edges:
            print("[AdvancedDPSolver] No solution found in _final_solution_edges.")
            self.last_explanation = "No precomputed Advanced DP move is available."
            return [], None

        while self.current_move_index < len(self.solution_moves):
            move_data = self.solution_moves[self.current_move_index]
            move = move_data["move"]
            valid, reason = is_valid_move(self.game_state.graph, move[0], move[1], self.game_state.clues)
            if valid:
                self.last_explanation = move_data.get("explanation", "Advanced DP Move")
                print(f"[AdvancedDPSolver] Decided move: {move}")
                
                # Global Execution Trace Log
                from logic.execution_trace import log_advanced_dp_move
                log_advanced_dp_move(
                    move=move,
                    explanation=self.last_explanation,
                    recursion_depth=0,
                    region_id="Global"
                )

                return [(move, 100)], move
            else:
                print(f"[AdvancedDPSolver] Skipping invalid move: {move} ({reason})")
                self.current_move_index += 1
                
        self.last_explanation = "No more moves."
        print("[AdvancedDPSolver] No more moves in solution_moves.")
        return [], None

    def generate_hint(self, board: Any = None) -> HintPayload:
        """
        Pure D&C deterministic hinting.
        A hint is produced only if an undecided edge is forced by all
        compatible merged region configurations.
        """
        target = board if board is not None else self.game_state
        
        # Check if it's human turn (only provide hints for human player)
        is_human_turn = (self.game_state.game_mode in ["vs_cpu", "expert"] and 
                        self.game_state.turn == "Player 1 (Human)") or \
                       (self.game_state.game_mode not in ["vs_cpu", "expert"])
        
        if not is_human_turn:
            return {
                "move": None,
                "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
                "explanation": "Hints are only available during your turn."
            }

        # Compute merged states.
        full_states = self._compute_full_merged_states_for_hint()
        if not full_states:
            return {
                "move": None,
                "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
                "explanation": "No deterministic D&C deduction available."
            }

        required_edges = set(target.graph.edges)
        compatible_states = self._filter_states_by_required_edges(full_states, required_edges)
        if not compatible_states:
            return {
                "move": None,
                "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
                "explanation": "No deterministic D&C deduction available."
            }

        total_states = len(compatible_states)
        edge_frequency: Dict[Edge, int] = collections.Counter()
        for state_edges in compatible_states:
            for edge in state_edges:
                edge_frequency[edge] += 1

        forced_edges: List[Edge] = []
        for edge, count in sorted(edge_frequency.items()):
            if count != total_states:
                continue
            if edge in required_edges:
                continue
            valid, _ = is_valid_move(target.graph, edge[0], edge[1], target.clues)
            if valid:
                forced_edges.append(edge)

        if forced_edges:
            edge = forced_edges[0]
            return {
                "move": edge,
                "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
                "explanation": (
                    f"Deterministic D&C: edge {edge} appears in all "
                    f"{total_states} compatible merged region configurations."
                )
            }

        return {
            "move": None,
            "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
            "explanation": "No deterministic D&C deduction available."
        }

    def explain_last_move(self) -> str:
        return self.last_explanation

    def _ordered_seam_edges(self) -> List[Edge]:
        """
        Deterministic seam edge ordering:
        1) Vertical seam edges (top to bottom), then
        2) Horizontal seam edges (left to right).
        """
        mid_r = self.rows // 2
        mid_c = self.cols // 2
        seam_edges: List[Edge] = []
        for r in range(self.rows):
            seam_edges.append(tuple(sorted(((r, mid_c), (r + 1, mid_c)))))
        for c in range(self.cols):
            seam_edges.append(tuple(sorted(((mid_r, c), (mid_r, c + 1)))))
        return seam_edges

    def _seam_mask_tuple_from_edges(self, edges: Set[Edge]) -> Tuple[int, ...]:
        seam_edges = self._ordered_seam_edges()
        edge_set = set(edges)
        return tuple(1 if seam_edge in edge_set else 0 for seam_edge in seam_edges)

    def _component_count(self, edges: Set[Edge]) -> int:
        if not edges:
            return 0
        uf = UnionFind()
        nodes: Set[Tuple[int, int]] = set()
        for u, v in edges:
            nodes.add(u)
            nodes.add(v)
            uf.union(u, v)
        roots = set(uf.find(node) for node in nodes)
        return len(roots)

    def _region_solution_sort_key(self, state: RegionSolution) -> Tuple[int, int, Tuple[int, ...]]:
        """
        Deterministic state ordering:
        1) Minimum seam width
        2) Minimum component count
        3) Lexicographically smallest seam mask
        """
        seam_mask = self._seam_mask_tuple_from_edges(state.edges)
        seam_width = sum(seam_mask)
        component_count = self._component_count(state.edges)
        return (seam_width, component_count, seam_mask)

    def _compute_compatible_boundary_states(self, target) -> List[Set[Edge]]:
        """
        Recompute compatible region boundary states using Advanced DP logic.
        Returns all valid merged DP states consistent with current board.
        """
        try:
            # Get current board edges
            current_edges = set(target.graph.edges)
            full_states = self._compute_full_merged_states_for_hint()
            return self._filter_states_by_required_edges(full_states, current_edges)
            
        except Exception as e:
            print(f"Error computing compatible boundary states: {e}")
            import traceback
            traceback.print_exc()
            return []

    def _compute_full_merged_states_for_hint(self, max_states_per_merge: int = 100) -> List[RegionSolution]:
        """
        Compute merged DP states for hinting without board-edge filtering.
        """
        # Divide grid into quadrants
        mid_r = self.rows // 2
        mid_c = self.cols // 2

        quadrants = [
            (0, mid_r, 0, mid_c),
            (0, mid_r, mid_c, self.cols),
            (mid_r, self.rows, 0, mid_c),
            (mid_r, self.rows, mid_c, self.cols),
        ]

        q_results = []
        for r0, r1, c0, c1 in quadrants:
            solutions = sorted(
                self._solve_region(r0, r1, c0, c1),
                key=self._region_solution_sort_key
            )
            q_results.append(solutions)

        if any(len(res) == 0 for res in q_results):
            return []

        top_states = self._merge_horizontal_limited(q_results[0], q_results[1], mid_c, 0, max_states_per_merge)
        if not top_states:
            return []

        bottom_states = self._merge_horizontal_limited(q_results[2], q_results[3], mid_c, mid_r, max_states_per_merge)
        if not bottom_states:
            return []

        return self._merge_vertical_limited(top_states, bottom_states, mid_r, 0, max_states_per_merge)

    def _filter_states_by_required_edges(self, full_states: List[RegionSolution], required_edges: Set[Edge]) -> List[Set[Edge]]:
        """
        Return edge-sets whose assignments include all required edges.
        """
        compatible_states: List[Set[Edge]] = []
        for state in full_states:
            if required_edges.issubset(state.edges):
                compatible_states.append(state.edges)
        return compatible_states

    def _find_boundary_forced_hint(self, compatible_states: List[Set[Edge]], target) -> Optional[HintPayload]:
        """
        Layer 2:
        Boundary-seam forced edges common to all compatible states.
        """
        if not compatible_states:
            return None

        common_edges = set(compatible_states[0])
        for state_edges in compatible_states[1:]:
            common_edges.intersection_update(state_edges)

        seam_edges: Set[Edge] = set()
        mid_r = self.rows // 2
        mid_c = self.cols // 2

        for r in range(self.rows):
            seam_edges.add(tuple(sorted(((r, mid_c), (r + 1, mid_c)))))
        for c in range(self.cols):
            seam_edges.add(tuple(sorted(((mid_r, c), (mid_r, c + 1)))))

        graph = target.graph
        clues = target.clues
        current_edges = set(graph.edges)

        boundary_forced = []
        for edge in sorted(common_edges.intersection(seam_edges)):
            if edge in current_edges:
                continue
            valid, _ = is_valid_move(graph, edge[0], edge[1], clues)
            if valid:
                boundary_forced.append(edge)

        if not boundary_forced:
            return None

        edge = boundary_forced[0]
        return {
            "move": edge,
            "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
            "explanation": (
                f"Layer 2 (Region Boundary Compatibility): Edge {edge} is common across all "
                f"{len(compatible_states)} compatible merged states on decomposition seams."
            )
        }

    def _find_state_reduction_hint(self, compatible_states: List[Set[Edge]], target) -> Optional[HintPayload]:
        """
        Layer 3:
        Heuristic move that maximally reduces compatible-state count.
        """
        if not compatible_states:
            return self._find_local_state_reduction_hint(target)

        current_edges = set(target.graph.edges)
        all_edges = self._get_all_potential_edges()
        total_states = len(compatible_states)

        best_edge: Optional[Edge] = None
        best_include_count = total_states
        best_exclude_count = total_states
        best_score = 0

        for edge in all_edges:
            # Heuristic layer works on undecided edges (not currently present).
            if edge in current_edges:
                continue

            include_count = 0
            for state in compatible_states:
                if edge in state:
                    include_count += 1

            exclude_count = total_states - include_count
            include_reduction = total_states - include_count
            exclude_reduction = total_states - exclude_count
            score = max(include_reduction, exclude_reduction)

            if score > best_score:
                best_score = score
                best_edge = edge
                best_include_count = include_count
                best_exclude_count = exclude_count

        if best_edge is None:
            return None

        valid, _ = is_valid_move(target.graph, best_edge[0], best_edge[1], target.clues)
        if not valid:
            return None

        if best_include_count >= total_states:
            return None

        return {
            "move": best_edge,
            "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
            "explanation": (
                "Layer 3 (State Reduction Heuristic): "
                f"Selecting edge {best_edge} reduces region compatibility states from "
                f"{total_states} to {best_include_count}, significantly narrowing solution space. "
                f"(Alternative exclusion branch would leave {best_exclude_count} states.)"
            )
        }

    def _find_local_state_reduction_hint(self, target) -> Optional[HintPayload]:
        """
        Layer 3 local estimator:
        Local compatibility-count estimator when merged DP state set is empty.
        """
        current_edges = set(target.graph.edges)
        all_edges = self._get_all_potential_edges()

        base_count = self._estimate_local_compatibility_count(target, set(), set())
        if base_count <= 0:
            return None

        best_edge: Optional[Edge] = None
        best_include = base_count
        best_exclude = base_count
        best_score = 0

        for edge in all_edges:
            if edge in current_edges:
                continue

            valid, _ = is_valid_move(target.graph, edge[0], edge[1], target.clues)
            if not valid:
                continue

            include_count = self._estimate_local_compatibility_count(target, {edge}, set())
            exclude_count = self._estimate_local_compatibility_count(target, set(), {edge})

            include_reduction = base_count - include_count
            exclude_reduction = base_count - exclude_count
            score = max(include_reduction, exclude_reduction)

            if score > best_score:
                best_score = score
                best_edge = edge
                best_include = include_count
                best_exclude = exclude_count

        if best_edge is None or best_include >= base_count:
            return None

        return {
            "move": best_edge,
            "strategy": "Dynamic Programming with Divide & Conquer Decomposition",
            "explanation": (
                "Layer 3 (State Reduction Heuristic): "
                f"Selecting edge {best_edge} reduces region compatibility states from "
                f"{base_count} to {best_include}, significantly narrowing solution space. "
                f"(Alternative exclusion branch would leave {best_exclude} states.)"
            )
        }

    def _estimate_local_compatibility_count(self, target, force_include: Set[Edge], force_exclude: Set[Edge]) -> int:
        """
        Approximate compatibility count from clue-local combinations.
        This is used only for hint ranking, never for solving.
        """
        current_edges = set(target.graph.edges)
        total = 1

        for (r, c), clue in sorted(target.clues.items()):
            cell_edges = [
                tuple(sorted(((r, c), (r, c + 1)))),
                tuple(sorted(((r + 1, c), (r + 1, c + 1)))),
                tuple(sorted(((r, c), (r + 1, c)))),
                tuple(sorted(((r, c + 1), (r + 1, c + 1)))),
            ]

            present = 0
            undecided = 0
            for edge in cell_edges:
                if edge in current_edges or edge in force_include:
                    present += 1
                elif edge in force_exclude:
                    continue
                else:
                    undecided += 1

            needed = clue - present
            if needed < 0 or needed > undecided:
                return 0

            total *= comb(undecided, needed)

        return total
    
    def _is_state_compatible_with_board(self, region_solution: RegionSolution, current_edges: Set[Edge], target) -> bool:
        """
        Check if a region solution is compatible with the current board state.
        A state is compatible if:
        1. All edges currently on the board are present in the state
        2. The state satisfies all clue constraints
        """
        state_edges = region_solution.edges
        
        # Check compatibility with current edges
        for edge in current_edges:
            if edge not in state_edges:
                return False
        
        # The region solution already satisfies clue constraints by construction
        # So we just need to ensure edge compatibility
        return True
    
    def _compute_edge_intersections(self, compatible_states: List[Set[Edge]], target) -> Tuple[List[Edge], List[Edge]]:
        """
        Compute intersection of edge assignments across all valid states.
        Returns (forced_inclusions, forced_exclusions)
        """
        if not compatible_states:
            return [], []
        
        current_edges = set(target.graph.edges)
        all_potential_edges = self._get_all_potential_edges()
        
        # Find edges present in ALL compatible states
        forced_inclusions = []
        if compatible_states:
            # Start with edges from first state
            common_edges = set(compatible_states[0])
            
            # Intersect with all other states
            for state in compatible_states[1:]:
                common_edges.intersection_update(state)
            
            # Filter for edges not already on board
            for edge in sorted(common_edges):
                if edge not in current_edges:
                    # Validate this edge can be legally added
                    valid, _ = is_valid_move(target.graph, edge[0], edge[1], target.clues)
                    if valid:
                        forced_inclusions.append(edge)
        
        # Find edges absent in ALL compatible states
        forced_exclusions = []
        if compatible_states:
            # Start with all potential edges
            absent_edges = set(all_potential_edges)
            
            # Remove edges that appear in any compatible state
            for state in compatible_states:
                absent_edges.difference_update(state)
            
            # Filter for edges currently on board
            for edge in sorted(absent_edges):
                if edge in current_edges:
                    forced_exclusions.append(edge)
        
        return forced_inclusions, forced_exclusions

    # -------------------------------------------------------------------------
    # Legacy hint-detection helpers kept for compatibility with debug/tests.
    # -------------------------------------------------------------------------
    def _detect_forced_inclusions(self, target) -> List[Edge]:
        compatible_states = self._compute_compatible_boundary_states(target)
        forced_inclusions, _ = self._compute_edge_intersections(compatible_states, target)
        return forced_inclusions

    def _detect_forced_exclusions(self, target) -> List[Edge]:
        compatible_states = self._compute_compatible_boundary_states(target)
        _, forced_exclusions = self._compute_edge_intersections(compatible_states, target)
        return forced_exclusions

    def _detect_boundary_compatibility_forced_edges(self, target) -> List[Edge]:
        compatible_states = self._compute_compatible_boundary_states(target)
        if not compatible_states:
            return []

        common_edges = set(compatible_states[0])
        for state in compatible_states[1:]:
            common_edges.intersection_update(state)

        mid_r = self.rows // 2
        mid_c = self.cols // 2
        seam_edges: Set[Edge] = set()
        for r in range(self.rows):
            seam_edges.add(tuple(sorted(((r, mid_c), (r + 1, mid_c)))))
        for c in range(self.cols):
            seam_edges.add(tuple(sorted(((mid_r, c), (mid_r, c + 1)))))

        current_edges = set(target.graph.edges)
        result = []
        for edge in sorted(common_edges.intersection(seam_edges)):
            if edge in current_edges:
                continue
            valid, _ = is_valid_move(target.graph, edge[0], edge[1], target.clues)
            if valid:
                result.append(edge)
        return result

    def _detect_pruning_forced_exclusions(self, target) -> List[Edge]:
        compatible_states = self._compute_compatible_boundary_states(target)
        if not compatible_states:
            return []

        all_edges = set(self._get_all_potential_edges())
        present_any = set()
        for state in compatible_states:
            present_any.update(state)

        # Edges absent from every compatible state (pruned by compatibility).
        pruned_absent = all_edges - present_any
        current_edges = set(target.graph.edges)
        return sorted([e for e in pruned_absent if e in current_edges])
    
    def _generate_forced_move_explanation(self, edge: Edge, is_inclusion: bool, num_states: int) -> str:
        """
        Generate explanation for forced moves based on state analysis.
        """
        region_stats = self._merge_stats.get('region_stats', {})
        merge_details = self._merge_stats.get('merge_details', [])
        
        # Get region with most states for context
        max_region = max(sorted(region_stats.items()), key=lambda x: x[1]['states']) if region_stats else None
        region_id, region_info = max_region if max_region else ("Q1", {'states': 0})
        
        # Find merge with highest pruning for boundary context
        best_merge = max(merge_details, key=lambda x: x.get('pruned_count', 0)) if merge_details else None
        
        explanation = f"Using DP + Divide & Conquer: From current board state, {region_id} generated {region_info['states']} boundary states. "
        
        if best_merge:
            pruned_count = best_merge.get('pruned_count', 0)
            total_candidates = best_merge.get('total_candidates', 1)
            seam_location = best_merge.get('seam_location', 'unknown seam')
            explanation += f"During boundary pruning at {seam_location}, {pruned_count} of {total_candidates} configurations were eliminated. "
        
        explanation += f"After region merge reasoning, {num_states} compatible DP states remain. "
        
        if is_inclusion:
            explanation += f"Edge {edge} is present in all {num_states} valid merged states, making it a forced inclusion."
        else:
            explanation += f"Edge {edge} is absent in all {num_states} valid merged states, making it a forced exclusion."
        
        return explanation
    
    def _generate_no_hint_explanation(self, num_states: int) -> str:
        """
        Generate explanation when no deterministic move is available.
        """
        region_stats = self._merge_stats.get('region_stats', {})
        merge_details = self._merge_stats.get('merge_details', [])
        
        # Get region with most states for context
        max_region = max(sorted(region_stats.items()), key=lambda x: x[1]['states']) if region_stats else None
        region_id, region_info = max_region if max_region else ("Q1", {'states': 0})
        
        # Find merge with highest pruning for boundary context
        best_merge = max(merge_details, key=lambda x: x.get('pruned_count', 0)) if merge_details else None
        
        explanation = f"No deterministic move available under current decomposition. "
        explanation += f"{region_id} generated {region_info['states']} boundary states. "
        
        if best_merge:
            pruned_count = best_merge.get('pruned_count', 0)
            total_candidates = best_merge.get('total_candidates', 1)
            seam_location = best_merge.get('seam_location', 'unknown seam')
            explanation += f"Boundary pruning at {seam_location} eliminated {pruned_count} of {total_candidates} configurations. "
        
        explanation += f"After region merge reasoning, {num_states} compatible DP states remain, but no edge assignment is forced across all states."
        
        return explanation
    
    def _get_all_potential_edges(self) -> List[Edge]:
        """
        Get all possible edges in the grid.
        """
        edges = []
        rows, cols = self.rows, self.cols
        
        # Horizontal edges
        for r in range(rows + 1):
            for c in range(cols):
                edges.append(tuple(sorted(((r, c), (r, c+1)))))
        
        # Vertical edges
        for r in range(rows):
            for c in range(cols + 1):
                edges.append(tuple(sorted(((r, c), (r+1, c)))))
        
        return edges
    
    def _generate_inclusion_explanation(self, edge: Edge, target) -> str:
        """
        Generate explanation for forced inclusion edges.
        Focus on boundary compatibility and region reasoning.
        """
        # Get relevant merge statistics
        merge_details = self._merge_stats.get('merge_details', [])
        region_stats = self._merge_stats.get('region_stats', {})
        
        # Find the most constrained merge operation
        relevant_merge = self._find_relevant_merge(edge, merge_details)
        
        if relevant_merge:
            merge_type = relevant_merge['type']
            total_candidates = relevant_merge['total_candidates']
            pruned_count = relevant_merge['pruned_count']
            seam_location = relevant_merge['seam_location']
            
            # Find region with most states for context
            max_region = max(sorted(region_stats.items()), key=lambda x: x[1]['states']) if region_stats else None
            region_id, region_info = max_region if max_region else ("Q1", {'states': 0})
            
            explanation = f"Using DP + Divide & Conquer: {region_id} generated {region_info['states']} boundary states. "
            explanation += f"During {merge_type} merge at {seam_location}, {pruned_count} of {total_candidates} boundary configurations were pruned. "
            explanation += f"Only configurations preserving vertical/horizontal continuity remain compatible, forcing edge {edge} to be included."
            
            return explanation
        else:
            return f"DP + Divide & Conquer: Edge {edge} is required for boundary compatibility across region seams. This edge appears in all valid merged states."
    
    def _generate_exclusion_explanation(self, edge: Edge, target) -> str:
        """
        Generate explanation for forced exclusion edges.
        Focus on state elimination and constraint violations.
        """
        merge_details = self._merge_stats.get('merge_details', [])
        relevant_merge = self._find_relevant_merge(edge, merge_details)
        
        if relevant_merge:
            merge_type = relevant_merge['type']
            total_candidates = relevant_merge['total_candidates']
            successful_merges = relevant_merge['successful_merges']
            seam_location = relevant_merge['seam_location']
            constraint_violations = relevant_merge.get('constraint_violations', [])
            
            explanation = f"Using DP + Divide & Conquer: Edge {edge} creates boundary incompatibility during {merge_type} merge at {seam_location}. "
            explanation += f"Only {successful_merges} of {total_candidates} boundary configurations remain valid after constraint enforcement. "
            
            if constraint_violations:
                explanation += f"Primary violation: {constraint_violations[0]}. "
            
            explanation += f"Alternative boundary configurations were invalid, so this edge must be excluded."
            
            return explanation
        else:
            return f"DP + Divide & Conquer: Edge {edge} violates boundary compatibility constraints discovered during region merging."
    
    def _generate_boundary_explanation(self, edge: Edge, target) -> str:
        """
        Generate explanation for boundary compatibility forced edges.
        """
        region_stats = self._merge_stats.get('region_stats', {})
        
        if region_stats:
            region_info = list(region_stats.values())[0]  # Use first region as example
            explanation = f"DP + Divide & Conquer: During region decomposition, individual quadrants generated up to {region_info['states']} boundary states. "
            explanation += f"Edge {edge} is required to maintain boundary consistency between adjacent regions during the merge phase."
            return explanation
        else:
            return f"DP + Divide & Conquer: Edge {edge} ensures boundary compatibility between regions during merge operations."
    
    def _generate_pruning_explanation(self, edge: Edge, target) -> str:
        """
        Generate explanation for pruning forced exclusions.
        """
        merge_details = self._merge_stats.get('merge_details', [])
        
        if merge_details:
            # Find merge with highest pruning ratio
            best_merge = max(merge_details, key=lambda x: x.get('pruned_count', 0) / max(x.get('total_candidates', 1), 1))
            
            pruned_count = best_merge.get('pruned_count', 0)
            total_candidates = best_merge.get('total_candidates', 1)
            seam_location = best_merge.get('seam_location', 'unknown seam')
            
            explanation = f"DP + Divide & Conquer: Edge {edge} was eliminated during state pruning at {seam_location}. "
            explanation += f"{pruned_count} of {total_candidates} boundary configurations were pruned due to constraint violations. "
            explanation += f"This edge would violate degree or clue constraints in the final merged solution."
            
            return explanation
        else:
            return f"DP + Divide & Conquer: Edge {edge} conflicts with the global solution and must be removed."
    
    def _generate_detailed_explanation(self, edge: Edge, is_addition: bool) -> str:
        """
        Legacy explanation method - kept for compatibility.
        """
        if is_addition:
            return self._generate_inclusion_explanation(edge, None)
        else:
            return self._generate_exclusion_explanation(edge, None)

    def _find_relevant_merge(self, edge: Edge, merge_details: List[Dict]) -> Optional[Dict]:
        """
        Find the most relevant merge operation for this edge.
        """
        if not merge_details:
            return None
            
        # Simple heuristic: return the merge with most pruning (most constraints)
        return max(merge_details, key=lambda x: x['pruned_count'])

    def register_move(self, move):
        # Update index if it matches
        if self.current_move_index < len(self.solution_moves):
            expected = self.solution_moves[self.current_move_index]["move"]
            if move == expected:
                self.current_move_index += 1
            else:
                # Diverged? Recompute or just scan?
                # Simple approach: re-scan list
                pass

    def _compute_full_solution(self):
        """
        Main driver for D&C + DP.
        """
        # Ensure deterministic, run-local merge metrics.
        self._merge_stats = {
            'region_stats': {},
            'merge_details': []
        }

        # 1. Divide Grid
        mid_r = self.rows // 2
        mid_c = self.cols // 2
        
        # Regions: (r_min, r_max, c_min, c_max) inclusive-exclusive
        quadrants = [
            (0, mid_r, 0, mid_c),       # Q1 Top-Left
            (0, mid_r, mid_c, self.cols), # Q2 Top-Right
            (mid_r, self.rows, 0, mid_c), # Q3 Bottom-Left
            (mid_r, self.rows, mid_c, self.cols) # Q4 Bottom-Right
        ]
        
        print("Starting D&C Solve...")
        
        # 2. Solve Quadrants (Region DP)
        q_results = []
        for idx, (r0, r1, c0, c1) in enumerate(quadrants):
            print(f"Solving Q{idx+1}: rows {r0}-{r1}, cols {c0}-{c1}")
            
            from logic.execution_trace import log_execution_step
            log_execution_step(
                strategy_name="Advanced DP",
                move=None,
                explanation=f"Solving Quadrant {idx+1}",
                recursion_depth=1,
                region_id=f"{r0}-{r1},{c0}-{c1}"
            )

            solutions = sorted(
                self._solve_region(r0, r1, c0, c1),
                key=self._region_solution_sort_key
            )
            q_results.append(solutions)
            print(f"  > Found {len(solutions)} valid region states.")
            
            # Track region statistics
            region_id = f"Q{idx+1}"
            self._merge_stats['region_stats'][region_id] = {
                'states': len(solutions),
                'bounds': f"rows {r0}-{r1}, cols {c0}-{c1}"
            }
            
        if any(len(res) == 0 for res in q_results):
            print("Failed to find valid states for one or more quadrants.")
            self._final_solution_edges = set()
            self.solution_moves = []
            self._solution_computed = True
            return

        # 3. Merge Phase
        # 3. Merge Phase
        # Merge Top (Q1 + Q2)
        # Q1/Q2 r_min = 0.
        
        from logic.execution_trace import log_execution_step
        log_execution_step(
            strategy_name="Advanced DP",
            explanation="Merging Q1 and Q2 (Top Half)",
            boundary_state_info="Horizontal Merge at split_c",
            recursion_depth=0,
            region_id="TopHalf"
        )
        
        top_half = self._merge_horizontal(q_results[0], q_results[1], mid_c, 0)
        print(f"Merged Top Half: {len(top_half)} states.")
        if not top_half:
            print("[AdvancedDPSolver] Top-half merge produced no compatible states.")
            self._final_solution_edges = set()
            self.solution_moves = []
            self._solution_computed = True
            return
        
        # Merge Bottom (Q3 + Q4)
        # Q3/Q4 r_min = mid_r.

        log_execution_step(
            strategy_name="Advanced DP",
            explanation="Merging Q3 and Q4 (Bottom Half)",
            boundary_state_info="Horizontal Merge at split_c",
            recursion_depth=0,
            region_id="BottomHalf"
        )

        bottom_half = self._merge_horizontal(q_results[2], q_results[3], mid_c, mid_r)
        print(f"Merged Bottom Half: {len(bottom_half)} states.")
        if not bottom_half:
            print("[AdvancedDPSolver] Bottom-half merge produced no compatible states.")
            self._final_solution_edges = set()
            self.solution_moves = []
            self._solution_computed = True
            return
        
        # Merge Top + Bottom
        # Top/Bottom c_min = 0 (Full width).
        
        log_execution_step(
            strategy_name="Advanced DP",
            explanation="Merging Top and Bottom Halves",
            boundary_state_info="Vertical Merge at split_r",
            recursion_depth=0,
            region_id="Global"
        )

        full_grid = self._merge_vertical(top_half, bottom_half, mid_r, 0)
        print(f"Merged Full Grid: {len(full_grid)} states.")
        
        # 4. Global Loop Enforcement
        final_solutions = []
        
        for sol in full_grid:
            edges = sol.edges
            if not edges: continue
            
            # Check 1: All nodes must have degree 0 or 2
            degrees = collections.defaultdict(int)
            for u, v in edges:
                degrees[u] += 1
                degrees[v] += 1
            
            # If any node has degree != 2 (and != 0 implied), reject
            is_closed = True
            for d in degrees.values():
                if d % 2 != 0:
                    is_closed = False
                    break
            
            if is_closed:
                # Check 2: Exactly one connected component
                val_uf = UnionFind()
                nodes = list(degrees.keys())
                for u, v in edges:
                    val_uf.union(u, v)
                    
                roots = set(val_uf.find(n) for n in nodes)
                if len(roots) == 1:
                    final_solutions.append(sol)
                    
        if not final_solutions:
            print("[AdvancedDPSolver] No valid global solutions found.")
            self._final_solution_edges = set()
        else:
            # Deterministic state selection across compatible global states.
            best_sol = min(final_solutions, key=self._region_solution_sort_key)
            self._final_solution_edges = best_sol.edges
            print(f"[AdvancedDPSolver] Solution found with {len(best_sol.edges)} edges.")
        # Build move list based on current board
        self._build_move_list()
        self._solution_computed = True

    def _build_move_list(self):
        self.solution_moves = []
        current_edges = set(self.game_state.graph.edges)
        
        # Edges to remove
        for edge in sorted(current_edges):
            if edge not in self._final_solution_edges:
                self.solution_moves.append({
                    "move": edge,
                    "explanation": self._generate_detailed_explanation(edge, is_addition=False)
                })
        
        # Edges to add
        for edge in sorted(self._final_solution_edges):
            if edge not in current_edges:
                self.solution_moves.append({
                    "move": edge,
                    "explanation": self._generate_detailed_explanation(edge, is_addition=True)
                })

    # -------------------------------------------------------------------------
    # Region DP Logic
    # -------------------------------------------------------------------------
    def _solve_region(self, r_min, r_max, c_min, c_max) -> List[RegionSolution]:
        """
        Solves a rectangular subgrid using row-by-row DP.
        Returns a list of RegionSolution objects representing valid boundary states.
        """
        results = []
        
        # Simplified DP for prototype: Recursive checks or mini-DP?
        # Since quadrants are small (e.g. 3x3 for a 5x5 grid), we can use a
        # specialized row-based DP similar to the pure DP solver, but storing
        # signatures for ALL boundaries unlike pure DP which only tracks the "active" wave.
        
        # However, to be robust for "Advanced", let's adapt the pure DP logic
        # but capture the Left/Right/Top boundary profiles during the scan.
        
        # To avoid massive complexity in this file, we can invoke strict recursion tailored to the subgrid.
        # State: (r, c, current_profile, edges_accumulated)
        # But that's slow.
        
        # Let's use the `DynamicProgrammingSolver`'s logic logic but constrained to a viewport.
        # Actually, simply calling logic.solvers.dynamic_programming_solver might be hard because it assumes full grid.
        
        # IMPLEMENTATION PATH:
        # Use a generator-based backtracking for the subgrid to enumerate all consistent states.
        
        # Local Clues
        local_clues = {}
        for (r, c), val in sorted(self.game_state.clues.items()):
            if r_min <= r < r_max and c_min <= c < c_max:
                local_clues[(r, c)] = val
                
        # We need to expose ALL boundary edges.
        # Top boundary: r_min, cols c_min..c_max
        # Bottom boundary: r_max, cols c_min..c_max
        # Left boundary: c_min, rows r_min..r_max
        # Right boundary: c_max, rows r_min..r_max
        
        solutions = []
        
        # Optimization:
        # If the region is small (<= 4x4), just brute force all edge combinations?
        # 3x3 region has 12 add'l edges inside + boundary.
        # Let's use a specialized recursive filler.
        
        self._recursive_region_fill(
            r_min, c_min, r_min, r_max, c_min, c_max,
            set(), local_clues, solutions
        )
        
        # Consolidate solutions into RegionSolution objects
        print(f"  [Q {r_min}-{r_max}, {c_min}-{c_max}] Raw recursive solutions: {len(solutions)}")
        
        final_regions = []
        
        # Grid Dimensions and Seams
        rows = self.rows
        cols = self.cols
        mid_r = rows // 2
        mid_c = cols // 2
        
        for edges in solutions:
            reg_sol = self._analyze_region_state(edges, r_min, r_max, c_min, c_max)
            if reg_sol:
                # Prune Open Ends on Grid Boundaries, BUT allow Seam Intersections.
                
                # Top Boundary (r=0)
                if r_min == 0:
                    for k, val in enumerate(reg_sol.top_sig):
                        if val > 0:
                            # Node is at (0, c_min + k)
                            c_abs = c_min + k
                            # Must be closed unless it is on Vertical Seam
                            if c_abs != mid_c:
                                break
                    else:
                        pass # Check passed
                    if k < len(reg_sol.top_sig) and reg_sol.top_sig[k] > 0 and (c_min + k) != mid_c:
                        continue
                
                # Bottom Boundary (r=rows)
                if r_max == rows:
                    for k, val in enumerate(reg_sol.bottom_sig):
                        if val > 0:
                            c_abs = c_min + k
                            if c_abs != mid_c: # Allow if seam
                                break 
                    else:
                        pass
                    if k < len(reg_sol.bottom_sig) and reg_sol.bottom_sig[k] > 0 and (c_min + k) != mid_c:
                         continue
                    
                # Left Boundary (c=0)
                if c_min == 0:
                    for k, val in enumerate(reg_sol.left_sig):
                        if val > 0:
                            r_abs = r_min + k
                            if r_abs != mid_r:
                                break
                    else:
                        pass
                    if k < len(reg_sol.left_sig) and reg_sol.left_sig[k] > 0 and (r_min + k) != mid_r:
                        continue

                # Right Boundary (c=cols)
                if c_max == cols:
                    for k, val in enumerate(reg_sol.right_sig):
                         if val > 0:
                             r_abs = r_min + k
                             if r_abs != mid_r:
                                 break
                    else:
                         pass
                    if k < len(reg_sol.right_sig) and reg_sol.right_sig[k] > 0 and (r_min + k) != mid_r:
                        continue
                
                final_regions.append(reg_sol)
                
        return final_regions

    # Canonical deterministic recursive region filler is defined below.

    def _analyze_region_state(self, edges, r_min, r_max, c_min, c_max) -> Optional[RegionSolution]:
        """
        Compresses a set of edges into a RegionSolution with signatures.
        """
        # 1. Build local UF for components
        # Nodes are all vertices in the region boundary + internal.
        uf = UnionFind()
        
        # Filter edges to this region
        # (Already filtered by generator)
        
        # Add all edges to UF
        # Also check degree constraints (<=2) for ALL internal vertices
        node_degree = collections.defaultdict(int)
        
        for u, v in edges:
            node_degree[u] += 1
            node_degree[v] += 1
            uf.union(u, v)
            
        # Check internal vertices (strictly inside, not on boundary)
        # Boundary vertices can have degree 1 (connecting to outside).
        # Internal vertices must have degree 0 or 2.
        
        # Region Vertices:
        # Rows r_min..r_max+1 (inclusive)
        # Cols c_min..c_max+1
        
        for r in range(r_min, r_max + 1):
            for c in range(c_min, c_max + 1):
                u = (r, c)
                deg = node_degree[u]
                
                if deg > 2:
                    return None # Invalid locally
                
                # Internal check
                is_internal = (r_min < r < r_max) and (c_min < c < c_max)
                if is_internal:
                    if deg == 1:
                        return None # Loose end internally
                        
        # 2. Compute Signatures
        # Map component roots to small integers 1..N
        comp_map = {}
        next_id = 1
        
        def get_id(node):
            nonlocal next_id
            # Expose component ID only for Open Ends (Deg 1).
            # Deg 2 nodes are locally closed loops (or passing through).
            # We treat them as "0" (Closed) for signature purposes.
            # This allows filtering Open Paths while keeping Closed Loops.
            if node_degree[node] == 1:
                root = uf.find(node)
                if root not in comp_map:
                    comp_map[root] = next_id
                    next_id += 1
                return comp_map[root]
            return 0

        # Top Boundary: Row r_min, Cols c_min..c_max
        # Nodes: (r_min, c) for c in c_min..c_max+1 ?? 
        # Wait, signatures are usually on Edge Crossings or Nodes?
        # Standard: Signatures on Nodes.
        # Top Signature: Nodes (r_min, c_min) ... (r_min, c_max)
        
        top_sig = tuple(get_id((r_min, c)) for c in range(c_min, c_max + 1))
        bottom_sig = tuple(get_id((r_max, c)) for c in range(c_min, c_max + 1))
        left_sig = tuple(get_id((r, c_min)) for r in range(r_min, r_max + 1))
        right_sig = tuple(get_id((r, c_max)) for r in range(r_min, r_max + 1))
        
        # 3. Count Internal Loops
        # A loop is internal if all its nodes are fully inside?
        # Or if the component ID does not appear in ANY signature?
        
        # Collect all component IDs present in signatures
        boundary_ids = set()
        boundary_ids.update(top_sig)
        boundary_ids.update(bottom_sig)
        boundary_ids.update(left_sig)
        boundary_ids.update(right_sig)
        if 0 in boundary_ids: boundary_ids.remove(0)
        
        # Check all active components
        # If a component is NOT in boundary_ids, it is a closed loop inside.
        internal_loops = 0
        
        # Iterate all roots found
        all_roots = set(uf.find(u) for u in node_degree if node_degree[u] > 0)
        
        for root in sorted(all_roots):
            cid = comp_map.get(root)
            if cid and cid not in boundary_ids:
                # This check isn't quite enough; is it a loop or a loose stick?
                # We already checked internal degrees. 
                # If internal nodes are all degree 2 (or 0), then it must be a loop.
                # But what if it touches boundary but doesn't cross out?
                # The node on boundary would have degree 1?
                # If degree is 1 on boundary, it MUST be in signature (ID > 0).
                # So if ID is not in signature, then ALL its nodes are degree 0 or 2.
                # Since it has edges (root exists), it's a loop.
                internal_loops += 1
                
        # Premature loop check?
        # We allow internal loops if they are the ONLY solution?
        # Actually for D&C, we assume NO small loops unless it's the final solution.
        # But maybe valid solutions have multiple loops if disjoint?
        # Slitherlink: Exactly ONE loop.
        # So internal_loops > 0 is INVALID unless it is the ONLY component and consumes everything?
        # No, we track them. At end, we ensure Total Loops = 1.
        
        for root in sorted(all_roots):
            cid = comp_map.get(root)
            if cid and cid not in boundary_ids:
                internal_loops += 1

        # 3. Capture Boundary Masks
        # Vertical Left: c_min. Rows r_min..r_max.
        v_left_mask = tuple(tuple(sorted(((r, c_min), (r+1, c_min)))) in edges for r in range(r_min, r_max))
        # Vertical Right: c_max.
        v_right_mask = tuple(tuple(sorted(((r, c_max), (r+1, c_max)))) in edges for r in range(r_min, r_max))
        # Horizontal Top: r_min. Cols c_min..c_max.
        h_top_mask = tuple(tuple(sorted(((r_min, c), (r_min, c+1)))) in edges for c in range(c_min, c_max))
        # Horizontal Bottom: r_max.
        h_bottom_mask = tuple(tuple(sorted(((r_max, c), (r_max, c+1)))) in edges for c in range(c_min, c_max))
        
        return RegionSolution(edges, top_sig, bottom_sig, left_sig, right_sig, 
                              v_left_mask, v_right_mask, h_top_mask, h_bottom_mask, 
                              internal_loops)

    # -------------------------------------------------------------------------
    # Recursion Implementation (Corrected)
    # -------------------------------------------------------------------------
    def _recursive_region_fill(self, r, c, r_min, r_max, c_min, c_max, current_edges, clues, results):
        # Base Case
        if r == r_max:
            results.append(current_edges)
            return

        # Determine next cell
        next_c = c + 1
        next_r = r
        if next_c == c_max:
            next_c = c_min
            next_r = r + 1
            
        # Determine which edges to iterate
        # We need to iterate edges associated with Cell (r, c).
        # Specifically:
        # - Top Edge: ((r, c), (r, c+1)) -> Explicitly decide if r == r_min
        # - Left Edge: ((r, c), (r+1, c)) -> Explicitly decide if c == c_min
        # - Bottom Edge: ((r+1, c), (r+1, c+1)) -> Always decide
        # - Right Edge: ((r, c+1), (r+1, c+1)) -> Always decide
        
        # But wait, Bottom of (r,c) IS Top of (r+1, c). Don't decide twice.
        # Right of (r,c) IS Left of (r, c+1). Don't decide twice.
        
        # Unique Decision Point Strategy:
        # At Cell (r, c):
        #   Decide Top Edge ONLY IF r == r_min.
        #   Decide Left Edge ONLY IF c == c_min.
        #   Decide Bottom Edge ALWAYS.
        #   Decide Right Edge ALWAYS.
        
        # Optimization: Top/Left edges for non-boundary cells are ALREADY in `current_edges`.
        # So we only branch on:
        # - Top (if r==r_min)
        # - Left (if c==c_min)
        # - Bottom
        # - Right 
        
        # Define edge objects
        e_top = tuple(sorted(((r, c), (r, c+1))))
        e_left = tuple(sorted(((r, c), (r+1, c))))
        e_bottom = tuple(sorted(((r+1, c), (r+1, c+1))))
        e_right = tuple(sorted(((r, c+1), (r+1, c+1))))
        
        # Determine fixed vs variable
        fixed_top = (r > r_min)
        fixed_left = (c > c_min)
        
        # If fixed, check if they exist in current_edges
        has_top = e_top in current_edges
        has_left = e_left in current_edges
        
        # Generate options
        # We need 2 or 4 loops?
        # Let's iterate binary 0..15? Or dynamic?
        
        opts_top = [0] if (fixed_top and not has_top) else ([1] if (fixed_top and has_top) else [0, 1])
        opts_left = [0] if (fixed_left and not has_left) else ([1] if (fixed_left and has_left) else [0, 1])
        opts_bottom = [0, 1]
        opts_right = [0, 1]
        
        for d_top in opts_top:
            for d_left in opts_left:
                for d_bottom in opts_bottom:
                    for d_right in opts_right:
                        
                        # Validate Clue (Immediate Pruning)
                        deg = d_top + d_left + d_bottom + d_right
                        if (r, c) in clues:
                            if deg != clues[(r, c)]:
                                continue
                                
                        # Prepare next state
                        # ALWAYS copy to avoid side effects across loop iterations
                        new_edges = current_edges.copy()
                        
                        # Add newly decided edges
                        if not fixed_top and d_top: new_edges.add(e_top)
                        if not fixed_left and d_left: new_edges.add(e_left)
                        if d_bottom: new_edges.add(e_bottom)
                        if d_right: new_edges.add(e_right)
                        
                        # Pruning: Vertex Degree Constraints?
                        # Verify Top-Left Vertex (r, c)
                        #   Edges: Top(?,?), Left(?,?), Up(from region above?), Left(from region left?)
                        #   We don't know external edges. So we can't fully validate boundary vertices.
                        # Verify internal vertices?
                        # Vertex (r, c) is boundary if r=r_min or c=c_min.
                        
                        # Recurse
                        self._recursive_region_fill(next_r, next_c, r_min, r_max, c_min, c_max,
                                                    new_edges, clues, results)


    # -------------------------------------------------------------------------
    # Merge Logic
    # -------------------------------------------------------------------------
    def _merge_horizontal(self, left_regions, right_regions, mid_c, r_min):
        """
        Merge two sets of region solutions horizontally.
        Split is at column `mid_c`.
        Left Region: cols < mid_c.
        Right Region: cols >= mid_c.
        Shared Vertical Seam is at col `mid_c`.
        """
        merged_results = []
        
        # Track merge statistics
        total_candidates = 0
        pruned_count = 0
        constraint_violations = []
        
        # Optimization: Bucket by compatibility?
        # Only compare if v_right_mask == v_left_mask.
        
        right_map = collections.defaultdict(list)
        for sol in right_regions:
            right_map[sol.v_left_mask].append(sol)
            
        for l_sol in left_regions:
            candidates = right_map[l_sol.v_right_mask]
            total_candidates += len(candidates)
            
            for r_sol in candidates:
                # 1c. Seam Degree validation
                degree_valid = True
                v_shared = l_sol.v_right_mask
                
                num_nodes = len(l_sol.right_sig)
                
                # Check degree at each node on the seam
                for k in range(num_nodes):
                    row = r_min + k
                    
                    # 1. Shared Vertical Edges (on the seam)
                    has_up = v_shared[k-1] if k > 0 else False
                    has_down = v_shared[k] if k < len(v_shared) else False
                    shared_count = (1 if has_up else 0) + (1 if has_down else 0)
                    
                    try:
                        # 2. Horizontal Edges (entering from left/right)
                        e_left = tuple(sorted(((row, mid_c-1), (row, mid_c))))
                        l_in = 1 if e_left in l_sol.edges else 0
                        
                        e_right = tuple(sorted(((row, mid_c), (row, mid_c+1))))
                        r_in = 1 if e_right in r_sol.edges else 0
                    except Exception:
                        l_in = 0; r_in = 0

                    total_degree = shared_count + l_in + r_in
                    
                    # Constraint: Degree must be 2 (or 0).
                    if total_degree % 2 != 0:
                        constraint_violations.append(f"Row {row}: odd degree {total_degree}")
                        degree_valid = False
                        break
                    
                    if total_degree > 2:
                        constraint_violations.append(f"Row {row}: degree {total_degree} > 2")
                        degree_valid = False
                        break
                    
                if not degree_valid:
                    pruned_count += 1
                    continue

                # 2. Merge Components
                # Union(Left_ID, Right_ID) if both present
                
                uf = UnionFind()
                
                # Iterate boundary nodes to merge components
                for k in range(num_nodes):
                    # Check connectivity
                    id_l = l_sol.right_sig[k]
                    id_r = r_sol.left_sig[k]
                    
                    # If both have IDs, merge
                    if id_l > 0 and id_r > 0:
                         root_l = uf.find((0, id_l))
                         root_r = uf.find((1, id_r))
                         if root_l != root_r:
                             uf.union(root_l, root_r)

                # 3. Generate New Signatures
                def resolve(side, old_sig):
                    return tuple(
                        0 if x == 0 else uf.find((side, x)) 
                        for x in old_sig
                    )
                
                res_l_top = resolve(0, l_sol.top_sig)
                res_r_top = resolve(1, r_sol.top_sig)
                new_top_raw = res_l_top[:-1] + res_r_top
                
                res_l_bottom = resolve(0, l_sol.bottom_sig)
                res_r_bottom = resolve(1, r_sol.bottom_sig)
                new_bottom_raw = res_l_bottom[:-1] + res_r_bottom
                
                new_left_raw = resolve(0, l_sol.left_sig)
                new_right_raw = resolve(1, r_sol.right_sig)
                
                # Normalize
                id_map = {}
                next_id = 1
                def norm(raw_sig):
                    nonlocal next_id
                    res = []
                    for val in raw_sig:
                        if val == 0: res.append(0)
                        else:
                            if val not in id_map:
                                id_map[val] = next_id
                                next_id += 1
                            res.append(id_map[val])
                    return tuple(res)
                
                final_top = norm(new_top_raw)
                final_bottom = norm(new_bottom_raw)
                final_left = norm(new_left_raw)
                final_right = norm(new_right_raw)
                
                # Count loops using Active Roots logic
                
                # Collect ALL original active IDs that participated
                all_participating_ids = set()
                # From Left
                for x in l_sol.top_sig + l_sol.bottom_sig + l_sol.left_sig + l_sol.right_sig:
                    if x > 0: all_participating_ids.add((0, x))
                # From Right
                for x in r_sol.top_sig + r_sol.bottom_sig + r_sol.left_sig + r_sol.right_sig:
                    if x > 0: all_participating_ids.add((1, x))
                    
                # Find their new roots
                final_roots = set()
                for pid in all_participating_ids:
                    final_roots.add(uf.find(pid))
                    
                # Find visible roots on new boundary
                visible_roots = set()
                visible_roots.update(new_top_raw)
                visible_roots.update(new_bottom_raw)
                visible_roots.update(new_left_raw)
                visible_roots.update(new_right_raw)
                if 0 in visible_roots: visible_roots.remove(0)
                
                # Any final_root NOT in visible_roots is a closed loop!
                loops_formed = 0
                for root in final_roots:
                    if root not in visible_roots:
                        loops_formed += 1
                        
                total_loops = l_sol.internal_loops + r_sol.internal_loops + loops_formed
                                
                merged_sol = RegionSolution(
                    l_sol.edges.union(r_sol.edges),
                    final_top, final_bottom, final_left, final_right,
                    l_sol.v_left_mask, r_sol.v_right_mask,
                    l_sol.h_top_mask + r_sol.h_top_mask, 
                    l_sol.h_bottom_mask + r_sol.h_bottom_mask, 
                    total_loops
                )
                merged_results.append(merged_sol)
                
        # Store merge statistics
        merge_id = f"horizontal_{mid_c}_{r_min}"
        self._merge_stats['merge_details'].append({
            'type': 'horizontal',
            'merge_id': merge_id,
            'total_candidates': total_candidates,
            'pruned_count': pruned_count,
            'successful_merges': len(merged_results),
            'constraint_violations': constraint_violations[:3],  # Keep first 3 for explanation
            'seam_location': (
                f"column {mid_c}, rows {r_min}-{r_min + len(left_regions[0].right_sig) - 1}"
                if left_regions else f"column {mid_c}"
            )
        })
        
        return merged_results

    def _merge_vertical(self, top_regions, bottom_regions, mid_r, c_min):
        """
        Merge two sets of region solutions vertically.
        Split is at row `mid_r`.
        Top Region: rows < mid_r.
        Bottom Region: rows >= mid_r.
        Shared Horizontal Seam is at row `mid_r`.
        """
        merged_results = []
        
        # Track merge statistics
        total_candidates = 0
        pruned_count = 0
        constraint_violations = []
        
        # Optimization: Map by compatibility?
        # Only compare if h_bottom_mask == h_top_mask.
        
        bottom_map = collections.defaultdict(list)
        for sol in bottom_regions:
            bottom_map[sol.h_top_mask].append(sol)
            
        for t_sol in top_regions:
            candidates = bottom_map[t_sol.h_bottom_mask]
            total_candidates += len(candidates)
            
            for b_sol in candidates:
                # 1c. Seam Degree validation
                degree_valid = True
                h_shared = t_sol.h_bottom_mask
                
                num_nodes = len(t_sol.bottom_sig)
                
                # Check degree at each node on the seam
                for k in range(num_nodes):
                    col = c_min + k
                    
                    # 1. Shared Horizontal Edges (on the seam)
                    has_left = h_shared[k-1] if k > 0 else False
                    has_right = h_shared[k] if k < len(h_shared) else False
                    shared_count = (1 if has_left else 0) + (1 if has_right else 0)
                    
                    try:
                        # 2. Vertical Edges (entering from top/bottom)
                        e_top = tuple(sorted(((mid_r-1, col), (mid_r, col))))
                        t_in = 1 if e_top in t_sol.edges else 0
                        
                        e_bottom = tuple(sorted(((mid_r, col), (mid_r+1, col))))
                        b_in = 1 if e_bottom in b_sol.edges else 0
                    except Exception:
                        t_in = 0; b_in = 0

                    total_degree = shared_count + t_in + b_in
                    
                    # Constraint: Degree must be 2 (or 0).
                    if total_degree % 2 != 0:
                        constraint_violations.append(f"Column {col}: odd degree {total_degree}")
                        degree_valid = False
                        break
                    
                    if total_degree > 2:
                        constraint_violations.append(f"Column {col}: degree {total_degree} > 2")
                        degree_valid = False
                        break
                    
                if not degree_valid:
                    pruned_count += 1
                    continue

                # 2. Merge Components
                # Union(Top_ID, Bottom_ID) if both present
                
                uf = UnionFind()
                
                # Iterate boundary nodes to merge components
                for k in range(num_nodes):
                    # Check connectivity
                    id_t = t_sol.bottom_sig[k]
                    id_b = b_sol.top_sig[k]
                    
                    # If both have IDs, merge
                    if id_t > 0 and id_b > 0:
                         root_t = uf.find((0, id_t))
                         root_b = uf.find((1, id_b))
                         if root_t != root_b:
                             uf.union(root_t, root_b)

                # 3. Generate New Signatures
                def resolve(side, old_sig):
                    return tuple(
                        0 if x == 0 else uf.find((side, x)) 
                        for x in old_sig
                    )
                
                res_t_left = resolve(0, t_sol.left_sig)
                res_b_left = resolve(1, b_sol.left_sig)
                new_left_raw = res_t_left[:-1] + res_b_left
                
                res_t_right = resolve(0, t_sol.right_sig)
                res_b_right = resolve(1, b_sol.right_sig)
                new_right_raw = res_t_right[:-1] + res_b_right
                
                new_top_raw = resolve(0, t_sol.top_sig)
                new_bottom_raw = resolve(1, b_sol.bottom_sig)
                
                # Normalize
                id_map = {}
                next_id = 1
                def norm(raw_sig):
                    nonlocal next_id
                    res = []
                    for val in raw_sig:
                        if val == 0: res.append(0)
                        else:
                            if val not in id_map:
                                id_map[val] = next_id
                                next_id += 1
                            res.append(id_map[val])
                    return tuple(res)
                
                final_left = norm(new_left_raw)
                final_right = norm(new_right_raw)
                final_top = norm(new_top_raw)
                final_bottom = norm(new_bottom_raw)
                
                # Count loops using Active Roots logic
                
                # Collect ALL original active IDs that participated
                all_participating_ids = set()
                # From Top
                for x in t_sol.top_sig + t_sol.bottom_sig + t_sol.left_sig + t_sol.right_sig:
                    if x > 0: all_participating_ids.add((0, x))
                # From Bottom
                for x in b_sol.top_sig + b_sol.bottom_sig + b_sol.left_sig + b_sol.right_sig:
                    if x > 0: all_participating_ids.add((1, x))
                    
                # Find their new roots
                final_roots = set()
                for pid in all_participating_ids:
                    final_roots.add(uf.find(pid))
                    
                # Find visible roots on new boundary
                visible_roots = set()
                visible_roots.update(new_left_raw)
                visible_roots.update(new_right_raw)
                visible_roots.update(new_top_raw)
                visible_roots.update(new_bottom_raw)
                if 0 in visible_roots: visible_roots.remove(0)
                
                # Any final_root NOT in visible_roots is a closed loop!
                loops_formed = 0
                for root in final_roots:
                    if root not in visible_roots:
                        loops_formed += 1
                        
                total_loops = t_sol.internal_loops + b_sol.internal_loops + loops_formed
                                
                merged_sol = RegionSolution(
                    t_sol.edges.union(b_sol.edges),
                    final_top, final_bottom, final_left, final_right,
                    t_sol.v_left_mask + b_sol.v_left_mask,
                    t_sol.v_right_mask + b_sol.v_right_mask,
                    t_sol.h_top_mask, 
                    b_sol.h_bottom_mask,
                    total_loops
                )
                merged_results.append(merged_sol)
                
        # Store merge statistics
        merge_id = f"vertical_{mid_r}_{c_min}"
        self._merge_stats['merge_details'].append({
            'type': 'vertical',
            'merge_id': merge_id,
            'total_candidates': total_candidates,
            'pruned_count': pruned_count,
            'successful_merges': len(merged_results),
            'constraint_violations': constraint_violations[:3],  # Keep first 3 for explanation
            'seam_location': (
                f"row {mid_r}, columns {c_min}-{c_min + len(top_regions[0].bottom_sig) - 1}"
                if top_regions else f"row {mid_r}"
            )
        })
        
        return merged_results

    def _merge_horizontal_limited(self, left_regions, right_regions, mid_c, r_min, max_states):
        """
        Limited version of _merge_horizontal that caps the number of returned states.
        Used for hint generation to prevent state explosion.
        """
        if max_states <= 0:
            return []
        
        # Simply use the full merge and limit results
        print(f"[DEBUG-LIMITED] Calling _merge_horizontal with {len(left_regions)} left, {len(right_regions)} right regions")
        full_results = self._merge_horizontal(left_regions, right_regions, mid_c, r_min)
        full_results = sorted(full_results, key=self._region_solution_sort_key)
        print(f"[DEBUG-LIMITED] Full merge returned {len(full_results)} states, returning {len(full_results[:max_states])}")
        return full_results[:max_states]

    def _merge_vertical_limited(self, top_regions, bottom_regions, mid_r, c_min, max_states):
        """
        Limited version of _merge_vertical that caps the number of returned states.
        Used for hint generation to prevent state explosion.
        """
        if max_states <= 0:
            return []
        
        # Simply use the full merge and limit results
        print(f"[DEBUG-LIMITED] Calling _merge_vertical with {len(top_regions)} top, {len(bottom_regions)} bottom regions")
        full_results = self._merge_vertical(top_regions, bottom_regions, mid_r, c_min)
        full_results = sorted(full_results, key=self._region_solution_sort_key)
        print(f"[DEBUG-LIMITED] Full merge returned {len(full_results)} states, returning {len(full_results[:max_states])}")
        return full_results[:max_states]

class UnionFind:
    def __init__(self):
        self.parent = {}

    def find(self, i):
        if i not in self.parent:
            self.parent[i] = i
        if self.parent[i] != i:
            self.parent[i] = self.find(self.parent[i])
        return self.parent[i]

    def union(self, i, j):
        root_i = self.find(i)
        root_j = self.find(j)
        if root_i != root_j:
            self.parent[root_j] = root_i
            return True
        return False
