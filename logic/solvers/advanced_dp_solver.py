"""
Advanced DP Solver
==================
Implements "Aggregated Profile Dynamic Programming with Divide & Conquer".

Strategy:
1. Spatial Decomposition: Dictionary-based DP structure for recursive regions.
2. Aggregated Profiles: Instead of storing all primitive solutions, we store:
   - Signature: Canonical boundary component IDs.
   - Count: Number of valid internal solutions matching this signature.
   - Edge Counts: Map[Edge, Count] tracking frequency of every internal edge.
3. Combinatorial Merging:
   - When merging Region A and Region B:
     - New Count = Count_A * Count_B
     - New Edge Count(e in A) = EdgeCount_A(e) * Count_B
     - New Edge Count(e in B) = Count_A * EdgeCount_B(e)
4. Certainty-Based Decisions:
   - Forced Inclusion: Edge Count == Total Count for valid global solutions.
   - Forced Exclusion: Edge Count == 0.
"""

from __future__ import annotations

import collections
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

from logic.solvers.solver_interface import AbstractSolver, HintPayload
from logic.validators import is_valid_move

# Type aliases
# Edge: ((r1, c1), (r2, c2)) sorted
Edge = Tuple[Tuple[int, int], Tuple[int, int]]
# Signature: tuple of component IDs along a boundary line
Signature = Tuple[int, ...]

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
            self.parent[root_i] = root_j
            return True
        return False

@dataclass(frozen=True)
class BoundarySignature:
    """
    Identifies the connectivity 'interface' of a region.
    Component IDs are canonicalized (min-based or normalized).
    """
    top: Signature
    bottom: Signature
    left: Signature
    right: Signature

@dataclass
class RegionProfile:
    """
    Aggregated stats for ALL solutions matching a specific BoundarySignature.
    """
    signature: BoundarySignature
    count: int = 0
    # Map edge -> how many solutions in this bucket have this edge?
    edge_counts: Dict[Edge, int] = field(default_factory=lambda: collections.defaultdict(int))
    # Keep one concrete example for traceback/debug
    sample_edges: Set[Edge] = field(default_factory=set)
    # Number of fully closed loops completely inside this region (accumulated)
    internal_loops: int = 0  

    def __repr__(self):
        return f"Profile(count={self.count}, loops={self.internal_loops}, sig={self.signature})"


class AdvancedDPSolver(AbstractSolver):
    def __init__(self, game_state: Any):
        self.game_state = game_state
        self.rows = game_state.rows
        self.cols = game_state.cols
        self.last_explanation = ""
        
        # Memoization for regions: (r_min, r_max, c_min, c_max) -> Map[BoundarySignature, RegionProfile]
        self._memo_regions: Dict[Tuple[int, int, int, int], Dict[BoundarySignature, RegionProfile]] = {}
        
        # Global solution stats
        self._solution_computed = False
        self._final_edge_probabilities: Dict[Edge, float] = {}
        self._total_valid_solutions = 0
        
        # Enhanced Hint System - Phase 1 Stats
        self._merge_stats = {
            "region_stats": {},
            "merge_details": []
        }
        self._boundary_edges: Set[Edge] = set()
        
        # Test Compatibility
        self._final_solution_edges: List[Edge] = []
        
        # Method Alias for Tests
        self._compute_full_solution = self._compute_full_logic
        
    def solve(self, board: Any = None):
        """Standard Solver API."""
        if not self._solution_computed:
            self._compute_full_logic()
            
        move = self._pick_best_move()
        return move

    def decide_move(self) -> Tuple[List[Tuple[Edge, int]], Optional[Edge]]:
        """UI Hook."""
        if not self._solution_computed:
            self._compute_full_logic()

        move = self._pick_best_move()
        if move:
            prob = self._final_edge_probabilities.get(move, 0.0)
            confidence = int(abs(prob - 0.5) * 200) # 0.5->0, 1.0->100
            return [(move, confidence)], move
        return [], None

    def explain_last_move(self) -> str:
        return self.last_explanation

    def generate_hint(self, board: Any = None) -> HintPayload:
        if not self._solution_computed:
            self._compute_full_logic()
            
        if self._total_valid_solutions == 0:
             return {
                "move": None,
                "strategy": "Advanced DP",
                "explanation": "No valid global solutions found consistent with current board."
            }

        current_edges = set(self.game_state.graph.edges)
        
        # 1. Certainty
        for edge, prob in sorted(self._final_edge_probabilities.items(), key=lambda x: -abs(x[1]-0.5)):
            if prob >= 0.999 and edge not in current_edges:
                return {
                    "move": edge,
                    "strategy": "Advanced DP",
                    "explanation": f"Certainty {prob*100:.1f}%: This edge appears in all {self._total_valid_solutions} valid solutions."
                }
            if prob <= 0.001 and edge in current_edges:
                return {
                    "move": edge,
                    "strategy": "Advanced DP",
                    "explanation": f"Certainty {(1-prob)*100:.1f}%: This edge appears in none of the {self._total_valid_solutions} valid solutions."
                }
                
        # 2. Key Moves (High Prob)
        best_edge = None
        best_certainty = -1.0
        
        all_potential_edges = self._get_all_potential_edges()
        
        for edge in all_potential_edges:
            if edge in current_edges: continue 
            
            prob = self._final_edge_probabilities.get(edge, 0.0)
            if prob > 0.5:
                certainty = prob
                if certainty > best_certainty:
                    best_certainty = certainty
                    best_edge = edge
                        
        if best_edge:
             return {
                "move": best_edge,
                "strategy": "Advanced DP",
                "explanation": f"Probabilistic Hint ({best_certainty*100:.1f}%): Analysis suggests adding this edge."
            }
            
        return {
            "move": None,
            "strategy": "Advanced DP",
            "explanation": "Ambiguous state. Multiple solutions possible with no strong shared moves."
        }

    # --------------------------------------------------------------------------
    # Enhanced Hint Logic (Phase 1)
    # --------------------------------------------------------------------------

    def _detect_forced_inclusions(self, game_state) -> List[Edge]:
        """Priority 1: Detect edges that MUST be included in ALL valid solutions."""
        if not self._solution_computed:
            self._compute_full_logic()
            
        forced_edges = []
        current_edges = set(game_state.graph.edges)
        
        for edge, prob in self._final_edge_probabilities.items():
            if prob >= 0.999 and edge not in current_edges:
                forced_edges.append(edge)
                
        return forced_edges

    def _detect_forced_exclusions(self, game_state) -> List[Edge]:
        """Priority 2: Detect edges that CANNOT be included in ANY valid solution."""
        if not self._solution_computed:
            self._compute_full_logic()
            
        forced_edges = []
        current_edges = set(game_state.graph.edges)
        
        for edge, prob in self._final_edge_probabilities.items():
            if prob <= 0.001 and edge in current_edges:
                forced_edges.append(edge)
                
        return forced_edges

    def _detect_boundary_compatibility_forced_edges(self, game_state) -> List[Edge]:
        """Priority 3: Detect edges forced by region boundary compatibility constraints."""
        forced_inclusions = self._detect_forced_inclusions(game_state)
        boundary_forced = []
        
        for edge in forced_inclusions:
            if edge in self._boundary_edges:
                boundary_forced.append(edge)
                
        return boundary_forced

    def _detect_pruning_forced_exclusions(self, game_state) -> List[Edge]:
        """Priority 4: Detect edges excluded due to pruning during DP merge."""
        forced_exclusions = self._detect_forced_exclusions(game_state)
        pruned_forced = []
        
        # Heuristic: If an edge is forced excluded AND it was part of a boundary merge
        # (or adjacent to one), it is likely due to pruning.
        # Simple implementation: Same as boundary exclusions for now.
        for edge in forced_exclusions:
            if edge in self._boundary_edges:
                pruned_forced.append(edge)
                
        return pruned_forced

    def _generate_inclusion_explanation(self, edge: Edge, total_solutions: int) -> str:
        return (f"Constructive Move (DP + Divide & Conquer): Analysis of {total_solutions} valid global solutions "
                f"confirms this edge is required in 100% of configurations.")

    def _generate_exclusion_explanation(self, edge: Edge, total_solutions: int) -> str:
        return (f"Reactive Move (DP Constraints): This edge appears in 0% of the {total_solutions} "
                f"valid solutions and must be removed to satisfy loop constraints.")

    def _generate_boundary_explanation(self, edge: Edge, total_solutions: int) -> str:
        return (f"Boundary Merge Analysis: This edge lies on a crucial region seam. "
                f"Merging sub-grid solutions forces this connection to maintain loop continuity.")

    def _generate_pruning_explanation(self, edge: Edge, total_solutions: int) -> str:
        return (f"State Pruning: During region merging, all configurations containing this edge "
                f"were pruned due to incompatibility with neighboring regions.")

    # --------------------------------------------------------------------------
    # Core Logic
    # --------------------------------------------------------------------------
    
    def _compute_full_logic(self):
        self._memo_regions.clear()
        self._final_edge_probabilities.clear()
        self._boundary_edges.clear()
        self._merge_stats["region_stats"].clear()
        self._merge_stats["merge_details"].clear()
        self._total_valid_solutions = 0
        
        root_profiles = self._solve_region_recursive(0, self.rows, 0, self.cols)
        
        valid_profiles = []
        total_count = 0
        
        for sig, profile in root_profiles.items():
            if any(sig.top) or any(sig.bottom) or any(sig.left) or any(sig.right):
                continue
            
            if profile.internal_loops != 1:
                continue
                
            valid_profiles.append(profile)
            total_count += profile.count
            
        self._total_valid_solutions = total_count
        
        if total_count == 0:
            self.last_explanation = "No valid solutions found."
            self._solution_computed = True
            return

        aggregated_edge_counts = collections.defaultdict(int)
        
        for profile in valid_profiles:
            for edge, count in profile.edge_counts.items():
                aggregated_edge_counts[edge] += count
                
        for edge, count in aggregated_edge_counts.items():
            prob = count / total_count
            self._final_edge_probabilities[edge] = prob
            
            # Populate final solution with high-probability edges
            if prob > 0.5:
                self._final_solution_edges.append(edge)
            
        self._solution_computed = True
        self.last_explanation = f"Analyzed {total_count} valid global solutions."
        
    def _solve_region_recursive(self, r_min, r_max, c_min, c_max) -> Dict[BoundarySignature, RegionProfile]:
        region_key = (r_min, r_max, c_min, c_max)
        if region_key in self._memo_regions:
            return self._memo_regions[region_key]
        
        rows = r_max - r_min
        cols = c_max - c_min
        
        # Base Case: 2x3 or smaller (<= 6 cells)
        if rows * cols <= 6:
            profiles = self._solve_base_case(r_min, r_max, c_min, c_max)
            self._memo_regions[region_key] = profiles
            return profiles
            
        if rows >= cols:
            split = r_min + (rows // 2)
            # Track horizontal boundary edges along split row
            for c in range(c_min, c_max):
                self._boundary_edges.add(tuple(sorted(((split, c), (split, c+1)))))
                
            top = self._solve_region_recursive(r_min, split, c_min, c_max)
            bottom = self._solve_region_recursive(split, r_max, c_min, c_max)
            profiles = self._merge_vertical_profiles(top, bottom, split, c_min, c_max)
        else:
            split = c_min + (cols // 2)
            # Track vertical boundary edges along split col
            for r in range(r_min, r_max):
                self._boundary_edges.add(tuple(sorted(((r, split), (r+1, split)))))
                
            left = self._solve_region_recursive(r_min, r_max, c_min, split)
            right = self._solve_region_recursive(r_min, r_max, split, c_max)
            profiles = self._merge_horizontal_profiles(left, right, r_min, r_max, split)
            
        self._memo_regions[region_key] = profiles
        
        # Stats tracking
        self._merge_stats["region_stats"][region_key] = len(profiles)
        
        return profiles

    # --------------------------------------------------------------------------
    # Base Case Enumeration (Optimized with Pruning)
    # --------------------------------------------------------------------------
    
    def _solve_base_case(self, r_min, r_max, c_min, c_max) -> Dict[BoundarySignature, RegionProfile]:
        local_clues = {}
        for (r, c), val in self.game_state.clues.items():
             if r_min <= r < r_max and c_min <= c < c_max:
                 local_clues[(r,c)] = val
                 
        solutions = []
        self._recursive_enumerate(r_min, c_min, r_min, r_max, c_min, c_max, set(), local_clues, solutions)
        
        profiles: Dict[BoundarySignature, RegionProfile] = {}
        
        for edges in solutions:
            sig, internal_loops = self._compute_signature(edges, r_min, r_max, c_min, c_max)
            if sig is None: continue 
            
            if sig not in profiles:
                profiles[sig] = RegionProfile(signature=sig, count=0, internal_loops=internal_loops)
                profiles[sig].sample_edges = edges
            
            prof = profiles[sig]
            prof.count += 1
            for edge in edges:
                prof.edge_counts[edge] += 1
                
        return profiles

    def _recursive_enumerate(self, r, c, r_min, r_max, c_min, c_max, current_edges, clues, results):
        if r == r_max:
            results.append(current_edges)
            return
            
        next_c = c + 1
        next_r = r
        if next_c == c_max:
            next_c = c_min
            next_r = r + 1
            
        e_top = tuple(sorted(((r, c), (r, c+1))))
        e_left = tuple(sorted(((r, c), (r+1, c))))
        e_bottom = tuple(sorted(((r+1, c), (r+1, c+1))))
        e_right = tuple(sorted(((r, c+1), (r+1, c+1))))
        
        has_top = e_top in current_edges
        has_left = e_left in current_edges
        
        edges_to_decide = []
        if r == r_min: edges_to_decide.append(e_top)
        if c == c_min: edges_to_decide.append(e_left)
        edges_to_decide.append(e_bottom)
        edges_to_decide.append(e_right)
        
        n = len(edges_to_decide)
        for i in range(1 << n):
            deg = 0
            temp_edges = set()
            
            if r > r_min and has_top: deg += 1
            if c > c_min and has_left: deg += 1
            
            for k in range(n):
                is_on = (i >> k) & 1
                if is_on:
                    deg += 1
                    temp_edges.add(edges_to_decide[k])
            
            if (r, c) in clues:
                if deg != clues[(r, c)]:
                    continue
                    
            # --- Pruning: Vertex Degree Check for (r, c) ---
            # Vertex (r, c) is fully connected once we decide its outgoing edges (Right, Down).
            # The incoming edges (Left, Up) were decided by previous cells.
            
            # Edges incident to Vertex (r, c):
            # 1. Right: (r, c)-(r, c+1) [e_top of THIS cell]
            # 2. Down: (r, c)-(r+1, c) [e_left of THIS cell]
            # 3. Left: (r, c-1)-(r, c) [e_top of LEFT neighbor]
            # 4. Up: (r-1, c)-(r, c) [e_left of TOP neighbor]
            
            # Since we iterate cells in row-major order:
            # We are currently Deciding e_top and e_left for (r,c).
            # We already decided incoming edges.
            
            # So Vertex (r,c) is complete.
            
            v_deg = 0
            # 1. Right edge (e_top of current cell)
            # e_top is index 0 in edges_to_decide IF r == r_min.
            # If r > r_min, e_top was decided previously! (It is e_bottom of (r-1, c)).
            # Wait. My edge definitions for Cell (r,c):
            # Top Edge: ((r,c), (r,c+1)). This is "Top" of cell.
            #   If r > r_min, this IS "Bottom" of cell (r-1, c).
            #   So it is ALREADY DECIDED and passed in `current_edges`. `has_top` tracks it.
            #   So we DO NOT decide it here.
            
            # My logic for `edges_to_decide`:
            # `if r == r_min: append(e_top)` -> Only decide top edge for first row.
            # Otherwise we rely on `has_top` (passed from prev row).
            
            # Same for `e_left`.
            
            # So:
            # Right Edge incident to Vertex (r,c):
            # This is `e_top` of Cell (r,c).
            # If r == r_min: it is being decided now (bit in `i`).
            # If r > r_min: it is `has_top` (decided).
            
            has_right_inc = False
            if r == r_min:
                # e_top is at index 0
                if (i >> 0) & 1: has_right_inc = True
            else:
                if has_top: has_right_inc = True
            if has_right_inc: v_deg += 1
            
            # Down Edge incident to Vertex (r,c):
            # This is `e_left` of Cell (r,c).
            # If c == c_min: decided now.
            # If c > c_min: decided previously (`has_left`).
            
            has_down_inc = False
            if c == c_min:
                # e_left is at index 1 (if r==r_min) or 0 (if r>r_min)?
                # Need consistent indexing.
                idx = 0
                if r == r_min: idx += 1 # e_top was 0
                if (i >> idx) & 1: has_down_inc = True
            else:
                if has_left: has_down_inc = True
            if has_down_inc: v_deg += 1
            
            # Left Edge incident to Vertex (r,c):
            # This is ((r, c-1), (r, c)). Top Edge of Cell (r, c-1).
            # We don't have direct access. We must check `current_edges`.
            if c > c_min:
                 e_prev_top = tuple(sorted(((r, c-1), (r, c))))
                 if e_prev_top in current_edges: v_deg += 1
            
            # Up Edge incident to Vertex (r,c):
            # This is ((r-1, c), (r, c)). Left Edge of Cell (r-1, c).
            if r > r_min:
                 e_prev_left = tuple(sorted(((r-1, c), (r, c))))
                 if e_prev_left in current_edges: v_deg += 1
                 
            # Vertex (r,c) constraints
            # If internal ((r,c) not on boundary), deg must be 0 or 2.
            # If boundary, deg <= 2.
            
            # Is (r,c) on boundary of the REGION?
            # Region vertices are r_min..r_max+1, c_min..c_max+1.
            # Current vertex is (r, c).
            
            on_boundary = (r == r_min) or (c == c_min) # Or r_max/c_max, but loop hasn't reached them?
            # Wait, (r,c) is Top-Left of current cell.
            # Cells iterate r_min..r_max-1.
            # So r is in [r_min, r_max-1].
            # Vertex (r,c) is strictly on top or left boundary of region if r=r_min or c=c_min.
            # What about r_max? No, loop bounds.
            
            # My current vertex is Top-Left of (r,c).
            # Is it possible for it to be internal?
            # Yes, if r > r_min and c > c_min.
            
            if v_deg > 2: continue
            
            # Internal pruning
            is_internal = (r > r_min) and (c > c_min)
            # We strictly enforce internal nodes to be 0 or 2.
            if is_internal and v_deg == 1: continue

            next_set = current_edges.union(temp_edges)
            self._recursive_enumerate(next_r, next_c, r_min, r_max, c_min, c_max, next_set, clues, results)


    def _compute_signature(self, edges, r_min, r_max, c_min, c_max) -> Tuple[Optional[BoundarySignature], int]:
        uf = collections.defaultdict(int)
        parent = {}
        def find(i):
            if i not in parent: parent[i] = i
            if parent[i] != i: parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i = find(i); root_j = find(j)
            if root_i != root_j: parent[root_i] = root_j
            
        degree = collections.defaultdict(int)
        for u, v in edges:
            degree[u] += 1
            degree[v] += 1
            union(u, v)
        
        # We already pruned internal degree 1s in enumerate.
        # But we must check bounds again or just signatures.
        
        mapping = {}
        next_id = 1
        
        def get_id(node):
            nonlocal next_id
            if degree[node] == 0: return 0
            root = find(node)
            if root not in mapping:
                mapping[root] = next_id
                next_id += 1
            return mapping[root]

        top = tuple(get_id((r_min, c)) for c in range(c_min, c_max + 1))
        bottom = tuple(get_id((r_max, c)) for c in range(c_min, c_max + 1))
        left = tuple(get_id((r, c_min)) for r in range(r_min, r_max + 1))
        right = tuple(get_id((r, c_max)) for r in range(r_min, r_max + 1))
        
        exposed_roots = set()
        for c in range(c_min, c_max + 1): exposed_roots.add(find((r_min, c))); exposed_roots.add(find((r_max, c)))
        for r in range(r_min, r_max + 1): exposed_roots.add(find((r, c_min))); exposed_roots.add(find((r, c_max)))
        
        internal_loops = 0
        all_roots = set(find(u) for u in degree)
        for root in all_roots:
            if root not in exposed_roots:
                internal_loops += 1
                
        return BoundarySignature(top, bottom, left, right), internal_loops

    def _merge_horizontal_profiles(self, left_map, right_map, r_min, r_max, split_c) -> Dict[BoundarySignature, RegionProfile]:
        new_profiles = {}
        right_by_seam = collections.defaultdict(list)
        for sig_r, prof_r in right_map.items():
            pattern = tuple(1 if x > 0 else 0 for x in sig_r.left)
            right_by_seam[pattern].append(prof_r)
            
        for sig_l, prof_l in left_map.items():
            pattern = tuple(1 if x > 0 else 0 for x in sig_l.right)
            compatible_rights = right_by_seam.get(pattern, [])
            for prof_r in compatible_rights:
                self._merge_pair(prof_l, prof_r, "horizontal", new_profiles)
        return new_profiles

    def _merge_vertical_profiles(self, top_map, bottom_map, split_r, c_min, c_max) -> Dict[BoundarySignature, RegionProfile]:
        new_profiles = {}
        bottom_by_seam = collections.defaultdict(list)
        for sig_b, prof_b in bottom_map.items():
            pattern = tuple(1 if x > 0 else 0 for x in sig_b.top)
            bottom_by_seam[pattern].append(prof_b)
            
        for sig_t, prof_t in top_map.items():
            pattern = tuple(1 if x > 0 else 0 for x in sig_t.bottom)
            compatible_bottoms = bottom_by_seam.get(pattern, [])
            for prof_b in compatible_bottoms:
                self._merge_pair(prof_t, prof_b, "vertical", new_profiles)
        return new_profiles

    def _merge_pair(self, p1: RegionProfile, p2: RegionProfile, mode: str, result_map):
        # Stats
        merge_info = {
            "p1_count": p1.count,
            "p2_count": p2.count,
            "mode": mode,
            "pruned": False
        }
        
        s1 = p1.signature
        s2 = p2.signature
        
        # 1. Union-Find to merge components
        max_id_1 = 0
        all_ids_1 = set(s1.top + s1.bottom + s1.left + s1.right)
        if all_ids_1: max_id_1 = max(all_ids_1)
        
        offset = max_id_1
        
        def shift(t): return tuple(x + offset if x > 0 else 0 for x in t)
        s2_top = shift(s2.top)
        s2_bottom = shift(s2.bottom)
        s2_left = shift(s2.left)
        s2_right = shift(s2.right)
        
        parent = {}
        def find(i):
            if i not in parent: parent[i] = i
            if parent[i] != i: parent[i] = find(parent[i])
            return parent[i]
        def union(i, j):
            root_i = find(i); root_j = find(j)
            if root_i != root_j: parent[root_i] = root_j
            
        if mode == "horizontal":
            seam_len = len(s1.right)
            for k in range(seam_len):
                id1 = s1.right[k]
                id2 = s2_left[k]
                if id1 > 0 and id2 > 0:
                    union(id1, id2)
            
            if s1.top[-1] > 0 and s2_top[0] > 0: union(s1.top[-1], s2_top[0])
            if s1.bottom[-1] > 0 and s2_bottom[0] > 0: union(s1.bottom[-1], s2_bottom[0])
            
            new_top = s1.top[:-1] + s2_top
            new_bottom = s1.bottom[:-1] + s2_bottom
            new_left = s1.left
            new_right = s2_right
            
        else: # vertical
            seam_len = len(s1.bottom)
            for k in range(seam_len):
                id1 = s1.bottom[k]
                id2 = s2_top[k]
                if id1 > 0 and id2 > 0:
                    union(id1, id2)
                    
            if s1.left[-1] > 0 and s2_left[0] > 0: union(s1.left[-1], s2_left[0])
            if s1.right[-1] > 0 and s2_right[0] > 0: union(s1.right[-1], s2_right[0])
            
            new_top = s1.top
            new_bottom = s2_bottom
            new_left = s1.left[:-1] + s2_left
            new_right = s1.right[:-1] + s2_right

        mapping = {}
        next_id = 1
        def get_final_id(old_id):
            nonlocal next_id
            if old_id == 0: return 0
            root = find(old_id)
            if root not in mapping:
                mapping[root] = next_id
                next_id += 1
            return mapping[root]
            
        final_top = tuple(get_final_id(x) for x in new_top)
        final_bottom = tuple(get_final_id(x) for x in new_bottom)
        final_left = tuple(get_final_id(x) for x in new_left)
        final_right = tuple(get_final_id(x) for x in new_right)
        
        new_count = p1.count * p2.count
        
        boundary_roots = set()
        for x in final_top + final_bottom + final_left + final_right:
            if x > 0: boundary_roots.add(x) 
            
        active_roots_1 = set(find(x) for x in all_ids_1 if x > 0)
        all_ids_2_shifted = set(shift(s2.top + s2.bottom + s2.left + s2.right))
        active_roots_2 = set(find(x) for x in all_ids_2_shifted if x > 0)
        combined_roots = active_roots_1.union(active_roots_2)
        
        loops_formed = 0
        for root in combined_roots:
            nid = get_final_id(root)
            if nid not in boundary_roots:
                loops_formed += 1
                
        total_internal_loops = p1.internal_loops + p2.internal_loops + loops_formed
        
        has_boundary_paths = any(x > 0 for x in final_top + final_bottom + final_left + final_right)
        if total_internal_loops > 0 and has_boundary_paths:
            merge_info["pruned"] = True
            merge_info["reason"] = "Premature loop with open paths"
            self._merge_stats["merge_details"].append(merge_info)
            return 
        if total_internal_loops > 1:
            merge_info["pruned"] = True
            merge_info["reason"] = "Multiple disconnected loops"
            self._merge_stats["merge_details"].append(merge_info)
            return 
            
        new_sig = BoundarySignature(final_top, final_bottom, final_left, final_right)
        
        if new_sig not in result_map:
            result_map[new_sig] = RegionProfile(new_sig, count=0, internal_loops=total_internal_loops)
            # Merge samples - just picking one representative
            result_map[new_sig].sample_edges = p1.sample_edges.union(p2.sample_edges)
            
        prof = result_map[new_sig]
        # Check if we should add count or max?
        # Standard Counting DP: Add count.
        # But wait, edge_counts?
        # If we have multiple paths to same Signature, we sum their counts.
        # BUT edge_counts logic: `new_count` is p1.count * p2.count.
        # This is for ONE pair of profiles.
        # If multiple pairs lead to same new_sig, we add to existing prof.
        prof.count += new_count
        
        for e, c in p1.edge_counts.items():
            prof.edge_counts[e] += c * p2.count
        for e, c in p2.edge_counts.items():
            prof.edge_counts[e] += c * p1.count

    def _pick_best_move(self) -> Optional[Edge]:
        current_edges = set(self.game_state.graph.edges)
        best_edge = None
        best_score = -1.0
        
        for edge, prob in self._final_edge_probabilities.items():
            if edge in current_edges:
                score = 1.0 - prob
                if score > 0.5 and score > best_score:
                    best_score = score
                    best_edge = edge
            else:
                score = prob
                if score > 0.5 and score > best_score:
                    best_score = score
                    best_edge = edge
                    
        return best_edge
    
    def _get_all_potential_edges(self) -> List[Edge]:
        edges = []
        for r in range(self.rows + 1):
            for c in range(self.cols):
                edges.append(tuple(sorted(((r, c), (r, c+1)))))
        for r in range(self.rows):
            for c in range(self.cols + 1):
                edges.append(tuple(sorted(((r, c), (r+1, c)))))
        return edges
