"""
Pure Spatial Divide & Conquer Solver
===================================

Design constraints implemented here:
- No cross-calls to other solver classes
- No backup chaining
- No DP state compression
- No beam search
- No randomness
- Deterministic recursion + merge order
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict, FrozenSet, List, Optional, Set, Tuple

from logic.graph import Graph
from logic.solvers.solver_errors import (
    ControlledStateExplosionError,
    STATE_SPACE_EXPLOSION_MESSAGE,
    resolve_safe_limit,
)
from logic.solvers.solver_interface import AbstractSolver, HintPayload
from logic.validators import check_win_condition, is_valid_move

Move = Tuple[Tuple[int, int], Tuple[int, int]]
Region = Tuple[int, int, int, int]  # (r_min, c_min, r_max, c_max) on cell coordinates


@dataclass(frozen=True)
class RegionConfiguration:
    region: Region
    edges: FrozenSet[Move]
    boundary_signature: Tuple[Tuple[Move, int], ...]
    component_snapshot: Tuple[Tuple[Tuple[int, int], int, int], ...]


class _UnionFind:
    def __init__(self):
        self.parent: Dict[Tuple[int, int], Tuple[int, int]] = {}
        self.rank: Dict[Tuple[int, int], int] = {}

    def add(self, node: Tuple[int, int]) -> None:
        if node not in self.parent:
            self.parent[node] = node
            self.rank[node] = 0

    def find(self, node: Tuple[int, int]) -> Tuple[int, int]:
        parent = self.parent[node]
        if parent != node:
            self.parent[node] = self.find(parent)
        return self.parent[node]

    def union(self, a: Tuple[int, int], b: Tuple[int, int]) -> bool:
        """
        Returns False when `a` and `b` are already connected (cycle edge).
        """
        self.add(a)
        self.add(b)
        root_a = self.find(a)
        root_b = self.find(b)
        if root_a == root_b:
            return False

        rank_a = self.rank[root_a]
        rank_b = self.rank[root_b]
        if rank_a < rank_b:
            self.parent[root_a] = root_b
        elif rank_a > rank_b:
            self.parent[root_b] = root_a
        else:
            self.parent[root_b] = root_a
            self.rank[root_a] += 1
        return True


class DivideConquerSolver(AbstractSolver):
    def __init__(self, game_state: Any):
        self.game_state = game_state
        self._last_explanation: str = ""
        self.BASE_CASE_SIZE = 3
        self.SAFE_LIMIT = resolve_safe_limit(game_state, "dnc_safe_limit")
        self._state_explosion_detected = False

        # Visualization exposure
        self.recursion_depth: int = 0
        self.current_region_id: str = ""
        self.merge_stage: str = ""

    # ---- Public solver interface ------------------------------------------
    def decide_move(self) -> Tuple[List[Any], Optional[Move]]:
        if self._state_explosion_detected:
            self._publish_state_explosion_message()
            return [], None
        move = self.solve()
        return [], move

    def solve(self, board: Any = None):
        if self._state_explosion_detected:
            self._publish_state_explosion_message()
            return None
        try:
            move, reason = self._run_divide_and_conquer()
            if move is not None:
                self._last_explanation = f"Using Divide & Conquer Strategy: {reason}"
                return move
            self._last_explanation = f"Using Divide & Conquer Strategy: {reason}"
        except ControlledStateExplosionError:
            self._handle_state_explosion()
        return None

    def generate_hint(self, board: Any = None) -> HintPayload:
        if self._state_explosion_detected:
            self._publish_state_explosion_message()
            return {
                "move": None,
                "strategy": "Divide & Conquer",
                "explanation": STATE_SPACE_EXPLOSION_MESSAGE,
            }

        is_human_turn = (
            self.game_state.game_mode in ["vs_cpu", "expert"]
            and self.game_state.turn == "Player 1 (Human)"
        ) or (self.game_state.game_mode not in ["vs_cpu", "expert"])

        if not is_human_turn:
            return {
                "move": None,
                "strategy": "Divide & Conquer",
                "explanation": "Hints are only available during your turn.",
            }

        try:
            move, reason = self._run_divide_and_conquer()
            return {
                "move": move,
                "strategy": "Divide & Conquer",
                "explanation": f"Strategy: Divide & Conquer\nReason: {reason}",
            }
        except ControlledStateExplosionError:
            self._handle_state_explosion()
            return {
                "move": None,
                "strategy": "Divide & Conquer",
                "explanation": STATE_SPACE_EXPLOSION_MESSAGE,
            }

    def register_move(self, move: Move):
        self.game_state.last_cpu_move_info = {
            "move": move,
            "explanation": self._last_explanation,
            "strategy": "Divide & Conquer",
            "recursion_depth": self.recursion_depth,
            "current_region_id": self.current_region_id,
            "merge_stage": self.merge_stage,
        }

    def explain_last_move(self) -> str:
        return self._last_explanation

    def _publish_state_explosion_message(self) -> None:
        self._last_explanation = STATE_SPACE_EXPLOSION_MESSAGE
        self.game_state.message = STATE_SPACE_EXPLOSION_MESSAGE

    def _handle_state_explosion(self) -> None:
        self._state_explosion_detected = True
        self._publish_state_explosion_message()

    def _assert_config_limit(self, count: int, context: str) -> None:
        if count > self.SAFE_LIMIT:
            raise ControlledStateExplosionError(
                safe_limit=self.SAFE_LIMIT,
                observed=count,
                context=context,
            )

    # ---- Core D&C ----------------------------------------------------------
    def _run_divide_and_conquer(self) -> Tuple[Optional[Move], str]:
        if not self.game_state.clues:
            return None, "No clues are present; no deterministic forced move exists."

        full_region = (0, 0, self.game_state.rows - 1, self.game_state.cols - 1)
        full_configs = self.solve_region(full_region, depth=0)
        self._assert_config_limit(len(full_configs), "dnc_full_region_configs")
        if not full_configs:
            return None, "No globally valid single-loop completion exists for this board."

        chosen_cfg = self._select_deterministic_configuration(full_configs)
        move, reason = self._select_move_from_configuration(chosen_cfg, len(full_configs))
        if move is None:
            return None, reason

        self._log_trace(
            depth=0,
            region=full_region,
            merge_stage="forced_move_selection",
            explanation=reason,
            move=move,
        )
        return move, reason

    def solve_region(self, region: Region, depth: int = 0) -> List[RegionConfiguration]:
        """
        solve_region(region):
        - base: solve_region_base(region)
        - recursive: split into 4 quadrants and merge in fixed order
        """
        if not self._region_valid(region):
            return []

        self.recursion_depth = depth
        self.current_region_id = self._region_id(region)

        r_min, c_min, r_max, c_max = region
        height = r_max - r_min + 1
        width = c_max - c_min + 1
        self._log_trace(depth, region, "enter", f"Entering region {region}.")

        if height <= self.BASE_CASE_SIZE and width <= self.BASE_CASE_SIZE:
            self.merge_stage = "base_case"
            base = self._solve_region_base(region, depth)
            base = self._dedupe_configs(base)
            self._assert_config_limit(len(base), f"dnc_base_{self._region_id(region)}")
            self._log_trace(
                depth,
                region,
                "base_case_done",
                f"Base-case enumeration produced {len(base)} configurations.",
            )
            return base

        mid_r = r_min + (height // 2)
        mid_c = c_min + (width // 2)
        tl = (r_min, c_min, mid_r - 1, mid_c - 1)
        tr = (r_min, mid_c, mid_r - 1, c_max)
        bl = (mid_r, c_min, r_max, mid_c - 1)
        br = (mid_r, mid_c, r_max, c_max)

        tl_cfg = self.solve_region(tl, depth + 1) if self._region_valid(tl) else []
        tr_cfg = self.solve_region(tr, depth + 1) if self._region_valid(tr) else []
        bl_cfg = self.solve_region(bl, depth + 1) if self._region_valid(bl) else []
        br_cfg = self.solve_region(br, depth + 1) if self._region_valid(br) else []
        self._assert_config_limit(len(tl_cfg), f"dnc_region_{self._region_id(tl)}")
        self._assert_config_limit(len(tr_cfg), f"dnc_region_{self._region_id(tr)}")
        self._assert_config_limit(len(bl_cfg), f"dnc_region_{self._region_id(bl)}")
        self._assert_config_limit(len(br_cfg), f"dnc_region_{self._region_id(br)}")

        top_half = self._merge_horizontal(
            tl_cfg,
            tr_cfg,
            mid_c=mid_c,
            r_min=r_min,
            r_max=mid_r - 1,
            depth=depth,
            merge_stage="merge_horizontal_top",
        )
        self._assert_config_limit(len(top_half), f"dnc_top_half_{self._region_id(region)}")
        bottom_half = self._merge_horizontal(
            bl_cfg,
            br_cfg,
            mid_c=mid_c,
            r_min=mid_r,
            r_max=r_max,
            depth=depth,
            merge_stage="merge_horizontal_bottom",
        )
        self._assert_config_limit(len(bottom_half), f"dnc_bottom_half_{self._region_id(region)}")

        merged = self._merge_vertical(
            top_half,
            bottom_half,
            mid_r=mid_r,
            c_min=c_min,
            c_max=c_max,
            depth=depth,
            merge_stage="merge_vertical_full",
            allow_closed_loop=self._is_full_board_region(region),
        )

        merged = self._dedupe_configs(merged)
        self._assert_config_limit(len(merged), f"dnc_merged_{self._region_id(region)}")
        if self._is_full_board_region(region):
            merged = self._global_validate(merged)
            self._assert_config_limit(len(merged), f"dnc_global_validated_{self._region_id(region)}")
            self._log_trace(
                depth,
                region,
                "global_validation",
                f"Global validation retained {len(merged)} configurations.",
            )

        return merged

    # ---- Base case ---------------------------------------------------------
    def _solve_region_base(self, region: Region, depth: int) -> List[RegionConfiguration]:
        edges = self._region_edges(region)
        edge_count = len(edges)
        is_full_region = self._is_full_board_region(region)

        clue_cells = [cell for cell in self._region_cells(region) if cell in self.game_state.clues]
        cell_edges: Dict[Tuple[int, int], List[Move]] = {cell: self._cell_edges(cell) for cell in clue_cells}
        edge_to_clue_cells: Dict[Move, List[Tuple[int, int]]] = defaultdict(list)
        for cell in sorted(cell_edges.keys()):
            c_edges = cell_edges[cell]
            for edge in c_edges:
                edge_to_clue_cells[edge].append(cell)

        vertices = self._region_vertices(region)
        region_edge_set = set(edges)
        constrained_vertices = self._constrained_vertices(region, region_edge_set)
        if is_full_region:
            constrained_vertices = vertices
        incident_edges: Dict[Tuple[int, int], List[Move]] = {v: [] for v in vertices}
        for edge in edges:
            u, v = edge
            incident_edges[u].append(edge)
            incident_edges[v].append(edge)

        degree: Dict[Tuple[int, int], int] = {v: 0 for v in vertices}
        remaining_incident: Dict[Tuple[int, int], int] = {v: len(incident_edges[v]) for v in vertices}
        adjacency: Dict[Tuple[int, int], Set[Tuple[int, int]]] = {v: set() for v in vertices}
        on_edges: Set[Move] = set()

        clue_on: Dict[Tuple[int, int], int] = {cell: 0 for cell in clue_cells}
        clue_unknown: Dict[Tuple[int, int], int] = {cell: 4 for cell in clue_cells}

        solutions: List[RegionConfiguration] = []

        def vertex_feasible(vertex: Tuple[int, int]) -> bool:
            deg = degree[vertex]
            rem = remaining_incident[vertex]
            if deg > 2:
                return False
            if vertex not in constrained_vertices:
                return True
            # Internal vertices in partial regions (and all vertices in full region)
            # must be able to end at degree 0 or 2.
            can_be_zero = deg == 0
            can_be_two = deg <= 2 and (deg + rem) >= 2
            return can_be_zero or can_be_two

        def clue_feasible(cell: Tuple[int, int]) -> bool:
            clue = self.game_state.clues[cell]
            on = clue_on[cell]
            unknown = clue_unknown[cell]
            if on > clue:
                return False
            if on + unknown < clue:
                return False
            return True

        def creates_cycle(u: Tuple[int, int], v: Tuple[int, int]) -> bool:
            if not adjacency[u] or not adjacency[v]:
                return False
            stack = [u]
            visited = {u}
            while stack:
                cur = stack.pop()
                if cur == v:
                    return True
                for nxt in adjacency[cur]:
                    if nxt not in visited:
                        visited.add(nxt)
                        stack.append(nxt)
            return False

        def backtrack(idx: int) -> None:
            if idx == edge_count:
                for cell in clue_cells:
                    if clue_on[cell] != self.game_state.clues[cell]:
                        return
                for vertex in sorted(constrained_vertices):
                    if degree[vertex] == 1:
                        return

                boundary_signature = self._boundary_signature(region, on_edges)
                solutions.append(
                    RegionConfiguration(
                        region=region,
                        edges=frozenset(on_edges),
                        boundary_signature=boundary_signature,
                        component_snapshot=self._component_snapshot(region, on_edges),
                    )
                )
                self._assert_config_limit(len(solutions), f"dnc_base_append_{self._region_id(region)}")
                return

            edge = edges[idx]
            u, v = edge

            # Deterministic branching: OFF first, then ON.
            for edge_on in (False, True):
                if edge_on and (degree[u] >= 2 or degree[v] >= 2):
                    continue

                cycle_formed = False
                if edge_on:
                    cycle_formed = creates_cycle(u, v)
                    if cycle_formed and not is_full_region:
                        continue

                affected_clues = edge_to_clue_cells.get(edge, [])

                # Apply edge
                remaining_incident[u] -= 1
                remaining_incident[v] -= 1
                if edge_on:
                    on_edges.add(edge)
                    degree[u] += 1
                    degree[v] += 1
                    adjacency[u].add(v)
                    adjacency[v].add(u)

                for cell in affected_clues:
                    clue_unknown[cell] -= 1
                    if edge_on:
                        clue_on[cell] += 1

                local_ok = True
                for cell in affected_clues:
                    if not clue_feasible(cell):
                        local_ok = False
                        break
                if local_ok and (not vertex_feasible(u) or not vertex_feasible(v)):
                    local_ok = False

                if local_ok:
                    backtrack(idx + 1)

                # Undo edge
                for cell in affected_clues:
                    if edge_on:
                        clue_on[cell] -= 1
                    clue_unknown[cell] += 1

                if edge_on:
                    adjacency[u].remove(v)
                    adjacency[v].remove(u)
                    degree[u] -= 1
                    degree[v] -= 1
                    on_edges.remove(edge)
                remaining_incident[u] += 1
                remaining_incident[v] += 1

        self._log_trace(
            depth,
            region,
            "base_case_enumeration",
            f"Enumerating {edge_count} region edges with clue/degree/cycle pruning.",
        )
        backtrack(0)
        return solutions

    # ---- Merge phase -------------------------------------------------------
    def _merge_horizontal(
        self,
        left_regions: List[RegionConfiguration],
        right_regions: List[RegionConfiguration],
        mid_c: int,
        r_min: int,
        r_max: int,
        depth: int,
        merge_stage: str,
    ) -> List[RegionConfiguration]:
        self.merge_stage = merge_stage
        self._assert_config_limit(len(left_regions), f"dnc_merge_h_left_{merge_stage}")
        self._assert_config_limit(len(right_regions), f"dnc_merge_h_right_{merge_stage}")
        if not left_regions and not right_regions:
            return []
        if not left_regions:
            return right_regions
        if not right_regions:
            return left_regions

        seam_edges = [self._norm_edge((r, mid_c), (r + 1, mid_c)) for r in range(r_min, r_max + 1)]
        seam_edges = sorted(seam_edges)

        right_by_sig: Dict[Tuple[int, ...], List[RegionConfiguration]] = defaultdict(list)
        for right_cfg in right_regions:
            right_by_sig[self._seam_signature(right_cfg.edges, seam_edges)].append(right_cfg)

        merged: List[RegionConfiguration] = []
        for left_cfg in left_regions:
            signature = self._seam_signature(left_cfg.edges, seam_edges)
            for right_cfg in right_by_sig.get(signature, []):
                candidate = self._merge_pair(
                    left_cfg,
                    right_cfg,
                    allow_closed_loop=False,
                )
                if candidate is not None:
                    merged.append(candidate)
                    self._assert_config_limit(len(merged), f"dnc_merge_h_accum_{merge_stage}")

        merged = self._dedupe_configs(merged)
        self._assert_config_limit(len(merged), f"dnc_merge_h_result_{merge_stage}")
        self._log_trace(
            depth,
            self._merged_region_bounds(left_regions[0].region, right_regions[0].region),
            merge_stage,
            f"Horizontal merge kept {len(merged)} states.",
        )
        return merged

    def _merge_vertical(
        self,
        top_regions: List[RegionConfiguration],
        bottom_regions: List[RegionConfiguration],
        mid_r: int,
        c_min: int,
        c_max: int,
        depth: int,
        merge_stage: str,
        allow_closed_loop: bool,
    ) -> List[RegionConfiguration]:
        self.merge_stage = merge_stage
        self._assert_config_limit(len(top_regions), f"dnc_merge_v_top_{merge_stage}")
        self._assert_config_limit(len(bottom_regions), f"dnc_merge_v_bottom_{merge_stage}")
        if not top_regions and not bottom_regions:
            return []
        if not top_regions:
            return bottom_regions
        if not bottom_regions:
            return top_regions

        seam_edges = [self._norm_edge((mid_r, c), (mid_r, c + 1)) for c in range(c_min, c_max + 1)]
        seam_edges = sorted(seam_edges)

        bottom_by_sig: Dict[Tuple[int, ...], List[RegionConfiguration]] = defaultdict(list)
        for bottom_cfg in bottom_regions:
            bottom_by_sig[self._seam_signature(bottom_cfg.edges, seam_edges)].append(bottom_cfg)

        merged: List[RegionConfiguration] = []
        for top_cfg in top_regions:
            signature = self._seam_signature(top_cfg.edges, seam_edges)
            for bottom_cfg in bottom_by_sig.get(signature, []):
                candidate = self._merge_pair(
                    top_cfg,
                    bottom_cfg,
                    allow_closed_loop=allow_closed_loop,
                )
                if candidate is not None:
                    merged.append(candidate)
                    self._assert_config_limit(len(merged), f"dnc_merge_v_accum_{merge_stage}")

        merged = self._dedupe_configs(merged)
        self._assert_config_limit(len(merged), f"dnc_merge_v_result_{merge_stage}")
        self._log_trace(
            depth,
            self._merged_region_bounds(top_regions[0].region, bottom_regions[0].region),
            merge_stage,
            f"Vertical merge kept {len(merged)} states.",
        )
        return merged

    def _merge_pair(
        self,
        first: RegionConfiguration,
        second: RegionConfiguration,
        allow_closed_loop: bool,
    ) -> Optional[RegionConfiguration]:
        merged_region = self._merged_region_bounds(first.region, second.region)
        merged_edges = frozenset(set(first.edges) | set(second.edges))

        if not self._validate_region_configuration(
            region=merged_region,
            edges=merged_edges,
            allow_closed_loop=allow_closed_loop,
        ):
            return None

        return RegionConfiguration(
            region=merged_region,
            edges=merged_edges,
            boundary_signature=self._boundary_signature(merged_region, merged_edges),
            component_snapshot=self._component_snapshot(merged_region, merged_edges),
        )

    def _validate_region_configuration(
        self,
        region: Region,
        edges: FrozenSet[Move],
        allow_closed_loop: bool,
    ) -> bool:
        degree: Dict[Tuple[int, int], int] = defaultdict(int)
        uf = _UnionFind()
        cycle_detected = False

        for edge in sorted(edges):
            u, v = edge
            degree[u] += 1
            degree[v] += 1
            if degree[u] > 2 or degree[v] > 2:
                return False
            if not uf.union(u, v):
                cycle_detected = True
                if not allow_closed_loop:
                    return False

        if not allow_closed_loop and cycle_detected:
            return False

        constrained_vertices = self._constrained_vertices(region, set(self._region_edges(region)))
        if allow_closed_loop and self._is_full_board_region(region):
            constrained_vertices = self._region_vertices(region)
        for vertex in sorted(constrained_vertices):
            if degree.get(vertex, 0) == 1:
                return False

        return True

    # ---- Final validation and move selection -------------------------------
    def _global_validate(self, configs: List[RegionConfiguration]) -> List[RegionConfiguration]:
        self._assert_config_limit(len(configs), "dnc_global_validate_input")
        valid: List[RegionConfiguration] = []
        for cfg in configs:
            graph = Graph(self.game_state.rows, self.game_state.cols)
            for edge in sorted(cfg.edges):
                graph.add_edge(edge[0], edge[1])
            won, _ = check_win_condition(graph, self.game_state.clues)
            if won:
                valid.append(cfg)
                self._assert_config_limit(len(valid), "dnc_global_validate_valid")
        return self._dedupe_configs(valid)

    def _select_deterministic_configuration(self, full_configs: List[RegionConfiguration]) -> RegionConfiguration:
        return min(full_configs, key=self._config_sort_key)

    def _select_move_from_configuration(
        self,
        full_config: RegionConfiguration,
        config_count: int,
    ) -> Tuple[Optional[Move], str]:
        all_edges = self._all_board_edges()
        current_edges = set(self.game_state.graph.edges)
        target_edges = set(full_config.edges)

        # Deterministic preference: add required target edges first.
        for edge in all_edges:
            if edge in target_edges and edge not in current_edges:
                valid, _ = is_valid_move(
                    self.game_state.graph,
                    edge[0],
                    edge[1],
                    self.game_state.clues,
                )
                if valid:
                    return edge, (
                        f"Selected lexicographically minimal full configuration among {config_count} valid states; "
                        f"next target edge {edge} must be added."
                    )

        # Then remove edges not in the selected target.
        for edge in all_edges:
            if edge in current_edges and edge not in target_edges:
                return edge, (
                    f"Selected lexicographically minimal full configuration among {config_count} valid states; "
                    f"edge {edge} is not in target and should be removed."
                )

        return None, "Board already matches selected deterministic full configuration."

    # ---- Geometry helpers --------------------------------------------------
    def _region_valid(self, region: Region) -> bool:
        r_min, c_min, r_max, c_max = region
        return r_min <= r_max and c_min <= c_max

    def _region_id(self, region: Region) -> str:
        r_min, c_min, r_max, c_max = region
        return f"{r_min}-{r_max},{c_min}-{c_max}"

    def _is_full_board_region(self, region: Region) -> bool:
        return region == (0, 0, self.game_state.rows - 1, self.game_state.cols - 1)

    def _merged_region_bounds(self, a: Region, b: Region) -> Region:
        return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))

    def _norm_edge(self, u: Tuple[int, int], v: Tuple[int, int]) -> Move:
        return tuple(sorted((u, v)))

    def _cell_edges(self, cell: Tuple[int, int]) -> List[Move]:
        r, c = cell
        return [
            self._norm_edge((r, c), (r, c + 1)),
            self._norm_edge((r + 1, c), (r + 1, c + 1)),
            self._norm_edge((r, c), (r + 1, c)),
            self._norm_edge((r, c + 1), (r + 1, c + 1)),
        ]

    def _region_cells(self, region: Region) -> List[Tuple[int, int]]:
        r_min, c_min, r_max, c_max = region
        return [(r, c) for r in range(r_min, r_max + 1) for c in range(c_min, c_max + 1)]

    def _region_edges(self, region: Region) -> List[Move]:
        edges: Set[Move] = set()
        for cell in self._region_cells(region):
            for edge in self._cell_edges(cell):
                edges.add(edge)
        return sorted(edges)

    def _region_vertices(self, region: Region) -> Set[Tuple[int, int]]:
        vertices: Set[Tuple[int, int]] = set()
        for edge in self._region_edges(region):
            vertices.add(edge[0])
            vertices.add(edge[1])
        return vertices

    def _region_boundary_edges(self, region: Region) -> List[Move]:
        r_min, c_min, r_max, c_max = region
        edges: List[Move] = []

        for c in range(c_min, c_max + 1):
            edges.append(self._norm_edge((r_min, c), (r_min, c + 1)))
            edges.append(self._norm_edge((r_max + 1, c), (r_max + 1, c + 1)))
        for r in range(r_min, r_max + 1):
            edges.append(self._norm_edge((r, c_min), (r + 1, c_min)))
            edges.append(self._norm_edge((r, c_max + 1), (r + 1, c_max + 1)))

        return sorted(set(edges))

    def _region_boundary_vertices(self, region: Region) -> Set[Tuple[int, int]]:
        r_min, c_min, r_max, c_max = region
        vertices: Set[Tuple[int, int]] = set()
        for r in range(r_min, r_max + 2):
            vertices.add((r, c_min))
            vertices.add((r, c_max + 1))
        for c in range(c_min, c_max + 2):
            vertices.add((r_min, c))
            vertices.add((r_max + 1, c))
        return vertices

    def _constrained_vertices(
        self,
        region: Region,
        region_edges: Set[Move],
    ) -> Set[Tuple[int, int]]:
        constrained: Set[Tuple[int, int]] = set()
        for vertex in sorted(self._region_vertices(region)):
            board_incident = self._incident_board_edges(vertex)
            if board_incident and board_incident.issubset(region_edges):
                constrained.add(vertex)
        return constrained

    def _incident_board_edges(self, vertex: Tuple[int, int]) -> Set[Move]:
        r, c = vertex
        rows = self.game_state.rows
        cols = self.game_state.cols
        edges: Set[Move] = set()
        if r > 0:
            edges.add(self._norm_edge((r - 1, c), (r, c)))
        if r < rows:
            edges.add(self._norm_edge((r, c), (r + 1, c)))
        if c > 0:
            edges.add(self._norm_edge((r, c - 1), (r, c)))
        if c < cols:
            edges.add(self._norm_edge((r, c), (r, c + 1)))
        return edges

    def _boundary_signature(self, region: Region, edges: Set[Move] | FrozenSet[Move]) -> Tuple[Tuple[Move, int], ...]:
        return tuple((edge, 1 if edge in edges else 0) for edge in self._region_boundary_edges(region))

    def _seam_signature(self, edges: Set[Move] | FrozenSet[Move], seam_edges: List[Move]) -> Tuple[int, ...]:
        return tuple(1 if edge in edges else 0 for edge in seam_edges)

    def _count_edges_around_cell(self, cell: Tuple[int, int], edges: FrozenSet[Move]) -> int:
        count = 0
        for edge in self._cell_edges(cell):
            if edge in edges:
                count += 1
        return count

    def _all_board_edges(self) -> List[Move]:
        rows = self.game_state.rows
        cols = self.game_state.cols
        edges: List[Move] = []

        for r in range(rows + 1):
            for c in range(cols):
                edges.append(self._norm_edge((r, c), (r, c + 1)))
        for r in range(rows):
            for c in range(cols + 1):
                edges.append(self._norm_edge((r, c), (r + 1, c)))
        return sorted(edges)

    def _dedupe_configs(self, configs: List[RegionConfiguration]) -> List[RegionConfiguration]:
        unique: Dict[FrozenSet[Move], RegionConfiguration] = {}
        ordered: List[RegionConfiguration] = []
        for cfg in configs:
            if cfg.edges not in unique:
                unique[cfg.edges] = cfg
                ordered.append(cfg)
        return ordered

    def _config_sort_key(self, cfg: RegionConfiguration) -> Tuple[Move, ...]:
        return tuple(sorted(cfg.edges))

    def _component_snapshot(
        self,
        region: Region,
        edges: Set[Move] | FrozenSet[Move],
    ) -> Tuple[Tuple[Tuple[int, int], int, int], ...]:
        """
        Snapshot for visualization/merge introspection:
        (boundary_vertex, component_id, degree)
        """
        uf = _UnionFind()
        degree: Dict[Tuple[int, int], int] = defaultdict(int)

        for edge in sorted(edges):
            u, v = edge
            uf.union(u, v)
            degree[u] += 1
            degree[v] += 1

        boundary_vertices = sorted(self._region_boundary_vertices(region))
        for vertex in boundary_vertices:
            uf.add(vertex)

        roots = sorted({uf.find(v) for v in boundary_vertices})
        root_to_id = {root: idx for idx, root in enumerate(roots)}

        snapshot: List[Tuple[Tuple[int, int], int, int]] = []
        for vertex in boundary_vertices:
            root = uf.find(vertex)
            snapshot.append((vertex, root_to_id[root], degree.get(vertex, 0)))
        return tuple(snapshot)

    def _log_trace(
        self,
        depth: int,
        region: Region,
        merge_stage: str,
        explanation: str,
        move: Optional[Move] = None,
    ) -> None:
        self.recursion_depth = depth
        self.current_region_id = self._region_id(region)
        self.merge_stage = merge_stage
        try:
            from logic.execution_trace import log_execution_step

            log_execution_step(
                strategy_name="Divide & Conquer",
                move=move,
                explanation=explanation,
                recursion_depth=depth,
                region_id=self.current_region_id,
                merge_info=merge_stage,
            )
        except Exception:
            pass
