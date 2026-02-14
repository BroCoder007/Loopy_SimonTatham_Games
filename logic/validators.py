"""
Game Validators
===============
Functions to validate moves and check game state conditions.
Uses DAA graph algorithms for connectivity and cycle checks.
"""

from daa.graph_algos import count_connected_components, has_cycle, dfs

def is_valid_move(graph, u, v, clues, check_global=False):
    """
    Check if toggling edge u-v is valid.
    Returns: (bool, reason)
    """
    edge = tuple(sorted((u, v)))
    adding = edge not in graph.edges
    
    if adding:
        # 1. Degree Constraint: No vertex > 2
        # If we add edge (u,v), degree of u and v increases by 1
        if graph.get_degree(u) >= 2:
            return False, "Vertex degree would exceed 2"
        if graph.get_degree(v) >= 2:
            return False, "Vertex degree would exceed 2"
            
        # 2. Clue Constraint
        if not check_clue_constraint(graph, u, v, clues, adding=True):
            return False, "Clue Violated"
            
        # 3. Premature Loop Prevention:
        # If adding (u, v) closes a cycle, allow it only when it completes a valid final loop.
        if is_reachable(graph, u, v):
            valid_cycle, reason = _is_valid_cycle_closure_after_add(graph, edge, clues)
            if not valid_cycle:
                return False, reason

    else:
        # Removing edge
        # Usually always allowed unless we want to enforce structure (e.g. don't break strict segments)
        # But standard Slitherlink allows backtracking.
        pass
            
    return True, "OK"


def _is_valid_cycle_closure_after_add(graph, edge, clues):
    """
    Validate whether a cycle-closing add is globally acceptable.
    Allowed only for a completed single-loop state.
    """
    future_edges = set(graph.edges)
    future_edges.add(edge)
    future_degree = _compute_future_degrees(graph, edge)

    active_vertices = [v for v in graph.vertices if future_degree.get(v, 0) > 0]
    if not active_vertices:
        return False, "Premature Loop (No active loop after closure)"

    degree_one_vertices = [v for v in active_vertices if future_degree.get(v, 0) == 1]
    if degree_one_vertices:
        return False, "Premature Loop (Leaves degree-1 vertices)"

    component_count, largest_component_size = _active_component_stats_via_dsu(future_edges)
    total_active_vertices = len(active_vertices)
    if component_count != 1 or largest_component_size != total_active_vertices:
        return False, "Premature Loop (Not a single connected active component)"

    if not _all_clues_exactly_satisfied(graph, edge, clues):
        return False, "Premature Loop (Clues not fully satisfied)"

    if _has_clue_adjacent_undecided_add(graph, future_edges, clues):
        return False, "Premature Loop (Clue-adjacent undecided edges remain)"

    return True, "OK"


def _compute_future_degrees(graph, edge):
    """
    Degree map after hypothetically adding `edge` without mutating graph.
    """
    u, v = edge
    degrees = {node: graph.get_degree(node) for node in graph.vertices}
    degrees[u] = degrees.get(u, 0) + 1
    degrees[v] = degrees.get(v, 0) + 1
    return degrees


def _all_clues_exactly_satisfied(graph, added_edge, clues):
    """
    Check clue equality in the post-add state.
    """
    for cell, clue in clues.items():
        count = count_edges_around_cell(graph, cell)
        if _edge_touches_cell(added_edge, cell):
            count += 1
        if count != clue:
            return False
    return True


def _has_clue_adjacent_undecided_add(graph, future_edges, clues):
    """
    True when any missing edge around a clue cell could still be added
    without immediately exceeding that clue.
    """
    for cell, clue in clues.items():
        current_count = 0
        cell_edges = _cell_edges(cell)
        for e in cell_edges:
            if e in future_edges:
                current_count += 1

        for e in cell_edges:
            if e in future_edges:
                continue
            # If adding e could keep clue not exceeded, clue state is still mutable.
            if current_count + 1 <= clue:
                return True
    return False


def _cell_edges(cell):
    r, c = cell
    return [
        tuple(sorted(((r, c), (r, c + 1)))),
        tuple(sorted(((r + 1, c), (r + 1, c + 1)))),
        tuple(sorted(((r, c), (r + 1, c)))),
        tuple(sorted(((r, c + 1), (r + 1, c + 1)))),
    ]


def _edge_touches_cell(edge, cell):
    return edge in _cell_edges(cell)


def _active_component_stats_via_dsu(edges):
    """
    Returns (component_count, largest_component_size) over active vertices.
    """
    if not edges:
        return 0, 0

    dsu = _DSU()
    vertices = set()
    for u, v in edges:
        vertices.add(u)
        vertices.add(v)
        dsu.union(u, v)

    component_size = {}
    for node in vertices:
        root = dsu.find(node)
        component_size[root] = component_size.get(root, 0) + 1

    sizes = list(component_size.values())
    return len(sizes), (max(sizes) if sizes else 0)


class _DSU:
    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, x):
        if x not in self.parent:
            self.parent[x] = x
            self.rank[x] = 0
            return x
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra = self.find(a)
        rb = self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

def is_reachable(graph, start, target):
    """
    Check if target is reachable from start using currently active edges.
    """
    # Simple BFS/DFS on graph.adj_list
    visited = set()
    stack = [start]
    visited.add(start)
    
    while stack:
        u = stack.pop()
        if u == target:
            return True
        for v in graph.adj_list.get(u, []):
            if v not in visited:
                visited.add(v)
                stack.append(v)
    return False

def check_clue_constraint(graph, u, v, clues, adding):
    """
    Check if adding/removing edge u-v violates any adjacent clues.
    """
    r1, c1 = u
    r2, c2 = v
    cells_to_check = []
    
    # Identify adjacent cells
    if r1 == r2: # Horizontal
        c_min = min(c1, c2)
        if r1 > 0: cells_to_check.append((r1-1, c_min))
        if r1 < graph.rows: cells_to_check.append((r1, c_min))
    else: # Vertical
        r_min = min(r1, r2)
        if c1 > 0: cells_to_check.append((r_min, c1-1))
        if c1 < graph.cols: cells_to_check.append((r_min, c1))
        
    for cell in cells_to_check:
        if cell in clues:
            current_edges = count_edges_around_cell(graph, cell)
            potential_edges = current_edges + (1 if adding else -1)
            
            if potential_edges > clues[cell]:
                return False
    return True

def count_edges_around_cell(graph, cell):
    r, c = cell
    edges = [
        tuple(sorted(((r, c), (r, c+1)))),
        tuple(sorted(((r+1, c), (r+1, c+1)))),
        tuple(sorted(((r, c), (r+1, c)))),
        tuple(sorted(((r, c+1), (r+1, c+1))))
    ]
    count = 0
    for e in edges:
        if e in graph.edges:
            count += 1
    return count

def check_win_condition(graph, clues):
    """
    Check if the game is won.
    Conditions:
    1. All clues satisfied.
    2. Single connected loop (1 component, all degrees=2).
    """
    # 1. Clues
    for cell, val in clues.items():
        if count_edges_around_cell(graph, cell) != val:
            return False, "Clues not satisfied"
            
    # 2. Loop Structure
    active_vertices = [v for v in graph.vertices if graph.get_degree(v) > 0]
    if not active_vertices:
        return False, "Empty board"
        
    for v in active_vertices:
        if graph.get_degree(v) != 2:
            return False, "Not a closed loop"
            
    # 3. Connectivity
    num_components = count_connected_components(graph.adj_list, graph.vertices)
    if num_components != 1:
        return False, "Multiple loops detected"
        
    return True, "Winner"
