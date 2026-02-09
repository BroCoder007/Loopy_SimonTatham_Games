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
            
        # 3. Premature Loop Prevention
        # We need to temporarily add the edge to check for cycles
        # Note: We can't actually modify the graph object passed in, 
        # but we can simulate connectivity checks.
        
        # If u and v are ALREADY connected, adding (u,v) creates a cycle.
        # Check if there is a path between u and v
        # We can use DFS/BFS from u to see if we can reach v
        # If yes, adding edge (u,v) closes a loop.
        
        # Exception: It is allowed IF this loop satisfies the Win Condition (all clues matched)
        # But we can't easily check 'all clues matched' for the FUTURE state without modifying.
        
        # Optimization: Use DAA `dfs` or `bfs`
        # We need to construct a temporary view of adjacency or just pass current graph
        # and see if v is reachable from u.
        
        if is_reachable(graph, u, v):
            # A cycle is being formed.
            # It is ONLY valid if this cycle completes the game.
            # To check this efficiently, we check:
            # - Are all OTHER active vertices part of this cycle? (Single Loop condition)
            # - Are all clues satisfied? (This is hard to check without applying)
            
            # Heuristic for DAA Project:
            # "Premature Loop" = A small loop that doesn't use all active segments.
            # If we close a loop, verify if it's the "Main" loop.
            
            # Let's count active edges. If the closed loop length < current active edges + 1,
            # it implies there are disjoint edges elsewhere, so this loop isolates them.
            # That's a "Premature Loop" (since we want a Single Loop).
            
            # Get path length? Overkill.
            # Simple check: If graph is currently a single connected component (of edges),
            # closing a loop is fine ONLY IF it connects the 'ends' of the line.
            # But Slitherlink can have multiple segments during play.
            
            # STRICT RULE: A move is invalid if it creates a CLOSED LOOP that does not
            # include ALL currently active edges.
            # Actually, even simpler:
            # If we create a closed loop, we must have NO loose ends left anywhere else.
            
            # Let's trust the validator to be called.
            # For now, we will ALLOW closing loops, but the GameState win checker
            # will tell us if we won.
            # BUT, the prompt asks to "Prevent Premature Loop".
            # So: If forming a cycle, check if (num_edges_in_cycle == total_edges_after_move).
            # If not, it means we have edges outside the cycle -> Invalid Premature Loop.
            
            # To do this efficiently:
            # 1. Check reachability (u -> v)
            # 2. If reachable, we are closing a loop.
            # 3. Check if any other component exists.
            #    If we are closing the loop, the new component count should be 1.
            #    Actually, if u and v are in the same component, adding edge keeps count same.
            #    If they were disjoint, count decreases.
            
            # WAIT. If u and v are connected, they are in the same component.
            # Adding edge (u,v) creates a cycle within that component.
            # If there are OTHER components (other lines), then this cycle is isolated. -> Premature.
            # If there are NO other components, is it the whole board?
            # Maybe there are some 'stranded' edges?
            
            # Let's use `count_connected_components` from daa.graph_algos
            # Note: graph.adj_list includes currently active edges.
            num_comps = count_connected_components(graph.adj_list, graph.vertices)
            
            # If num_comps > 1, then closing a loop in one component leaves others stranded. -> Invalid.
            if num_comps > 1:
                return False, "Premature Loop (Disconnected Segments)"
                
            # If num_comps == 1, we are closing the ONLY component.
            # But does it contain ALL active vertices?
            # A loop has all degrees == 2.
            # We are adding (u,v). u and v were degree 1 (endpoints). Now they become degree 2.
            # If ANY other vertex in this component has degree < 2 (i.e. degree 1), 
            # then we are closing a sub-loop while leaving a tail??
            # No, if u,v were connected, and we add (u,v), and (u,v) were the ONLY degree 1 nodes,
            # then result is a perfect loop.
            # If there were OTHER degree 1 nodes, we can't close a loop between u and v 
            # because u and v must be the tips.
            
            # So: Check if there are any degree 1 nodes OTHER than u and v.
            degree_ones = [node for node in graph.vertices if graph.get_degree(node) == 1]
            # Since we are adding (u,v), u and v must sort of be in degree_ones for valid line extension.
            # If we are closing a loop, u and v MUST be degree 1.
            
            # If there are any degree 1 nodes NOT in {u, v}, then those will remain tails.
            # A closed loop cannot have tails.
            # So if `len(degree_ones) > 2` (meaning more than just u and v), 
            # closing a loop now would leave tails. -> Premature Loop.
            
            if len(degree_ones) > 2:
                 return False, "Premature Loop (Leaves Tails)"

    else:
        # Removing edge
        # Usually always allowed unless we want to enforce structure (e.g. don't break strict segments)
        # But standard Slitherlink allows backtracking.
        pass
            
    return True, "OK"

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
