"""
DAA Unit 1: Graph Algorithms
============================
This module implements standard graph traversal algorithms.
Used for game logic validation (connectivity, cycle detection).

Algorithms:
1. BFS (Breadth-First Search) - O(V + E)
2. DFS (Depth-First Search) - O(V + E)
3. Cycle Detection - O(V + E)
4. Connected Components - O(V + E)
"""

from collections import deque

def bfs(adj_list, start_node):
    """
    Standard BFS traversal.
    Returns: Set of visited nodes.
    Time Complexity: O(V + E)
    """
    visited = set()
    queue = deque([start_node])
    visited.add(start_node)
    
    while queue:
        u = queue.popleft()
        for v in adj_list.get(u, []):
            if v not in visited:
                visited.add(v)
                queue.append(v)
    return visited

def dfs(adj_list, start_node, visited=None):
    """
    Standard DFS traversal (Recursive).
    Returns: Set of visited nodes.
    Time Complexity: O(V + E)
    """
    if visited is None:
        visited = set()
    
    visited.add(start_node)
    for v in adj_list.get(start_node, []):
        if v not in visited:
            dfs(adj_list, v, visited)
    return visited

def count_connected_components(adj_list, vertices):
    """
    Counts connected components in the graph.
    Only considers vertices that have at least one edge (degree > 0).
    Time Complexity: O(V + E)
    """
    visited = set()
    count = 0
    
    # Filter for active vertices
    active_vertices = [v for v in vertices if adj_list.get(v)]
    
    for v in active_vertices:
        if v not in visited:
            comp_visited = bfs(adj_list, v)
            visited.update(comp_visited)
            count += 1
            
    return count

def has_cycle(adj_list, vertices):
    """
    Detects if there is a cycle in the undirected graph using DFS.
    Time Complexity: O(V + E)
    """
    visited = set()
    
    def dfs_cycle_check(u, parent):
        visited.add(u)
        for v in adj_list.get(u, []):
            if v == parent:
                continue
            if v in visited:
                return True
            if dfs_cycle_check(v, u):
                return True
        return False
    
    active_vertices = [v for v in vertices if adj_list.get(v)]
    
    for v in active_vertices:
        if v not in visited:
            if dfs_cycle_check(v, None):
                return True
    return False

def get_degrees(adj_list, vertices):
    """
    Returns a dictionary of degrees for all vertices.
    """
    return {v: len(adj_list.get(v, [])) for v in vertices}
