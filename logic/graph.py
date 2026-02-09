"""
Graph Representation
====================
Explicit graph data structure for the Slitherlink grid.
Nodes are (r, c) tuples. Edges are connections between them.
"""

class Graph:
    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols
        self.vertices = []
        self.adj_list = {}
        self.edges = set() # Stores tuples of sorted((u, v))
        
        # Initialize vertices
        for r in range(rows + 1):
            for c in range(cols + 1):
                v = (r, c)
                self.vertices.append(v)
                self.adj_list[v] = []

    def add_edge(self, u, v):
        """Add an edge between u and v."""
        edge = tuple(sorted((u, v)))
        if edge not in self.edges:
            self.edges.add(edge)
            self.adj_list[u].append(v)
            self.adj_list[v].append(u)
            return True
        return False

    def remove_edge(self, u, v):
        """Remove an edge between u and v."""
        edge = tuple(sorted((u, v)))
        if edge in self.edges:
            self.edges.remove(edge)
            if v in self.adj_list[u]: self.adj_list[u].remove(v)
            if u in self.adj_list[v]: self.adj_list[v].remove(u)
            return True
        return False

    def get_degree(self, v):
        return len(self.adj_list.get(v, []))
    
    def get_neighbors(self, v):
        return self.adj_list.get(v, [])
    
    def copy(self):
        """Create a deep copy of the graph structure."""
        new_graph = Graph(self.rows, self.cols)
        new_graph.edges = self.edges.copy()
        for u, neighbors in self.adj_list.items():
            new_graph.adj_list[u] = neighbors[:]
        return new_graph
