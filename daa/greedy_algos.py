"""
Greedy Algorithms Implementation
================================
Detailed implementations of standard greedy algorithms.
Includes:
1. Fractional Knapsack Problem
2. Dijkstra's Shortest Path Algorithm
3. Prim's Minimum Spanning Tree Algorithm
4. Kruskal's Minimum Spanning Tree Algorithm
5. Huffman Coding

Each algorithm is implemented with detailed comments explaining the greedy choice property.
"""

import heapq
from collections import defaultdict

# -----------------------------------------------------------------------------
# 1. Fractional Knapsack Problem
# -----------------------------------------------------------------------------
def fractional_knapsack(capacity, weights, values):
    """
    Solves the Fractional Knapsack problem using a Greedy approach.
    
    Greedy Choice: Pick the item with the highest value-to-weight ratio.
    
    Args:
        capacity (float): Maximum weight the knapsack can hold.
        weights (list): List of weights of items.
        values (list): List of values of items.
        
    Returns:
        float: Maximum total value.
        list: List of chosen item indices (including partials).
    """
    n = len(weights)
    # Calculate value-to-weight ratio for each item
    items = []
    for i in range(n):
        ratio = values[i] / weights[i]
        items.append({'index': i, 'ratio': ratio, 'weight': weights[i], 'value': values[i]})
        
    # Sort items by ratio in descending order (Greedy Step)
    items.sort(key=lambda x: x['ratio'], reverse=True)
    
    total_value = 0.0
    current_weight = 0.0
    selected_indices = []
    
    for item in items:
        if current_weight + item['weight'] <= capacity:
            # Take the whole item
            current_weight += item['weight']
            total_value += item['value']
            selected_indices.append(item['index'])
        else:
            # Take fraction of the item
            remaining_capacity = capacity - current_weight
            fraction = remaining_capacity / item['weight']
            total_value += item['value'] * fraction
            current_weight += remaining_capacity
            selected_indices.append(item['index'])
            break # Knapsack is full
            
    return total_value, selected_indices

# -----------------------------------------------------------------------------
# 2. Dijkstra's Shortest Path Algorithm
# -----------------------------------------------------------------------------
def dijkstra(graph, start_node):
    """
    Finds the shortest paths from a start node to all other nodes in a weighted graph.
    
    Greedy Choice: Always expand the unvisited node with the smallest known distance.
    
    Args:
        graph (dict): Adjacency list where graph[u] = [(v, weight), ...].
        start_node: The starting node.
        
    Returns:
        dict: Shortest distances to each node.
        dict: Predecessors map for path reconstruction.
    """
    # Initialize distances to infinity
    distances = {node: float('infinity') for node in graph}
    distances[start_node] = 0
    predecessors = {node: None for node in graph}
    
    # Priority queue to store (distance, node), ordered by distance
    pq = [(0, start_node)]
    
    while pq:
        current_dist, current_node = heapq.heappop(pq)
        
        # If we found a shorter path to this node already, skip
        if current_dist > distances[current_node]:
            continue
            
        # Explore neighbors
        if current_node in graph:
            for neighbor, weight in graph[current_node]:
                distance = current_dist + weight
                
                # Relaxation step
                if distance < distances[neighbor]:
                    distances[neighbor] = distance
                    predecessors[neighbor] = current_node
                    heapq.heappush(pq, (distance, neighbor))
                
    return distances, predecessors

# -----------------------------------------------------------------------------
# 3. Prim's Minimum Spanning Tree Algorithm
# -----------------------------------------------------------------------------
def prim_mst(graph, start_node, max_nodes=None):
    """
    Finds the Minimum Spanning Tree (MST) of a connected, undirected graph.
    
    Greedy Choice: Grow the MST by adding the cheapest edge connecting the tree to a non-tree vertex.
    
    Args:
        graph (dict): Adjacency list where graph[u] = [(v, weight), ...].
        start_node: Arbitrary starting node.
        max_nodes (int, optional): Stop after visiting this many nodes.
        
    Returns:
        list: List of edges (u, v, weight) in the MST.
        float: Total weight of the MST.
    """
    mst_edges = []
    total_weight = 0
    visited = set([start_node])
    
    # Priority queue: (weight, from_node, to_node)
    edges_pq = []
    if start_node in graph:
        for neighbor, weight in graph[start_node]:
            heapq.heappush(edges_pq, (weight, start_node, neighbor))
        
    while edges_pq:
        if max_nodes and len(visited) >= max_nodes:
            break
            
        weight, u, v = heapq.heappop(edges_pq)
        
        if v in visited:
            continue
            
        # Add edge to MST
        visited.add(v)
        mst_edges.append((u, v, weight))
        total_weight += weight
        
        # Add new edges from v to the queue
        if v in graph:
            for next_node, next_weight in graph[v]:
                if next_node not in visited:
                    heapq.heappush(edges_pq, (next_weight, v, next_node))
                
    return mst_edges, total_weight, visited

# -----------------------------------------------------------------------------
# 4. Kruskal's Minimum Spanning Tree Algorithm
# -----------------------------------------------------------------------------
class DisjointSet:
    """Helper structure for Kruskal's Algorithm (Union-Find)."""
    def __init__(self, vertices):
        self.parent = {v: v for v in vertices}
        self.rank = {v: 0 for v in vertices}
        
    def find(self, item):
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item]) # Path compression
        return self.parent[item]
        
    def union(self, x, y):
        root_x = self.find(x)
        root_y = self.find(y)
        
        if root_x != root_y:
            # Union by rank
            if self.rank[root_x] < self.rank[root_y]:
                self.parent[root_x] = root_y
            elif self.rank[root_x] > self.rank[root_y]:
                self.parent[root_y] = root_x
            else:
                self.parent[root_y] = root_x
                self.rank[root_x] += 1
            return True
        return False

def kruskal_mst(vertices, edges):
    """
    Finds the Minimum Spanning Tree (MST) using Kruskal's Algorithm.
    
    Greedy Choice: Sort all edges by weight and iteratively add the cheapest edge that doesn't form a cycle.
    
    Args:
        vertices (list): List of all vertex labels.
        edges (list): List of (u, v, weight) tuples.
        
    Returns:
        list: List of edges in the MST.
        float: Total weight.
    """
    mst_edges = []
    total_weight = 0
    ds = DisjointSet(vertices)
    
    # Sort edges by weight (Greedy Step)
    sorted_edges = sorted(edges, key=lambda x: x[2])
    
    for u, v, weight in sorted_edges:
        # Check if adding this edge forms a cycle
        if ds.find(u) != ds.find(v):
            ds.union(u, v)
            mst_edges.append((u, v, weight))
            total_weight += weight
            
    return mst_edges, total_weight

# -----------------------------------------------------------------------------
# 5. Huffman Coding
# -----------------------------------------------------------------------------
class HuffmanNode:
    def __init__(self, char, freq):
        self.char = char
        self.freq = freq
        self.left = None
        self.right = None
        
    def __lt__(self, other):
        return self.freq < other.freq

def huffman_coding(char_freqs):
    """
    Builds the Huffman Tree for optimal prefix coding.
    
    Greedy Choice: Repeatedly combine the two symbols with the lowest frequency.
    
    Args:
        char_freqs (dict): Dictionary mapping characters to frequencies.
        
    Returns:
        dict: Mapping of characters to their Huffman codes.
    """
    heap = [HuffmanNode(char, freq) for char, freq in char_freqs.items()]
    heapq.heapify(heap)
    
    while len(heap) > 1:
        # Extract two nodes with lowest frequency
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        
        # Create new internal node
        merged = HuffmanNode(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        
        heapq.heappush(heap, merged)
        
    root = heap[0]
    codes = {}
    
    def generate_codes(node, current_code):
        if node is None:
            return
        if node.char is not None:
            codes[node.char] = current_code
            return
        generate_codes(node.left, current_code + "0")
        generate_codes(node.right, current_code + "1")
        
    generate_codes(root, "")
    return codes
