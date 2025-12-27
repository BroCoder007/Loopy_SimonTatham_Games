"""
Game State Manager
==================
Central controller for the game.
Manages:
- Graph state
- Undo/Redo stack
- Player turns
- Difficulty settings
- Interaction with CPU and Validators
"""

import os
import random
import pickle
import heapq
from collections import Counter
from logic.graph import Graph
from logic.validators import is_valid_move, check_win_condition
from logic.greedy_cpu import GreedyCPU
from logic.statistics import StatisticsManager
from daa.greedy_algos import prim_mst, dijkstra, huffman_coding

class GameState:
    def __init__(self, rows=5, cols=5, difficulty="Medium", game_mode="vs_cpu"):
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        self.game_mode = game_mode  # "vs_cpu", "two_player", "greedy_challenge"
        self.teacher_mode = False # DAA Feature: Teacher Mode
        
        self.graph = Graph(rows, cols)
        self.clues = {}
        # CPU is active in vs_cpu AND expert mode (Duel)
        self.cpu = GreedyCPU(self) if game_mode in ["vs_cpu", "expert"] else None
        self.stats_mgr = StatisticsManager()
        
        # Greedy Mode Mechanics
        self.edge_weights = {}
        self.energy = 100 
        self.max_energy = 100
        self.energy_cpu = 100
        self.max_energy_cpu = 100
        
        self._assign_weights()

        if game_mode == "expert":
            self.turn = "Player 1 (Human)" # Expert is now PvE
            self._generate_clues() # This sets self.energy based on solution
            # Give CPU same starting energy as player for fairness
            self.energy_cpu = self.energy
            self.max_energy_cpu = self.energy
        elif game_mode == "vs_cpu":
            self.turn = "Player 1 (Human)"
            self._generate_clues()
        else:
            self.turn = "Player 1"
            self._generate_clues()
        
        self.game_over = False
        self.winner = None
        self.message = "Game Start!"
        
        self.undo_stack = []
        self.redo_stack = []
        
        # self.solution_edges = set() # Set in _generate_clues

    def _assign_weights(self):
        # Assign random weights to all potential edges
        for r in range(self.rows + 1):
            for c in range(self.cols):
                u, v = (r, c), (r, c+1)
                self.edge_weights[tuple(sorted((u, v)))] = random.randint(1, 9)
                
        for r in range(self.rows):
            for c in range(self.cols + 1):
                u, v = (r, c), (r+1, c)
                self.edge_weights[tuple(sorted((u, v)))] = random.randint(1, 9)

    def _calculate_required_energy(self):
        return 100 # Placeholder

    def make_move(self, u, v, is_cpu=False):
        if self.game_over: return False
        
        # Validate Turn
        if self.game_mode in ["vs_cpu", "expert"]:
            if self.turn == "Player 2 (CPU)" and not is_cpu:
                return False # Human trying to move on CPU turn
            if self.turn == "Player 1 (Human)" and is_cpu:
                return False # CPU trying to move on Human turn
        
        edge = tuple(sorted((u, v)))
        cost = self.edge_weights.get(edge, 0)
        
        if edge in self.graph.edges:
            # Removing edge
            self.graph.remove_edge(u, v)
            action = "remove"
            self.message = "Edge Removed"
            
            # Refund energy in Expert mode
            if self.game_mode == "expert":
                if is_cpu:
                    self.energy_cpu += cost
                else:
                    self.energy += cost
        else:
            # Adding edge
            if self.game_mode == "expert":
                if not is_cpu:
                    # Human Energy Check
                    if self.energy < cost:
                        self.message = "Not enough Energy! You LOSE!"
                        self.game_over = True
                        self.winner = "Player 2 (CPU)"
                        return False
                else:
                    # CPU Energy Check
                    if self.energy_cpu < cost:
                        self.message = "CPU Out of Energy! You WIN!"
                        self.game_over = True
                        self.winner = "Player 1 (Human)"
                        return False
                    
            valid, reason = is_valid_move(self.graph, u, v, self.clues)
            if not valid:
                self.message = f"Invalid: {reason}"
                return False
            self.graph.add_edge(u, v)
            action = "add"
            
            if self.game_mode == "expert":
                self.message = f"Edge Added (Cost: {cost})"
                if is_cpu:
                    self.energy_cpu -= cost
                else:
                    self.energy -= cost
            else:
                self.message = "Edge Added"
            
        # Record for Undo (Clear redo stack on new move)
        self.undo_stack.append((u, v, action))
        self.redo_stack.clear()
        
        self._check_game_status()
        
        if not self.game_over:
            self.switch_turn()
            
        return True

    def get_hint(self):
        """
        Robust Hint System (Consolidated):
        1. "Loose Ends" Logic using Dijkstra's Algorithm (Refactored).
           Find shortest path between two degree-1 vertices.
        2. Fallback to Helper Logic (Solution edge suggestion).
        """
        
        # 1. Dijkstra for Loose Ends (Any Mode provided hints are allowed)
        # Find all vertices with degree 1
        degree_ones = [node for node in self.graph.vertices if self.graph.get_degree(node) == 1]
        
        if len(degree_ones) >= 2:
            # Attempt to connect two loose ends
            start = degree_ones[0]
            targets = set(degree_ones[1:])
            
            # Construct Weighted Graph of VALID MOVES using edge_weights
            # Nodes: All grid vertices
            # Edges: All valid, non-existing edges
            dijkstra_graph = {}
            
            # We need to build the graph for all possible nodes
            # A node u has neighbors v if (u,v) is valid add
            
            # Optimisation: Build on demand or full build? Full build is safer for Dijkstra.
            for r in range(self.rows + 1):
                for c in range(self.cols + 1):
                    u = (r, c)
                    dijkstra_graph[u] = []
                    
                    # Check potential neighbors
                    candidates = []
                    if r < self.rows: candidates.append(((r+1, c), tuple(sorted((u, (r+1, c)))))) # Down
                    if r > 0: candidates.append(((r-1, c), tuple(sorted((u, (r-1, c)))))) # Up
                    if c < self.cols: candidates.append(((r, c+1), tuple(sorted((u, (r, c+1)))))) # Right
                    if c > 0: candidates.append(((r, c-1), tuple(sorted((u, (r, c-1)))))) # Left
                    
                    for neighbor, edge_key in candidates:
                         if edge_key not in self.graph.edges:
                             # Is it valid?
                             valid, _ = is_valid_move(self.graph, u, neighbor, self.clues)
                             if valid:
                                 weight = self.edge_weights.get(edge_key, 1)
                                 dijkstra_graph[u].append((neighbor, weight))

            # Run Dijkstra
            dists, preds = dijkstra(dijkstra_graph, start)
            
            # Find closest target
            min_dist = float('inf')
            best_target = None
            
            for t in targets:
                if dists.get(t, float('inf')) < min_dist:
                    min_dist = dists[t]
                    best_target = t
            
            if best_target and min_dist != float('inf'):
                # Backtrack to find first move
                curr = best_target
                path = []
                while curr != start and curr is not None:
                    path.append(curr)
                    curr = preds[curr]
                
                if path:
                    first_step_node = path[-1] # The one connected to start
                    return (start, first_step_node), "Hint: Connect this loose end!"

        # 2. Fallback: Suggest solution edge
        if not self.solution_edges:
            return None, "No solution available."

        for edge in self.solution_edges:
            if edge not in self.graph.edges:
                u, v = edge
                valid, reason = is_valid_move(self.graph, u, v, self.clues)
                if valid:
                    return (u, v), "Add this solution edge!"
                else:
                    # Check blockers
                    for neighbor in self.graph.adj_list[u]:
                        existing_edge = tuple(sorted((u, neighbor)))
                        if existing_edge not in self.solution_edges:
                            return existing_edge, "Remove this incorrect edge!"
                    for neighbor in self.graph.adj_list[v]:
                        existing_edge = tuple(sorted((v, neighbor)))
                        if existing_edge not in self.solution_edges:
                            return existing_edge, "Remove this incorrect edge!"
                            
        for edge in self.graph.edges:
            if edge not in self.solution_edges:
                return edge, "Remove this extra edge!"
                
        return None, "Puzzle solved?"

    def _generate_clues(self):
        """
        Generate clues using Prim's Algorithm to create a connected region.
        """
        # 1. Build Graph of Cells for Prim's
        cell_graph = {}
        for r in range(self.rows):
            for c in range(self.cols):
                u = (r, c)
                neighbors = []
                # Down
                if r < self.rows - 1: 
                    v = (r+1, c)
                    neighbors.append((v, random.randint(1, 100)))
                # Up
                if r > 0: 
                    v = (r-1, c)
                    neighbors.append((v, random.randint(1, 100)))
                # Right
                if c < self.cols - 1: 
                    v = (r, c+1)
                    neighbors.append((v, random.randint(1, 100)))
                # Left
                if c > 0: 
                    v = (r, c-1)
                    neighbors.append((v, random.randint(1, 100)))
                
                cell_graph[u] = neighbors
        
        # 2. Run Prim's to get a Random Tree (Region)
        start_cell = (random.randint(0, self.rows-1), random.randint(0, self.cols-1))
        target_size = int(self.rows * self.cols * random.uniform(0.4, 0.7))
        
        # Call Prim's with max_nodes to limit region size
        _, _, region_visited = prim_mst(cell_graph, start_cell, max_nodes=target_size)
        region = region_visited # Set of cells
        
        # 3. Calculate boundary edges of this region (MST Boundary)
        solution_edges = set()
        
        # Horizontal edges
        for r in range(self.rows + 1):
            for c in range(self.cols):
                cell_above = (r-1, c) if r > 0 else None
                cell_below = (r, c) if r < self.rows else None
                
                in_above = cell_above in region
                in_below = cell_below in region
                
                if in_above != in_below: # XOR
                    solution_edges.add(tuple(sorted(((r, c), (r, c+1)))))
                    
        # Vertical edges
        for r in range(self.rows):
            for c in range(self.cols + 1):
                cell_left = (r, c-1) if c > 0 else None
                cell_right = (r, c) if c < self.cols else None
                
                in_left = cell_left in region
                in_right = cell_right in region
                
                if in_left != in_right:
                    solution_edges.add(tuple(sorted(((r, c), (r+1, c)))))
                    
        # 4. Generate Clues based on logic
        temp_clues = {}
        for r in range(self.rows):
            for c in range(self.cols):
                count = 0
                if tuple(sorted(((r, c), (r, c+1)))) in solution_edges: count += 1
                if tuple(sorted(((r+1, c), (r+1, c+1)))) in solution_edges: count += 1
                if tuple(sorted(((r, c), (r+1, c)))) in solution_edges: count += 1
                if tuple(sorted(((r, c+1), (r+1, c+1)))) in solution_edges: count += 1
                temp_clues[(r, c)] = count
                
        keep_prob = 0.6
        if self.difficulty == "Easy": keep_prob = 0.8
        elif self.difficulty == "Hard": keep_prob = 0.4
        
        self.clues = {}
        for cell, val in temp_clues.items():
            if random.random() < keep_prob:
                self.clues[cell] = val
                
        self.solution_edges = solution_edges
        
        if self.game_mode == "expert":
            total_weight = 0
            for edge in self.solution_edges:
                total_weight += self.edge_weights.get(edge, 0)
            self.energy = total_weight + 20
            self.max_energy = self.energy

    def save_game(self, filename="savegame.bin"):
        """
        Compresses undo stack using Huffman Coding and saves to binary file.
        """
        # 1. Convert Undo Stack to String Representation
        # Format: "u_r,u_c|v_r,v_c|action;"
        data_str = ""
        for item in self.undo_stack:
            u, v, action = item
            data_str += f"{u[0]},{u[1]}|{v[0]},{v[1]}|{action};"
            
        if not data_str:
            data_str = "EMPTY"

        # 2. Frequency Analysis
        freqs = Counter(data_str)
        
        # 3. Generate Huffman Codes
        codes = huffman_coding(freqs)
        
        # 4. Encode Data
        encoded_bits = ""
        for char in data_str:
            encoded_bits += codes[char]
            
        # 5. Save (Pickle the components)
        save_data = {
            "codes": codes,
            "encoded_bits": encoded_bits,
            "original_length": len(data_str),
            "rows": self.rows,
            "cols": self.cols,
            "difficulty": self.difficulty,
            "game_mode": self.game_mode,
            "clues": self.clues,
            "edge_weights": self.edge_weights,
            "solution_edges": getattr(self, "solution_edges", set()),
            "energy": self.energy,
            "energy_cpu": self.energy_cpu,
            "turn": self.turn
        }
        
        try:
            with open(filename, "wb") as f:
                pickle.dump(save_data, f)
            self.message = "Game Saved (Huffman Compressed)!"
            return True
        except Exception as e:
            self.message = f"Save Failed: {e}"
            return False

    def load_game(self, filename="savegame.bin"):
        """
        Loads game state from binary file, decoding Huffman compression.
        """
        if not os.path.exists(filename):
            self.message = "No save file found."
            return False

        try:
            with open(filename, "rb") as f:
                save_data = pickle.load(f)
            
            codes = save_data["codes"]
            encoded_bits = save_data["encoded_bits"]
            
            # Reconstruct decoding map
            # codes is char -> bitstring
            # we need bitstring -> char
            # But Huffman decoding usually involves traversing a tree or waiting for a match.
            # Since we have the codes dict, we can reverse it.
            reverse_codes = {v: k for k, v in codes.items()}
            
            decoded_str = ""
            current_bits = ""
            
            for bit in encoded_bits:
                current_bits += bit
                if current_bits in reverse_codes:
                    decoded_str += reverse_codes[current_bits]
                    current_bits = ""
                    
            if decoded_str == "EMPTY":
                self.message = "Loaded empty game."
                return True
                
            # Restore Board Configuration
            self.rows = save_data.get("rows", self.rows)
            self.cols = save_data.get("cols", self.cols)
            self.difficulty = save_data.get("difficulty", self.difficulty)
            self.game_mode = save_data.get("game_mode", self.game_mode)
            self.clues = save_data.get("clues", self.clues)
            self.edge_weights = save_data.get("edge_weights", self.edge_weights)
            self.solution_edges = save_data.get("solution_edges", set())
            self.energy = save_data.get("energy", 100)
            self.energy_cpu = save_data.get("energy_cpu", 100)
            self.turn = save_data.get("turn", "Player 1 (Human)")
            
            # Reconstruct Undo Stack
            # Format: "u_r,u_c|v_r,v_c|action;"
            items = decoded_str.split(';')
            
            # Reset current game state before replaying
            self.graph = Graph(self.rows, self.cols)
            
            # Parsing logic 
            new_undo_stack = []
            
            for item_str in items:
                if not item_str: continue
                parts = item_str.split('|')
                if len(parts) != 3: continue
                
                u_str, v_str, action = parts
                u = tuple(map(int, u_str.split(',')))
                v = tuple(map(int, v_str.split(',')))
                
                new_undo_stack.append((u, v, action))
                
                # Replay effect
                if action == "add":
                    self.graph.add_edge(u, v)
                elif action == "remove":
                    self.graph.remove_edge(u, v)
                    
            self.undo_stack = new_undo_stack
            self.redo_stack = [] # Clear redo
            self.message = "Game Loaded Successfully!"
            return True
            
        except Exception as e:
            self.message = f"Load Failed: {e}"
            print(f"Load Error: {e}")
            return False

    def undo(self):
        if not self.undo_stack: return False
        
        u, v, action = self.undo_stack.pop()
        
        # Expert Mode: Handle Energy Undo
        cost = 0
        if self.game_mode == "expert":
             edge = tuple(sorted((u, v)))
             cost = self.edge_weights.get(edge, 0)

        # Reverse action
        if action == "add":
            self.graph.remove_edge(u, v)
            if self.game_mode == "expert":
                if "CPU" in self.turn: 
                    self.energy += cost
                else: 
                    self.energy_cpu += cost
                    
        else: # action == "remove"
            self.graph.add_edge(u, v)
            if self.game_mode == "expert":
                if "CPU" in self.turn:
                    self.energy -= cost
                else:
                    self.energy_cpu -= cost
            
        self.redo_stack.append((u, v, action))
        self.switch_turn() 
        self.message = "Undo Successful"
        return True

    def redo(self):
        if not self.redo_stack: return False
        
        u, v, action = self.redo_stack.pop()
        
        cost = 0
        if self.game_mode == "expert":
             edge = tuple(sorted((u, v)))
             cost = self.edge_weights.get(edge, 0)
        
        if action == "add":
            self.graph.add_edge(u, v)
            if self.game_mode == "expert":
                if "Human" in self.turn:
                    self.energy -= cost
                else:
                    self.energy_cpu -= cost
        else:
            self.graph.remove_edge(u, v)
            if self.game_mode == "expert":
                if "Human" in self.turn:
                    self.energy += cost
                else:
                    self.energy_cpu += cost
            
        self.undo_stack.append((u, v, action))
        self.switch_turn()
        self.message = "Redo Successful"
        return True

    def switch_turn(self):
        if self.game_mode in ["vs_cpu", "expert"]:
            if self.turn == "Player 1 (Human)":
                self.turn = "Player 2 (CPU)"
            else:
                self.turn = "Player 1 (Human)"
        else:  # two_player mode
            if self.turn == "Player 1":
                self.turn = "Player 2"
            else:
                self.turn = "Player 1"

    def get_all_valid_moves(self):
        moves = []
        # Horizontal
        for r in range(self.rows + 1):
            for c in range(self.cols):
                u, v = (r, c), (r, c+1)
                if tuple(sorted((u, v))) not in self.graph.edges:
                    valid, _ = is_valid_move(self.graph, u, v, self.clues)
                    if valid:
                        moves.append((u, v))
                        
        # Vertical
        for r in range(self.rows):
            for c in range(self.cols + 1):
                u, v = (r, c), (r+1, c)
                if tuple(sorted((u, v))) not in self.graph.edges:
                    valid, _ = is_valid_move(self.graph, u, v, self.clues)
                    if valid:
                        moves.append((u, v))
        return moves

    def _check_game_status(self):
        won, reason = check_win_condition(self.graph, self.clues)
        if won:
            self.game_over = True
            self.winner = self.turn
            self.message = f"GAME OVER! {self.winner} Wins!"
            self.stats_mgr.record_game(self.winner)
            return

        if not self.get_all_valid_moves():
            self.game_over = True
            self.winner = "Stalemate"
            self.message = "GAME OVER! No valid moves left. Stalemate."
