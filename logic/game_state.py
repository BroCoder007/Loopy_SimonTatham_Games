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
from collections import Counter
from logic.graph import Graph
from logic.validators import is_valid_move, check_win_condition
from logic.solvers.greedy_solver import GreedySolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.strategy_controller import StrategyController
from logic.generators.dnc_generator import DivideAndConquerPuzzleGenerator
from logic.statistics import StatisticsManager
from daa.greedy_algos import prim_mst, huffman_coding

class GameState:
    def __init__(self, rows=5, cols=5, difficulty="Medium", game_mode="vs_cpu", solver_strategy=None, generator_type="prim"):
        self.rows = rows
        self.cols = cols
        self.difficulty = difficulty
        self.game_mode = game_mode  # "vs_cpu", "two_player", "greedy_challenge"
        self.solver_strategy = self._normalize_solver_strategy(solver_strategy)
        self.generator_type = generator_type
        
        self.graph = Graph(rows, cols)
        self.clues = {}
        self.solution_edges = set()  # Filled by _generate_clues()
        if game_mode in ["vs_cpu", "expert"]:
            if self.solver_strategy == "dynamic_programming":
                self.cpu = DynamicProgrammingSolver(self)
            elif self.solver_strategy == "divide_conquer":
                self.cpu = DivideConquerSolver(self)
            else:  # "greedy"
                self.cpu = GreedySolver(self)
            self.strategy_controller = StrategyController(self, self.solver_strategy)
        else:
            self.cpu = None
            self.strategy_controller = None
        self.stats_mgr = StatisticsManager()
        self.last_cpu_move_info = None  # To store explanation of the last CPU move
        
        # Greedy Mode Mechanics
        self.edge_weights = {}
        self.energy = 100 
        self.max_energy = 100
        self.energy_cpu = 100
        self.max_energy_cpu = 100
        
        self._initialize_edge_weights()

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

        self.live_analysis_table = []  # Stores rows of comparative analysis data
        self.teacher_mode = False  # Teacher mode for hinting bad moves

    def _normalize_solver_strategy(self, strategy):
        """
        Restrict runtime routing to exactly three solver strategies:
        - greedy
        - divide_conquer
        - dynamic_programming
        """
        if strategy in ("divide_conquer", "divide_and_conquer"):
            return "divide_conquer"
        if strategy == "dynamic_programming":
            return "dynamic_programming"
        return "greedy"

    def _initialize_edge_weights(self):
        # Deterministic uniform edge cost map.
        for r in range(self.rows + 1):
            for c in range(self.cols):
                u, v = (r, c), (r, c+1)
                self.edge_weights[tuple(sorted((u, v)))] = 1
                
        for r in range(self.rows):
            for c in range(self.cols + 1):
                u, v = (r, c), (r+1, c)
                self.edge_weights[tuple(sorted((u, v)))] = 1

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
        Strategy-routed hint dispatcher.
        Delegates hint generation to the currently selected solver strategy.
        """
        strategy = self._normalize_solver_strategy(self.solver_strategy)

        if strategy == "divide_conquer":
            solver = self.cpu if isinstance(self.cpu, DivideConquerSolver) else DivideConquerSolver(self)
        elif strategy == "dynamic_programming":
            solver = self.cpu if isinstance(self.cpu, DynamicProgrammingSolver) else DynamicProgrammingSolver(self)
        else:
            solver = self.cpu if isinstance(self.cpu, GreedySolver) else GreedySolver(self)

        hint = solver.generate_hint(self)
        if isinstance(hint, dict):
            return hint

        move, reason = (None, "No hint available.")
        if isinstance(hint, tuple) and len(hint) == 2:
            move, reason = hint

        return {
            "move": move,
            "strategy": strategy.replace("_", " ").title(),
            "explanation": reason if isinstance(reason, str) else "No hint available.",
        }

    def get_next_cpu_move(self):
        """
        Controller-routed CPU move selection with per-turn fallback reset.

        Returns:
            (move, source_label, solver_used, fallback_message)
        """
        if self.strategy_controller is None:
            return None, "No moves available", None, None

        move, source = self.strategy_controller.get_next_cpu_move()
        solver_used = self.strategy_controller.get_solver_for_source(source)
        fallback_message = self.strategy_controller.get_fallback_message(source)
        return move, source, solver_used, fallback_message

    def _generate_clues(self):
        """
        Generate clues using selected generator algorithm.
        Default: Prim's Algorithm.
        Alternative: Divide & Conquer.
        """
        if self.generator_type == "dnc":
            print("[GameState] Using Divide & Conquer Generator...")
            generator = DivideAndConquerPuzzleGenerator(self.rows, self.cols, self.difficulty)
            self.clues, self.solution_edges = generator.generate()
        else:
            # Legacy/Default Prim's MST based generation
            self._generate_clues_prim()

        if self.game_mode == "expert":
            total_weight = 0
            for edge in self.solution_edges:
                total_weight += self.edge_weights.get(edge, 0)
            self.energy = total_weight + 20
            self.max_energy = self.energy

    def _generate_clues_prim(self):
        """
        Original logic moved here.
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
            "turn": self.turn,
            "generator_type": getattr(self, "generator_type", "prim")
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

    def clone_for_simulation(self):
        """
        Creates a lightweight copy of the GameState for simulation purposes.
        Only copies essential data (graph, clues, settings).
        Does NOT copy UI bindings, undo stack, or existing solver state.
        """
        # Create new instance
        # We pass None for strategies to start clean, then set them manually if needed
        new_state = GameState(
            rows=self.rows, 
            cols=self.cols, 
            difficulty=self.difficulty, 
            game_mode=self.game_mode, 
            solver_strategy=self.solver_strategy,
            generator_type=self.generator_type
        )
        
        # Copy Board State
        new_state.graph.edges = self.graph.edges.copy()
        new_state.graph.vertices = self.graph.vertices # Reference is fine, immutable tuples
        new_state.clues = self.clues.copy()
        new_state.edge_weights = self.edge_weights.copy()
        new_state.solution_edges = self.solution_edges.copy() if hasattr(self, "solution_edges") else set()
        
        # Settings
        new_state.energy = self.energy
        new_state.energy_cpu = self.energy_cpu
        new_state.turn = self.turn
        
        # We do NOT copy the 'cpu' solver instance. 
        # The simulation service will instantiate its own solvers on this new state.
        new_state.cpu = None
        new_state.strategy_controller = None

        return new_state
