"""
Greedy CPU Logic
================
AI Player implementation using Greedy Strategy and Sorting.
Refactored for DAA Project: "No-Cheat" Heuristics.
"""

from logic.validators import is_valid_move, count_edges_around_cell
from daa.sorting import merge_sort
from daa.greedy_algos import fractional_knapsack

class GreedyCPU:
    def __init__(self, game_state):
        self.game_state = game_state
        self.reasoning = "" # For "Thought Bubble"
        
    def make_move(self):
        """
        Execute the best move based on lawful greedy scoring.
        """
        candidates, best_move = self.decide_move()
        
        if not best_move:
            self.game_state.message = "CPU: No valid moves!"
            return None
            
        # 3. Update Message with Reasoning
        score = 0
        for m, s in candidates:
            if m == best_move:
                score = s
                break
                
        self.generate_reasoning(best_move, score)
        return best_move

    def decide_move(self):
        """
        Calculates and returns (candidates, best_move).
        Uses Fractional Knapsack to filter considered moves.
        Then sorts the result.
        """
        # 1. Get Candidate Moves (Raw)
        raw_candidates = self.get_ranked_moves()
        
        if not raw_candidates:
            return [], None
            
        # 2. FILTERING using FRACTIONAL KNAPSACK
        # Goal: Select moves that fit within "Attention/Energy" budget using Knapsack logic.
        # Items: Valid Moves
        # Weight: Move Cost (or default 1)
        # Value: Smart Score (must be > 0 for Knapsack)
        
        # Pre-process for Knapsack
        knapsack_items_map = [] # To map index back to move
        weights = []
        values = []
        
        valid_indices = []
        for i, (move, score) in enumerate(raw_candidates):
            if score > 0: # Knapsack needs positive values
                # Get cost
                edge = tuple(sorted(move))
                cost = self.game_state.edge_weights.get(edge, 1) # Default cost 1 (Normal Mode) / weighted (Expert)
                
                weights.append(cost)
                values.append(score)
                knapsack_items_map.append(i)
                valid_indices.append(i)
        
        # If no positive moves, use raw candidates directly (desperation)
        if not weights:
            final_candidates = raw_candidates
        else:
            capacity = 50 # "Attention Span"
            
            # Call Fractional Knapsack
            # Returns total_value, selected_indices (indices into the weights/values arrays)
            _, selected_indices = fractional_knapsack(capacity, weights, values)
            
            # Map back to raw_candidates
            final_candidates = []
            for idx in selected_indices:
                original_idx = knapsack_items_map[idx]
                final_candidates.append(raw_candidates[original_idx])
                
            # If knapsack selected nothing (rare), use direct selection
            if not final_candidates:
                final_candidates = raw_candidates

        # 3. Select Best Move using Sorting (DAA Requirement)
        moves_to_sort = final_candidates[:] 
        sorted_moves = merge_sort(moves_to_sort, key=lambda x: x[1], reverse=True)
                
        best_move = sorted_moves[0][0]
        best_score = sorted_moves[0][1]
        
        # Generate reasoning for the best move so it's ready for UI
        self.generate_reasoning(best_move, best_score)
        
        return sorted_moves, best_move

    def get_ranked_moves(self):
        """
        Returns all valid moves with their calculated scores.
        """
        possible_moves = self.get_all_valid_moves()
        scored_moves = []
        
        for move in possible_moves:
            score = self.calculate_smart_score(move)
            scored_moves.append((move, score))
            
        return scored_moves

    def get_all_valid_moves(self):
        moves = []
        rows = self.game_state.rows
        cols = self.game_state.cols
        graph = self.game_state.graph
        
        # Horizontal
        for r in range(rows + 1):
            for c in range(cols):
                u, v = (r, c), (r, c+1)
                if tuple(sorted((u, v))) not in graph.edges:
                    valid, _ = is_valid_move(graph, u, v, self.game_state.clues)
                    if valid:
                        moves.append((u, v))
                        
        # Vertical
        for r in range(rows):
            for c in range(cols + 1):
                u, v = (r, c), (r+1, c)
                if tuple(sorted((u, v))) not in graph.edges:
                    valid, _ = is_valid_move(graph, u, v, self.game_state.clues)
                    if valid:
                        moves.append((u, v))
        return moves

    def calculate_smart_score(self, move):
        """
        True Greedy Heuristic based on local topology.
        No cheating!
        """
        u, v = move
        score = 0
        graph = self.game_state.graph
        clues = self.game_state.clues
        
        # ---------------------------------------------------------
        # DEADLOCK PREVENTION (User Request)
        # ---------------------------------------------------------
        # If the human JUST removed this edge, do not add it back immediately.
        # Check independent of clues/neighbors.
        if self.game_state.undo_stack:
            last_u, last_v, last_action = self.game_state.undo_stack[-1]
            if last_action == "remove":
                if tuple(sorted((u, v))) == tuple(sorted((last_u, last_v))):
                    return -2000 # Strict penalty to prevent infinite loops

        # ---------------------------------------------------------
        # 1. Clue Heuristics (The "0" and "3" Rules)
        # ---------------------------------------------------------
        
        # Identify adjacent cells to this edge
        adj_cells = []
        r1, c1 = u
        r2, c2 = v
        if r1 == r2: # Horizontal
            c_min = min(c1, c2)
            if r1 > 0: adj_cells.append((r1-1, c_min)) # Above
            if r1 < graph.rows: adj_cells.append((r1, c_min)) # Below
        else: # Vertical
            r_min = min(r1, r2)
            if c1 > 0: adj_cells.append((r_min, c1-1)) # Left
            if c1 < graph.cols: adj_cells.append((r_min, c1)) # Right
            
        for cell in adj_cells:
            if cell in clues:
                val = clues[cell]
                current_edges = count_edges_around_cell(graph, cell)
                
                # Rule "0": Never place edge next to 0
                if val == 0:
                    return -1000 # Instant rejection
                
                # Rule "3":
                if val == 3:
                    score += 50
                    # Corner Bonus: If adding this edge connects to existing edges to specific corner
                    if current_edges >= 1:
                        score += 20
                
                # Completion Bonus:
                if current_edges + 1 == val:
                    score += 15 # Completing a clue is generally good
                elif current_edges + 1 > val:
                    # This should be invalid via validator, but as heuristic:
                    score -= 500

        # ---------------------------------------------------------
        # 2. Degree Heuristics (Continuity)
        # ---------------------------------------------------------
        
        # Use existing degrees (before move)
        deg_u = graph.get_degree(u)
        deg_v = graph.get_degree(v)
        
        # Extensions are good: Degree 1 -> 2
        if deg_u == 1: score += 100
        if deg_v == 1: score += 100
        
        # New segments are neutral/low: Degree 0 -> 1
        # We prefer extending lines over starting new ones
        if deg_u == 0 and deg_v == 0:
            score -= 10 # Slight penalty to discourage scattered starts
            
        # ---------------------------------------------------------
        # 4. EXPERT MODE: Energy Awareness
        # ---------------------------------------------------------
        if self.game_state.game_mode == "expert":
            # Get the weight (cost) of this edge
            edge_key = tuple(sorted(move))
            cost = self.game_state.edge_weights.get(edge_key, 0)
            
            # Key Logic: Penalize High Cost Edges
            # Cost ranges 1-9.
            # Penalty: Cost * 5. (Range -5 to -45)
            # This makes the CPU prioritize cheap edges (Green) over expensive ones (Red).
            score -= (cost * 5)
        
        return score

    def generate_reasoning(self, move, score):
        """
        Generate a "Thought Bubble" string explaining the move.
        """
        graph = self.game_state.graph
        clues = self.game_state.clues
        u, v = move
        
        # Analyze why we picked this
        reasons = []
        
        # Check adj cells for clues
        adj_cells = []
        r1, c1 = u
        r2, c2 = v
        if r1 == r2: # Horizontal
            c_min = min(c1, c2)
            if r1 > 0: adj_cells.append((r1-1, c_min))
            if r1 < graph.rows: adj_cells.append((r1, c_min))
        else: # Vertical
            r_min = min(r1, r2)
            if c1 > 0: adj_cells.append((r_min, c1-1))
            if c1 < graph.cols: adj_cells.append((r_min, c1))
            
        found_clue_reason = False
        for cell in adj_cells:
            if cell in clues:
                val = clues[cell]
                current = count_edges_around_cell(graph, cell)
                if val == 3:
                    reasons.append(f"satisfy '3' at {cell}")
                    found_clue_reason = True
                if current + 1 == val:
                    reasons.append(f"complete clue {val}")
                    found_clue_reason = True
                    
        if not found_clue_reason:
            # Maybe path extension?
            if graph.get_degree(u) == 1 or graph.get_degree(v) == 1:
                reasons.append("extend the path")
            else:
                reasons.append("explore new path")
                
        reason_str = ", ".join(reasons)
        self.reasoning = reason_str
        self.game_state.message = f"CPU: Placed at {u}-{v} to {reason_str}."
