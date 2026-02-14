#!/usr/bin/env python3
"""
Debug the vertical merge degree validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState

def debug_vertical_merge_degrees():
    """Debug degree validation in vertical merge."""
    
    print("=== DEBUG: Vertical Merge Degree Validation ===\n")
    
    rows, cols = 3, 3
    clues = {
        (0, 0): 1, (0, 1): 3, (0, 2): 1,
        (1, 0): 3, (1, 1): 0, (1, 2): 3,
        (2, 0): 1, (2, 1): 3, (2, 2): 1
    }
    
    game_state = GameState(rows, cols, clues)
    solver = AdvancedDPSolver(game_state)
    
    mid_r = solver.rows // 2
    mid_c = solver.cols // 2
    
    # Solve and merge
    q1 = solver._solve_region(0, mid_r, 0, mid_c)
    q2 = solver._solve_region(0, mid_r, mid_c, solver.cols)
    q3 = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    q4 = solver._solve_region(mid_r, solver.rows, mid_c, solver.cols)
    
    top_half = solver._merge_horizontal(q1, q2, mid_c, 0)
    bottom_half = solver._merge_horizontal(q3, q4, mid_c, mid_r)
    
    print(f"Top half: {len(top_half)} states")
    print(f"Bottom half: {len(bottom_half)} states\n")
    
    # Find matching pairs
    matching_pairs = []
    for i, top_sol in enumerate(top_half):
        for j, bottom_sol in enumerate(bottom_half):
            if top_sol.h_bottom_mask == bottom_sol.h_top_mask:
                matching_pairs.append((i, j, top_sol, bottom_sol))
    
    print(f"Matching boundary mask pairs: {len(matching_pairs)}\n")
    
    # Check degree validation for each pair
    for pair_idx, (i, j, top_sol, bottom_sol) in enumerate(matching_pairs):
        print(f"=== Pair {pair_idx}: Top[{i}] + Bottom[{j}] ===")
        print(f"Shared h_mask: {top_sol.h_bottom_mask}")
        
        h_shared = top_sol.h_bottom_mask
        num_nodes = len(top_sol.bottom_sig)
        
        print(f"Seam at row {mid_r}, nodes: {num_nodes}")
        print(f"Top bottom_sig: {top_sol.bottom_sig}")
        print(f"Bottom top_sig: {bottom_sol.top_sig}")
        
        all_valid = True
        for k in range(num_nodes):
            col = k  # c_min = 0
            
            # Shared horizontal edges
            has_left = h_shared[k-1] if k > 0 else False
            has_right = h_shared[k] if k < len(h_shared) else False
            shared_count = (1 if has_left else 0) + (1 if has_right else 0)
            
            # Vertical edges (top and bottom contributions)
            e_top = tuple(sorted(((mid_r-1, col), (mid_r, col))))
            t_in = 1 if e_top in top_sol.edges else 0
            
            e_bottom = tuple(sorted(((mid_r, col), (mid_r+1, col))))
            b_in = 1 if e_bottom in bottom_sol.edges else 0
            
            total_degree = shared_count + t_in + b_in
            
            status = "OK" if (total_degree % 2 == 0 and total_degree <= 2) else "INVALID"
            print(f"  Node {k} (col {col}): shared={shared_count}, top_in={t_in}, bottom_in={b_in}, total={total_degree} [{status}]")
            
            if total_degree % 2 != 0 or total_degree > 2:
                all_valid = False
        
        print(f"  All valid: {all_valid}\n")

if __name__ == "__main__":
    debug_vertical_merge_degrees()
