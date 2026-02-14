#!/usr/bin/env python3
"""
Debug Q3 and Q4 boundary masks in detail.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState

def debug_q3_q4_boundaries():
    """Debug Q3 and Q4 boundary mask generation."""
    
    print("=== DEBUG: Q3 and Q4 Boundary Analysis ===\n")
    
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
    
    # Solve Q3
    print(f"Q3: rows {mid_r}-{solver.rows}, cols 0-{mid_c}")
    q3_solutions = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    print(f"Q3 solutions: {len(q3_solutions)}\n")
    
    print("=== Q3 Solutions (ALL) ===")
    for i, sol in enumerate(q3_solutions):
        print(f"\nQ3[{i}]: {len(sol.edges)} edges")
        print(f"  Edges: {sorted(sol.edges)[:10]}")
        print(f"  v_right_mask: {sol.v_right_mask}")
        print(f"  right_sig: {sol.right_sig}")
        print(f"  h_top_mask: {sol.h_top_mask}")
        print(f"  top_sig: {sol.top_sig}")
        print(f"  h_bottom_mask: {sol.h_bottom_mask}")
        print(f"  bottom_sig: {sol.bottom_sig}")
    
    # Solve Q4
    print(f"\n\nQ4: rows {mid_r}-{solver.rows}, cols {mid_c}-{solver.cols}")
    q4_solutions = solver._solve_region(mid_r, solver.rows, mid_c, solver.cols)
    print(f"Q4 solutions: {len(q4_solutions)}\n")
    
    print("=== Q4 Solutions (ALL) ===")
    for i, sol in enumerate(q4_solutions):
        print(f"\nQ4[{i}]: {len(sol.edges)} edges")
        print(f"  Edges: {sorted(sol.edges)[:10]}")
        print(f"  v_left_mask: {sol.v_left_mask}")
        print(f"  left_sig: {sol.left_sig}")
        print(f"  h_top_mask: {sol.h_top_mask}")
        print(f"  top_sig: {sol.top_sig}")
        print(f"  h_bottom_mask: {sol.h_bottom_mask}")
        print(f"  bottom_sig: {sol.bottom_sig}")
    
    # The seam is at column mid_c
    print(f"\n=== Seam Analysis (column {mid_c}) ===")
    print(f"Q3 needs v_right_mask to match Q4's v_left_mask")
    print(f"\nQ3 v_right_mask values:")
    for i, sol in enumerate(q3_solutions):
        print(f"  Q3[{i}]: {sol.v_right_mask}")
    
    print(f"\nQ4 v_left_mask values:")
    for i, sol in enumerate(q4_solutions):
        print(f"  Q4[{i}]: {sol.v_left_mask}")
    
    # Check for matches
    q3_masks = set(sol.v_right_mask for sol in q3_solutions)
    q4_masks = set(sol.v_left_mask for sol in q4_solutions)
    
    print(f"\nQ3 unique v_right_mask: {q3_masks}")
    print(f"Q4 unique v_left_mask: {q4_masks}")
    print(f"Intersection: {q3_masks.intersection(q4_masks)}")
    
    if not q3_masks.intersection(q4_masks):
        print("\n*** CRITICAL: No matching boundary masks! ***")
        print("The regions cannot be merged.")

if __name__ == "__main__":
    debug_q3_q4_boundaries()
