#!/usr/bin/env python3
"""
Debug the vertical merge boundary compatibility.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState

def debug_vertical_boundaries():
    """Debug why top and bottom halves don't merge vertically."""
    
    print("=== DEBUG: Vertical Merge Boundaries ===\n")
    
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
    
    # Solve all quadrants
    q1 = solver._solve_region(0, mid_r, 0, mid_c)
    q2 = solver._solve_region(0, mid_r, mid_c, solver.cols)
    q3 = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    q4 = solver._solve_region(mid_r, solver.rows, mid_c, solver.cols)
    
    print(f"Q1: {len(q1)} solutions")
    print(f"Q2: {len(q2)} solutions")
    print(f"Q3: {len(q3)} solutions")
    print(f"Q4: {len(q4)} solutions\n")
    
    # Merge into halves
    top_half = solver._merge_horizontal(q1, q2, mid_c, 0)
    bottom_half = solver._merge_horizontal(q3, q4, mid_c, mid_r)
    
    print(f"Top half: {len(top_half)} states")
    print(f"Bottom half: {len(bottom_half)} states\n")
    
    # Check horizontal boundary masks for vertical merge
    print("=== Top Half h_bottom_mask (needs to match bottom's h_top_mask) ===")
    for i, sol in enumerate(top_half):
        print(f"  Top[{i}]: h_bottom_mask={sol.h_bottom_mask}")
        print(f"         bottom_sig={sol.bottom_sig}")
    
    print("\n=== Bottom Half h_top_mask ===")
    for i, sol in enumerate(bottom_half):
        print(f"  Bottom[{i}]: h_top_mask={sol.h_top_mask}")
        print(f"           top_sig={sol.top_sig}")
    
    # Check compatibility
    top_masks = set(sol.h_bottom_mask for sol in top_half)
    bottom_masks = set(sol.h_top_mask for sol in bottom_half)
    
    print(f"\nTop h_bottom_mask values: {top_masks}")
    print(f"Bottom h_top_mask values: {bottom_masks}")
    print(f"Common masks: {top_masks.intersection(bottom_masks)}")
    
    if not top_masks.intersection(bottom_masks):
        print("\n*** CRITICAL: No matching horizontal boundary masks! ***")
        print("Top and bottom halves cannot be merged vertically.")
        
        # This is the root cause - need to understand why masks don't match
        # Let's check individual quadrant boundaries
        print("\n=== Individual Quadrant Boundary Analysis ===")
        print(f"\nQ1 h_bottom_mask values:")
        for i, sol in enumerate(q1):
            print(f"  Q1[{i}]: {sol.h_bottom_mask}")
        
        print(f"\nQ2 h_bottom_mask values:")
        for i, sol in enumerate(q2):
            print(f"  Q2[{i}]: {sol.h_bottom_mask}")
        
        print(f"\nQ3 h_top_mask values:")
        for i, sol in enumerate(q3):
            print(f"  Q3[{i}]: {sol.h_top_mask}")
        
        print(f"\nQ4 h_top_mask values:")
        for i, sol in enumerate(q4):
            print(f"  Q4[{i}]: {sol.h_top_mask}")

if __name__ == "__main__":
    debug_vertical_boundaries()
