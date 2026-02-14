#!/usr/bin/env python3
"""
Debug the vertical merge issue between top and bottom halves.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver, RegionSolution
from logic.game_state import GameState
import collections

def debug_vertical_merge():
    """Debug why top and bottom don't merge."""
    
    print("=== DEBUG: Vertical Merge (Top + Bottom) ===\n")
    
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
    
    # Check boundary masks for compatibility
    print("=== Top Half h_bottom_mask values ===")
    for i, sol in enumerate(top_half[:5]):
        print(f"  Top[{i}]: h_bottom_mask={sol.h_bottom_mask}, bottom_sig={sol.bottom_sig}")
    
    print("\n=== Bottom Half h_top_mask values ===")
    for i, sol in enumerate(bottom_half[:5]):
        print(f"  Bottom[{i}]: h_top_mask={sol.h_top_mask}, top_sig={sol.top_sig}")
    
    # Check compatibility
    print("\n=== Checking Compatibility ===")
    top_masks = set(sol.h_bottom_mask for sol in top_half)
    bottom_masks = set(sol.h_top_mask for sol in bottom_half)
    
    print(f"Unique top h_bottom_mask values: {len(top_masks)}")
    print(f"Unique bottom h_top_mask values: {len(bottom_masks)}")
    print(f"Common masks: {top_masks.intersection(bottom_masks)}")
    
    compatible = 0
    for i, top_sol in enumerate(top_half):
        for j, bottom_sol in enumerate(bottom_half):
            if top_sol.h_bottom_mask == bottom_sol.h_top_mask:
                compatible += 1
                if compatible <= 5:
                    print(f"  Compatible: Top[{i}] v Bottom[{j}]")
                    print(f"    Shared mask: {top_sol.h_bottom_mask}")
    
    print(f"\nTotal compatible pairs: {compatible}")
    
    if compatible == 0:
        print("\n*** PROBLEM: No compatible horizontal masks! ***")
        print("Top masks:", sorted(top_masks)[:10])
        print("Bottom masks:", sorted(bottom_masks)[:10])
    
    # Try the merge
    print("\n=== Attempting Vertical Merge ===")
    merged = solver._merge_vertical(top_half, bottom_half, mid_r, 0)
    print(f"Merged results: {len(merged)} states")

if __name__ == "__main__":
    debug_vertical_merge()
