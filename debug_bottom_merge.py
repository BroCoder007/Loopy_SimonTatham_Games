#!/usr/bin/env python3
"""
Debug the bottom half merge issue.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver, RegionSolution
from logic.game_state import GameState
import collections

def debug_bottom_half_merge():
    """Debug why Q3 and Q4 don't merge."""
    
    print("=== DEBUG: Bottom Half Merge ===\n")
    
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
    
    # Solve Q3 and Q4
    print("Solving Q3 (Bottom-Left)...")
    q3_solutions = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    print(f"Q3 solutions: {len(q3_solutions)}\n")
    
    print("Solving Q4 (Bottom-Right)...")
    q4_solutions = solver._solve_region(mid_r, solver.rows, mid_c, solver.cols)
    print(f"Q4 solutions: {len(q4_solutions)}\n")
    
    # Show boundary signatures
    print("=== Q3 Boundary Signatures ===")
    for i, sol in enumerate(q3_solutions[:3]):
        print(f"  Q3 Sol {i}:")
        print(f"    v_right_mask: {sol.v_right_mask}")
        print(f"    right_sig: {sol.right_sig}")
        print(f"    h_top_mask: {sol.h_top_mask}")
        print(f"    top_sig: {sol.top_sig}")
        print(f"    edges: {len(sol.edges)}")
    
    print("\n=== Q4 Boundary Signatures ===")
    for i, sol in enumerate(q4_solutions[:3]):
        print(f"  Q4 Sol {i}:")
        print(f"    v_left_mask: {sol.v_left_mask}")
        print(f"    left_sig: {sol.left_sig}")
        print(f"    h_top_mask: {sol.h_top_mask}")
        print(f"    top_sig: {sol.top_sig}")
        print(f"    edges: {len(sol.edges)}")
    
    # Check compatibility
    print("\n=== Checking Compatibility ===")
    compatible_pairs = 0
    for i, q3_sol in enumerate(q3_solutions):
        for j, q4_sol in enumerate(q4_solutions):
            if q3_sol.v_right_mask == q4_sol.v_left_mask:
                compatible_pairs += 1
                print(f"  Compatible: Q3[{i}] v Q4[{j}]")
                print(f"    Shared v_mask: {q3_sol.v_right_mask}")
    
    print(f"\nTotal compatible pairs: {compatible_pairs}")
    
    if compatible_pairs == 0:
        print("\n*** PROBLEM: No compatible boundary masks found! ***")
        print("This means Q3 and Q4 cannot be merged horizontally.")
    
    # Try the actual merge
    print("\n=== Attempting Merge ===")
    merged = solver._merge_horizontal(q3_solutions, q4_solutions, mid_c, mid_r)
    print(f"Merged results: {len(merged)} states")
    
    if not merged:
        print("\n*** Merge returned 0 states! ***")
        print("Checking merge constraints...")
        
        # Check what the merge sees
        right_map = collections.defaultdict(list)
        for sol in q4_solutions:
            right_map[sol.v_left_mask].append(sol)
        
        print(f"\nQ4 v_left_mask distribution:")
        for mask, sols in right_map.items():
            print(f"  {mask}: {len(sols)} solutions")
        
        print(f"\nQ3 v_right_mask values:")
        for i, sol in enumerate(q3_solutions):
            print(f"  Q3[{i}]: {sol.v_right_mask}")
            matches = right_map.get(sol.v_right_mask, [])
            print(f"    -> Matches in Q4: {len(matches)}")

if __name__ == "__main__":
    debug_bottom_half_merge()
