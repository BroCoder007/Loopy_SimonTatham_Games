#!/usr/bin/env python3
"""
Debug the merge degree validation between Q3 and Q4.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState

def debug_merge_validation():
    """Debug why Q3 and Q4 don't merge - check degree validation."""
    
    print("=== DEBUG: Merge Degree Validation ===\n")
    
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
    q3_solutions = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    q4_solutions = solver._solve_region(mid_r, solver.rows, mid_c, solver.cols)
    
    print(f"Q3 solutions: {len(q3_solutions)}")
    print(f"Q4 solutions: {len(q4_solutions)}\n")
    
    # Find matching pairs
    for i, q3_sol in enumerate(q3_solutions):
        for j, q4_sol in enumerate(q4_solutions):
            if q3_sol.v_right_mask == q4_sol.v_left_mask:
                print(f"\n=== Checking Q3[{i}] + Q4[{j}] ===")
                print(f"Shared v_mask: {q3_sol.v_right_mask}")
                
                v_shared = q3_sol.v_right_mask
                num_nodes = len(q3_sol.right_sig)
                
                print(f"Number of seam nodes: {num_nodes}")
                print(f"Q3 right_sig: {q3_sol.right_sig}")
                print(f"Q4 left_sig: {q4_sol.left_sig}")
                print(f"Q3 edges: {sorted(q3_sol.edges)}")
                print(f"Q4 edges: {sorted(q4_sol.edges)}")
                
                # Check each node on seam
                for k in range(num_nodes):
                    row = mid_r + k
                    
                    # Shared vertical edges
                    has_up = v_shared[k-1] if k > 0 else False
                    has_down = v_shared[k] if k < len(v_shared) else False
                    shared_count = (1 if has_up else 0) + (1 if has_down else 0)
                    
                    # Horizontal edges
                    e_left = tuple(sorted(((row, mid_c-1), (row, mid_c))))
                    l_in = 1 if e_left in q3_sol.edges else 0
                    
                    e_right = tuple(sorted(((row, mid_c), (row, mid_c+1))))
                    r_in = 1 if e_right in q4_sol.edges else 0
                    
                    total_degree = shared_count + l_in + r_in
                    
                    print(f"\n  Node at row {row}, col {mid_c}:")
                    print(f"    has_up={has_up}, has_down={has_down}, shared_count={shared_count}")
                    print(f"    e_left={e_left} in Q3: {l_in}")
                    print(f"    e_right={e_right} in Q4: {r_in}")
                    print(f"    TOTAL DEGREE: {total_degree}")
                    
                    if total_degree % 2 != 0:
                        print(f"    *** VIOLATION: Odd degree! ***")
                    if total_degree > 2:
                        print(f"    *** VIOLATION: Degree > 2! ***")

if __name__ == "__main__":
    debug_merge_validation()
