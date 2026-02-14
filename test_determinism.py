#!/usr/bin/env python3
"""
Compare region solutions between two solver calls.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState

def test_determinism():
    """Test if _solve_region produces deterministic results."""
    
    print("=== Testing Solver Determinism ===\n")
    
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
    
    # First call
    print("First call to _solve_region for Q3:")
    q3_first = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    print(f"  Solutions: {len(q3_first)}")
    
    # Second call
    print("\nSecond call to _solve_region for Q3:")
    q3_second = solver._solve_region(mid_r, solver.rows, 0, mid_c)
    print(f"  Solutions: {len(q3_second)}")
    
    # Compare
    if len(q3_first) != len(q3_second):
        print(f"\n*** NON-DETERMINISTIC! Different solution counts ***")
    else:
        # Compare boundary masks
        first_masks = set(sol.v_right_mask for sol in q3_first)
        second_masks = set(sol.v_right_mask for sol in q3_second)
        
        print(f"\nFirst call masks: {first_masks}")
        print(f"Second call masks: {second_masks}")
        
        if first_masks != second_masks:
            print("*** DIFFERENT BOUNDARY MASKS! ***")
        else:
            print("Boundary masks match.")
    
    # Now test full computation + hint generation
    print("\n\n=== Testing Full Computation + Hint ===")
    
    # Create fresh solver
    game_state2 = GameState(rows, cols, clues)
    solver2 = AdvancedDPSolver(game_state2)
    
    # Compute full solution
    print("Computing full solution...")
    solver2._compute_full_solution()
    print(f"Final solution edges: {len(solver2._final_solution_edges)}")
    
    # Now get hint
    print("\nGetting hint...")
    hint = solver2.generate_hint()
    print(f"Hint move: {hint.get('move')}")

if __name__ == "__main__":
    test_determinism()
