#!/usr/bin/env python3
"""
Debug script to trace why hints are not being generated.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver, RegionSolution
from logic.game_state import GameState
from logic.graph import Graph
from logic.validators import is_valid_move

def debug_hint_system():
    """Debug the hint system step by step."""
    
    print("=== DEBUG: Advanced DP Hint System ===\n")
    
    # Create a simple 3x3 puzzle
    rows, cols = 3, 3
    
    clues = {
        (0, 0): 1, (0, 1): 3, (0, 2): 1,
        (1, 0): 3, (1, 1): 0, (1, 2): 3,
        (2, 0): 1, (2, 1): 3, (2, 2): 1
    }
    
    game_state = GameState(rows, cols, clues)
    solver = AdvancedDPSolver(game_state)
    
    print(f"Board: {rows}x{cols} with {len(clues)} clues")
    print(f"Solution computed: {solver._solution_computed}")
    
    # Step 1: Check if solution computation works
    print("\n=== Step 1: Computing full solution ===")
    try:
        solver._compute_full_solution()
        print(f"Solution computed: {solver._solution_computed}")
        print(f"Final solution edges: {len(solver._final_solution_edges)} edges")
        print(f"Solution moves: {len(solver.solution_moves)} moves")
    except Exception as e:
        print(f"ERROR during solution computation: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Step 2: Check merge stats
    print(f"\n=== Step 2: Merge Statistics ===")
    print(f"Region stats: {solver._merge_stats.get('region_stats', {})}")
    print(f"Merge details count: {len(solver._merge_stats.get('merge_details', []))}")
    
    # Step 3: Try to compute compatible boundary states
    print(f"\n=== Step 3: Computing compatible boundary states ===")
    try:
        compatible_states = solver._compute_compatible_boundary_states(game_state)
        print(f"Compatible states found: {len(compatible_states)}")
        
        if not compatible_states:
            print("WARNING: No compatible states found - this is why hints aren't working!")
            
            # Debug the quadrants
            print("\n=== Debugging quadrant solutions ===")
            mid_r = solver.rows // 2
            mid_c = solver.cols // 2
            
            quadrants = [
                (0, mid_r, 0, mid_c),
                (0, mid_r, mid_c, solver.cols),
                (mid_r, solver.rows, 0, mid_c),
                (mid_r, solver.rows, mid_c, solver.cols)
            ]
            
            for i, (r0, r1, c0, c1) in enumerate(quadrants):
                print(f"\nQ{i+1}: rows {r0}-{r1}, cols {c0}-{c1}")
                try:
                    solutions = solver._solve_region(r0, r1, c0, c1)
                    print(f"  Solutions: {len(solutions)}")
                    if solutions:
                        print(f"  First solution edges: {len(solutions[0].edges)}")
                except Exception as e:
                    print(f"  ERROR: {e}")
            
            return False
        
        # Step 4: Check edge intersections
        print(f"\n=== Step 4: Computing edge intersections ===")
        forced_inclusions, forced_exclusions = solver._compute_edge_intersections(compatible_states, game_state)
        print(f"Forced inclusions: {len(forced_inclusions)}")
        print(f"Forced exclusions: {len(forced_exclusions)}")
        
        if forced_inclusions:
            print(f"First inclusion: {forced_inclusions[0]}")
        if forced_exclusions:
            print(f"First exclusion: {forced_exclusions[0]}")
        
        # Step 5: Generate hint
        print(f"\n=== Step 5: Generating hint ===")
        hint = solver.generate_hint()
        print(f"Hint move: {hint.get('move')}")
        print(f"Hint strategy: {hint.get('strategy')}")
        print(f"Hint explanation: {hint.get('explanation', 'N/A')[:200]}...")
        
        return True
        
    except Exception as e:
        print(f"ERROR during compatible state computation: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = debug_hint_system()
    if success:
        print("\n✅ Debug completed successfully - hints should be working!")
    else:
        print("\n❌ Debug found issues - hints are broken!")
