#!/usr/bin/env python3
"""
Detailed debug script to trace why merged solutions fail validation.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver, RegionSolution, UnionFind
from logic.game_state import GameState
from logic.validators import is_valid_move
import collections

def debug_solution_validation():
    """Debug why merged solutions fail validation."""
    
    print("=== DETAILED DEBUG: Solution Validation ===\n")
    
    # Create a simple 3x3 puzzle
    rows, cols = 3, 3
    
    clues = {
        (0, 0): 1, (0, 1): 3, (0, 2): 1,
        (1, 0): 3, (1, 1): 0, (1, 2): 3,
        (2, 0): 1, (2, 1): 3, (2, 2): 1
    }
    
    game_state = GameState(rows, cols, clues)
    solver = AdvancedDPSolver(game_state)
    
    print(f"Board: {rows}x{cols}")
    print(f"Clues: {clues}\n")
    
    # Divide grid
    mid_r = solver.rows // 2
    mid_c = solver.cols // 2
    
    quadrants = [
        (0, mid_r, 0, mid_c),
        (0, mid_r, mid_c, solver.cols),
        (mid_r, solver.rows, 0, mid_c),
        (mid_r, solver.rows, mid_c, solver.cols)
    ]
    
    # Solve quadrants
    q_results = []
    for i, (r0, r1, c0, c1) in enumerate(quadrants):
        print(f"Q{i+1}: rows {r0}-{r1}, cols {c0}-{c1}")
        solutions = solver._solve_region(r0, r1, c0, c1)
        q_results.append(solutions)
        print(f"  Solutions: {len(solutions)}")
        if solutions:
            for j, sol in enumerate(solutions[:2]):  # Show first 2
                print(f"    Sol {j}: {len(sol.edges)} edges, loops={sol.internal_loops}")
                print(f"      Top sig: {sol.top_sig}")
                print(f"      Bottom sig: {sol.bottom_sig}")
                print(f"      Left sig: {sol.left_sig}")
                print(f"      Right sig: {sol.right_sig}")
        print()
    
    # Merge top half
    print("=== MERGING TOP HALF (Q1 + Q2) ===")
    top_half = solver._merge_horizontal(q_results[0], q_results[1], mid_c, 0)
    print(f"Top half merged: {len(top_half)} states\n")
    
    if top_half:
        for i, sol in enumerate(top_half[:2]):
            print(f"  Top Sol {i}: {len(sol.edges)} edges, loops={sol.internal_loops}")
            print(f"    Top sig: {sol.top_sig}")
            print(f"    Bottom sig: {sol.bottom_sig}")
            # Check degrees
            degrees = collections.defaultdict(int)
            for u, v in sol.edges:
                degrees[u] += 1
                degrees[v] += 1
            odd_degree_nodes = [n for n, d in degrees.items() if d % 2 != 0]
            print(f"    Nodes with odd degree: {len(odd_degree_nodes)}")
            if odd_degree_nodes:
                print(f"    Odd degree nodes: {odd_degree_nodes[:5]}")
    
    # Merge bottom half
    print("\n=== MERGING BOTTOM HALF (Q3 + Q4) ===")
    bottom_half = solver._merge_horizontal(q_results[2], q_results[3], mid_c, mid_r)
    print(f"Bottom half merged: {len(bottom_half)} states\n")
    
    # Merge full grid
    print("=== MERGING FULL GRID (Top + Bottom) ===")
    full_grid = solver._merge_vertical(top_half, bottom_half, mid_r, 0)
    print(f"Full grid merged: {len(full_grid)} states\n")
    
    if full_grid:
        for i, sol in enumerate(full_grid[:3]):
            print(f"  Full Sol {i}: {len(sol.edges)} edges, loops={sol.internal_loops}")
            
            # Detailed degree analysis
            degrees = collections.defaultdict(int)
            for u, v in sol.edges:
                degrees[u] += 1
                degrees[v] += 1
            
            # Count degrees
            deg_counts = collections.Counter(degrees.values())
            print(f"    Degree distribution: {dict(deg_counts)}")
            
            odd_degree_nodes = [n for n, d in degrees.items() if d % 2 != 0]
            print(f"    Nodes with odd degree: {len(odd_degree_nodes)}")
            
            if not odd_degree_nodes:
                # Check connectivity
                val_uf = UnionFind()
                nodes = list(degrees.keys())
                for u, v in sol.edges:
                    val_uf.union(u, v)
                roots = set(val_uf.find(n) for n in nodes)
                print(f"    Connected components: {len(roots)}")
                if len(roots) == 1:
                    print(f"    *** VALID SOLUTION ***")
    else:
        print("No full grid solutions!")
    
    # Now run the full validation
    print("\n=== GLOBAL LOOP ENFORCEMENT ===")
    valid_count = 0
    for sol in full_grid:
        edges = sol.edges
        if not edges: 
            continue
        
        degrees = collections.defaultdict(int)
        for u, v in edges:
            degrees[u] += 1
            degrees[v] += 1
        
        is_closed = True
        for d in degrees.values():
            if d % 2 != 0:
                is_closed = False
                break
        
        if is_closed:
            val_uf = UnionFind()
            nodes = list(degrees.keys())
            for u, v in edges:
                val_uf.union(u, v)
            roots = set(val_uf.find(n) for n in nodes)
            if len(roots) == 1:
                valid_count += 1
    
    print(f"Valid solutions after global enforcement: {valid_count}")

if __name__ == "__main__":
    debug_solution_validation()
