#!/usr/bin/env python3
"""
Debug the full merge process for Q3 and Q4.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver, UnionFind
from logic.game_state import GameState
import collections

def debug_full_merge():
    """Debug the complete merge process."""
    
    print("=== DEBUG: Complete Q3+Q4 Merge Process ===\n")
    
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
    
    print(f"Q3: {len(q3_solutions)} solutions")
    print(f"Q4: {len(q4_solutions)} solutions\n")
    
    # Manually trace one valid merge: Q3[5] + Q4[0]
    if len(q3_solutions) > 5 and len(q4_solutions) > 0:
        l_sol = q3_solutions[5]
        r_sol = q4_solutions[0]
        
        print(f"Tracing Q3[5] + Q4[0]:")
        print(f"  v_right_mask: {l_sol.v_right_mask}")
        print(f"  v_left_mask: {r_sol.v_left_mask}")
        print(f"  Match: {l_sol.v_right_mask == r_sol.v_left_mask}")
        
        # Check degree validation
        v_shared = l_sol.v_right_mask
        num_nodes = len(l_sol.right_sig)
        
        print(f"\n  Seam nodes: {num_nodes}")
        print(f"  right_sig: {l_sol.right_sig}")
        print(f"  left_sig: {r_sol.left_sig}")
        
        all_valid = True
        for k in range(num_nodes):
            row = mid_r + k
            has_up = v_shared[k-1] if k > 0 else False
            has_down = v_shared[k] if k < len(v_shared) else False
            shared_count = (1 if has_up else 0) + (1 if has_down else 0)
            
            e_left = tuple(sorted(((row, mid_c-1), (row, mid_c))))
            l_in = 1 if e_left in l_sol.edges else 0
            
            e_right = tuple(sorted(((row, mid_c), (row, mid_c+1))))
            r_in = 1 if e_right in r_sol.edges else 0
            
            total_degree = shared_count + l_in + r_in
            
            print(f"\n  Node {k} (row {row}):")
            print(f"    shared_count={shared_count}, l_in={l_in}, r_in={r_in}")
            print(f"    total_degree={total_degree}")
            
            if total_degree % 2 != 0 or total_degree > 2:
                print(f"    *** INVALID ***")
                all_valid = False
        
        print(f"\n  All degrees valid: {all_valid}")
        
        if all_valid:
            # Check component merging
            print(f"\n  Component merging:")
            uf = UnionFind()
            
            for k in range(num_nodes):
                id_l = l_sol.right_sig[k]
                id_r = r_sol.left_sig[k]
                
                print(f"    Node {k}: id_l={id_l}, id_r={id_r}")
                
                if id_l > 0 and id_r > 0:
                    root_l = uf.find((0, id_l))
                    root_r = uf.find((1, id_r))
                    print(f"      Merging ({0}, {id_l})={root_l} with ({1}, {id_r})={root_r}")
                    if root_l != root_r:
                        uf.union(root_l, root_r)
            
            # Check loop formation
            print(f"\n  Loop detection:")
            all_participating_ids = set()
            for x in l_sol.top_sig + l_sol.bottom_sig + l_sol.left_sig + l_sol.right_sig:
                if x > 0: all_participating_ids.add((0, x))
            for x in r_sol.top_sig + r_sol.bottom_sig + r_sol.left_sig + r_sol.right_sig:
                if x > 0: all_participating_ids.add((1, x))
            
            print(f"    All participating IDs: {all_participating_ids}")
            
            final_roots = set()
            for pid in all_participating_ids:
                final_roots.add(uf.find(pid))
            print(f"    Final roots: {final_roots}")
            
            # Compute new signatures
            def resolve(side, old_sig):
                return tuple(
                    0 if x == 0 else uf.find((side, x)) 
                    for x in old_sig
                )
            
            res_l_top = resolve(0, l_sol.top_sig)
            res_r_top = resolve(1, r_sol.top_sig)
            print(f"\n    Top sigs: l={l_sol.top_sig}, r={r_sol.top_sig}")
            print(f"    Resolved: l={res_l_top}, r={res_r_top}")
            
            # Check for closed loops
            visible_roots = set()
            visible_roots.update(res_l_top)
            visible_roots.update(res_r_top)
            visible_roots.update(resolve(0, l_sol.bottom_sig))
            visible_roots.update(resolve(1, r_sol.bottom_sig))
            if 0 in visible_roots: visible_roots.remove(0)
            
            print(f"    Visible roots: {visible_roots}")
            
            loops_formed = 0
            for root in final_roots:
                if root not in visible_roots:
                    loops_formed += 1
                    print(f"    *** Loop formed: {root} ***")
            
            total_loops = l_sol.internal_loops + r_sol.internal_loops + loops_formed
            print(f"\n    Total loops: {total_loops}")
            print(f"    Q3 loops: {l_sol.internal_loops}, Q4 loops: {r_sol.internal_loops}, new: {loops_formed}")
    
    # Now try the actual merge
    print(f"\n\n=== Actual Merge Result ===")
    merged = solver._merge_horizontal(q3_solutions, q4_solutions, mid_c, mid_r)
    print(f"Merged states: {len(merged)}")
    
    if merged:
        for i, sol in enumerate(merged[:3]):
            print(f"\n  Merged[{i}]: {len(sol.edges)} edges, {sol.internal_loops} loops")
            print(f"    top_sig: {sol.top_sig}")
            print(f"    bottom_sig: {sol.bottom_sig}")
    else:
        print("  No merged states - checking why...")
        
        # Check merge stats
        for detail in solver._merge_stats.get('merge_details', []):
            if 'horizontal' in detail.get('type', ''):
                print(f"\n  Merge stats: {detail}")

if __name__ == "__main__":
    debug_full_merge()
