
import sys
import os
import time

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.execution_trace import get_trace, clear_trace

def verify_trace():
    print("Starting Trace Verification...")
    try:
        # 1. Verify Generator Trace
        print("\n--- Testing DnCGenerator ---")
        clear_trace()
        # Use small grid to be fast
        gs = GameState(rows=4, cols=4, difficulty="Medium", generator_type="dnc")
        # GameState init calls _generate_clues which calls generator.generate()
        
        trace = get_trace()
        print(f"Generator Trace Length: {len(trace)}")
        
        has_region = any(t['strategy_name'] == 'DnC Generator' and 'Generating Region' in t['explanation'] for t in trace)
        has_merge = any(t['strategy_name'] == 'DnC Generator' and 'Merging' in t['explanation'] for t in trace)
        
        if has_region and has_merge:
            print("✅ Generator Trace: OK (Found Region and Merge steps)")
        else:
            print("❌ Generator Trace: MISSING steps")
            print([t['explanation'] for t in trace[:5]])
            
        # 2. Verify Greedy Solver
        print("\n--- Testing GreedySolver ---")
        clear_trace()
        gs = GameState(rows=4, cols=4, game_mode="vs_cpu", solver_strategy="greedy")
        # GreedySolver is wrapper. logic/greedy_cpu.py doesn't have trace, but wrapper does.
        # We need to make a move.
        
        # Force a move
        move = gs.cpu.solve(gs)
        # The wrapper's solve calls make_move which calls register_move which logs.
        # Wait, solve calls make_move?
        # verify greedy_solver.py: 
        # def solve(self, ...): move = self.make_move(); return move
        
        trace = get_trace()
        print(f"Greedy Trace Length: {len(trace)}")
        if len(trace) > 0 and trace[0]['strategy_name'] == 'Greedy':
            print("✅ Greedy Trace: OK")
        else:
            print("❌ Greedy Trace: FAILED")
            # print(trace)

        # 3. Verify Pure DP Solver
        print("\n--- Testing DynamicProgrammingSolver ---")
        clear_trace()
        gs = GameState(rows=3, cols=3, game_mode="vs_cpu", solver_strategy="dynamic_programming")
        
        # Trigger solution computation
        move = gs.cpu.solve(gs)
        
        trace = get_trace()
        print(f"DP Trace Length: {len(trace)}")
        if len(trace) > 0 and trace[0]['strategy_name'] == 'Pure DP':
            print(f"✅ DP Trace: OK (Count: {trace[0].get('dp_state_count')})")
        else:
            print("❌ DP Trace: FAILED")
            # print(trace)

        # 4. Verify Advanced DP Solver
        print("\n--- Testing AdvancedDPSolver ---")
        clear_trace()
        gs = GameState(rows=4, cols=4, game_mode="vs_cpu", solver_strategy="advanced_dp")
        
        # Trigger solve (this should trigger _compute_full_solution which logs regions)
        move = gs.cpu.solve(gs)
        
        trace = get_trace()
        print(f"Advanced DP Trace Length: {len(trace)}")
        
        has_region_dp = any(t['strategy_name'] == 'Advanced DP' and 'Solving Quadrant' in t['explanation'] for t in trace)
        has_merge_dp = any(t['strategy_name'] == 'Advanced DP' and 'Merging' in t['explanation'] for t in trace)
        has_move = any(t['strategy_name'] == 'Advanced DP' and t['move'] is not None for t in trace)
        
        if has_region_dp and has_merge_dp:
            print("✅ Advanced DP Structure: OK")
        else:
            print("❌ Advanced DP Structure: MISSING")
            
        if has_move:
            print("✅ Advanced DP Move: OK")
        else:
            print("❌ Advanced DP Move: MISSING (Maybe no move found?)")

    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR: {e}")

if __name__ == "__main__":
    verify_trace()
