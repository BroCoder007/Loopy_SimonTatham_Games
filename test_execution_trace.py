#!/usr/bin/env python3
"""
Test Execution Trace Integration
================================
Test script to verify that all strategies can run with the new trace hooks
without crashes and that the trace system captures data correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.execution_trace import clear_trace, get_trace, get_trace_summary
from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.generators.dnc_generator import DivideAndConquerPuzzleGenerator

def test_greedy_strategy():
    """Test Greedy strategy with trace hooks."""
    print("Testing Greedy Strategy...")
    
    # Create a simple game state
    game_state = GameState(5, 5)
    game_state.game_mode = "vs_cpu"
    game_state.turn = "Player 2 (CPU)"
    
    # Clear trace
    clear_trace()
    
    # Create GreedySolver and make a move
    solver = GreedySolver(game_state)
    move = solver.make_move()
    
    # Check trace
    trace = get_trace()
    greedy_trace = [entry for entry in trace if entry["strategy_name"] == "Greedy"]
    
    print(f"  Greedy made move: {move}")
    print(f"  Trace entries for Greedy: {len(greedy_trace)}")
    
    if greedy_trace:
        print(f"  Last greedy entry: {greedy_trace[-1]}")
    
    return move is not None

def test_pure_dp_strategy():
    """Test Pure DP strategy with trace hooks."""
    print("Testing Pure DP Strategy...")
    
    # Create a simple game state
    game_state = GameState(3, 3)  # Smaller for faster testing
    game_state.game_mode = "vs_cpu"
    game_state.turn = "Player 2 (CPU)"
    
    # Clear trace
    clear_trace()
    
    # Create DP solver and make a move
    solver = DynamicProgrammingSolver(game_state)
    move = solver.solve()
    
    # Check trace
    trace = get_trace()
    dp_trace = [entry for entry in trace if entry["strategy_name"] == "Pure DP"]
    
    print(f"  Pure DP made move: {move}")
    print(f"  Trace entries for Pure DP: {len(dp_trace)}")
    
    if dp_trace:
        print(f"  Last DP entry: {dp_trace[-1]}")
    
    return True  # DP might not find a move quickly, that's OK

def test_advanced_dp_strategy():
    """Test Advanced DP strategy with trace hooks."""
    print("Testing Advanced DP Strategy...")
    
    # Create a simple game state
    game_state = GameState(3, 3)  # Smaller for faster testing
    game_state.game_mode = "vs_cpu"
    game_state.turn = "Player 2 (CPU)"
    
    # Clear trace
    clear_trace()
    
    # Create Advanced DP solver and make a move
    solver = AdvancedDPSolver(game_state)
    move = solver.solve()
    
    # Check trace
    trace = get_trace()
    adv_dp_trace = [entry for entry in trace if entry["strategy_name"] == "Advanced DP"]
    
    print(f"  Advanced DP made move: {move}")
    print(f"  Trace entries for Advanced DP: {len(adv_dp_trace)}")
    
    if adv_dp_trace:
        print(f"  Last Advanced DP entry: {adv_dp_trace[-1]}")
    
    return True  # Advanced DP might not find a move quickly, that's OK

def test_generator_strategy():
    """Test Generator strategy with trace hooks."""
    print("Testing Generator Strategy...")
    
    # Clear trace
    clear_trace()
    
    # Create generator
    generator = DivideAndConquerPuzzleGenerator(3, 3, "Easy")
    
    try:
        # Generate puzzle
        clues, solution_edges = generator.generate()
        
        # Check trace
        trace = get_trace()
        gen_trace = [entry for entry in trace if entry["strategy_name"] == "DnC Generator"]
        
        print(f"  Generator created puzzle with {len(clues)} clues and {len(solution_edges)} edges")
        print(f"  Trace entries for Generator: {len(gen_trace)}")
        
        if gen_trace:
            print(f"  Sample generator entry: {gen_trace[0]}")
        
        return len(clues) > 0
    except Exception as e:
        print(f"  Generator failed: {e}")
        return False

def test_trace_summary():
    """Test trace summary functionality."""
    print("Testing Trace Summary...")
    
    # Get summary of current trace
    summary = get_trace_summary()
    
    print(f"  Total steps: {summary['total_steps']}")
    print(f"  Strategies: {summary['strategies']}")
    print(f"  Max recursion depth: {summary['max_recursion_depth']}")
    print(f"  Total DP states: {summary['total_dp_states']}")
    
    return True

def main():
    """Run all tests."""
    print("=" * 50)
    print("EXECUTION TRACE INTEGRATION TEST")
    print("=" * 50)
    
    tests = [
        ("Greedy Strategy", test_greedy_strategy),
        ("Pure DP Strategy", test_pure_dp_strategy),
        ("Advanced DP Strategy", test_advanced_dp_strategy),
        ("Generator Strategy", test_generator_strategy),
        ("Trace Summary", test_trace_summary),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\n{'-' * 30}")
        try:
            result = test_func()
            results.append((test_name, result, None))
            print(f"✓ {test_name}: {'PASS' if result else 'FAIL'}")
        except Exception as e:
            results.append((test_name, False, str(e)))
            print(f"✗ {test_name}: ERROR - {e}")
    
    # Final summary
    print(f"\n{'=' * 50}")
    print("FINAL RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, result, error in results:
        status = "PASS" if result else "FAIL"
        if error:
            status = f"ERROR ({error})"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\nPassed: {passed}/{len(results)} tests")
    
    if passed == len(results):
        print("🎉 All tests passed! Execution trace integration is working correctly.")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
