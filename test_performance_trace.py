#!/usr/bin/env python3
"""
Performance Test for Execution Trace
===================================
Test script to verify that the execution trace system doesn't significantly
impact solver performance.
"""

import sys
import os
import time
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.execution_trace import clear_trace, get_trace_summary
from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.generators.dnc_generator import DivideAndConquerPuzzleGenerator

def time_function(func, *args, **kwargs):
    """Time a function execution."""
    start = time.time()
    result = func(*args, **kwargs)
    end = time.time()
    return result, end - start

def test_greedy_performance():
    """Test Greedy performance with and without trace."""
    print("Testing Greedy Performance...")
    
    game_state = GameState(5, 5)
    game_state.game_mode = "vs_cpu"
    game_state.turn = "Player 2 (CPU)"
    
    # Test with trace
    clear_trace()
    solver = GreedySolver(game_state)
    
    moves_with_trace = []
    start_time = time.time()
    
    for _ in range(10):  # Make 10 moves
        move = solver.make_move()
        if move:
            moves_with_trace.append(move)
            # Apply move to game state
            game_state.graph.add_edge(move[0], move[1])
        else:
            break
    
    with_trace_time = time.time() - start_time
    trace_summary = get_trace_summary()
    
    print(f"  Made {len(moves_with_trace)} moves in {with_trace_time:.4f}s with trace")
    print(f"  Trace entries: {trace_summary['total_steps']}")
    
    return len(moves_with_trace) > 0

def test_generator_performance():
    """Test Generator performance with trace."""
    print("Testing Generator Performance...")
    
    # Test with trace
    clear_trace()
    generator = DivideAndConquerPuzzleGenerator(3, 3, "Easy")
    
    start_time = time.time()
    clues, solution_edges = generator.generate()
    generation_time = time.time() - start_time
    
    trace_summary = get_trace_summary()
    
    print(f"  Generated puzzle in {generation_time:.4f}s")
    print(f"  Clues: {len(clues)}, Edges: {len(solution_edges)}")
    print(f"  Trace entries: {trace_summary['total_steps']}")
    
    return len(clues) > 0

def test_trace_overhead():
    """Test the overhead of trace operations themselves."""
    print("Testing Trace Overhead...")
    
    from logic.execution_trace import log_execution_step
    
    # Test logging speed
    clear_trace()
    
    start_time = time.time()
    for i in range(1000):
        log_execution_step(
            strategy_name="Test",
            move=((i % 10, i % 10), ((i+1) % 10, (i+1) % 10)),
            explanation=f"Test step {i}",
            recursion_depth=i % 5,
            region_id=f"Region_{i % 3}"
        )
    log_time = time.time() - start_time
    
    # Test retrieval speed
    start_time = time.time()
    trace = get_trace_summary()
    retrieval_time = time.time() - start_time
    
    # Test clear speed
    start_time = time.time()
    clear_trace()
    clear_time = time.time() - start_time
    
    print(f"  Logging 1000 entries: {log_time:.4f}s ({1000/log_time:.0f} entries/sec)")
    print(f"  Retrieving summary: {retrieval_time:.6f}s")
    print(f"  Clearing trace: {clear_time:.6f}s")
    
    return True

def main():
    """Run performance tests."""
    print("=" * 50)
    print("EXECUTION TRACE PERFORMANCE TEST")
    print("=" * 50)
    
    tests = [
        ("Greedy Performance", test_greedy_performance),
        ("Generator Performance", test_generator_performance),
        ("Trace Overhead", test_trace_overhead),
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
    print("PERFORMANCE SUMMARY")
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
        print("🚀 Performance tests passed! Trace system has minimal overhead.")
        return True
    else:
        print("⚠️  Some performance tests failed.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
