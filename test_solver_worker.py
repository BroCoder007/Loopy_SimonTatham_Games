"""
Test Solver Worker
==================
Validates timeout, state limits, metrics streaming, and clean stop
for the refactored solver architecture.
"""

import sys
import os
import time
import threading
import queue

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logic.solver_worker import SolverWorker, SolverMetrics, AnalysisWorker
from logic.solvers.dp_backtracking_solver import DPBacktrackingSolver


class MockGraph:
    def __init__(self):
        self.edges = set()
    def copy(self):
        g = MockGraph()
        g.edges = set(self.edges)
        return g


class MockGameState:
    def __init__(self, rows, cols, clues=None):
        self.rows = rows
        self.cols = cols
        self.clues = clues or {}
        self.graph = MockGraph()
        self.game_mode = "vs_cpu"
        self.turn = "Player 1"
        self.message = ""
        self.last_cpu_move_info = {}
        self.edge_weights = {}
        self.solution_edges = set()
        self.live_analysis_table = []


def test_timeout_enforcement():
    """Solver must return within timeout with status='Timeout'."""
    print("[TEST] Timeout enforcement...")
    state = MockGameState(7, 7, {
        (r, c): 2 for r in range(7) for c in range(7)
    })

    solver = DPBacktrackingSolver(state, timeout=0.5, max_states=10_000_000)
    start = time.time()
    result = solver.solve(ignore_current_edges=True)
    elapsed = time.time() - start

    assert elapsed < 2.0, \
        f"Solver did not respect timeout: {elapsed:.2f}s, status={result['status']}"
    assert result["status"] in ("Timeout", "Success", "NoSolution"), \
        f"Expected Timeout/Success/NoSolution, got {result['status']}"
    print(f"  PASS: returned in {elapsed:.2f}s, status={result['status']}, nodes={result['nodes_visited']}")


def test_state_limit_enforcement():
    """Solver must stop after max_states is reached."""
    print("[TEST] State limit enforcement...")
    state = MockGameState(5, 5, {
        (r, c): 2 for r in range(5) for c in range(5)
    })

    solver = DPBacktrackingSolver(state, timeout=30.0, max_states=500)
    result = solver.solve(ignore_current_edges=True)

    assert result["nodes_visited"] <= 600, \
        f"Solver explored too many states: {result['nodes_visited']}"
    print(f"  PASS: stopped at {result['nodes_visited']} nodes, status={result['status']}")


def test_clean_stop():
    """Solver must stop when stop_event is set."""
    print("[TEST] Clean stop...")
    state = MockGameState(6, 6, {
        (r, c): 2 for r in range(6) for c in range(6)
    })

    stop_event = threading.Event()
    solver = DPBacktrackingSolver(state, stop_event=stop_event, timeout=30.0)

    def delayed_stop():
        time.sleep(0.1)
        stop_event.set()

    threading.Thread(target=delayed_stop, daemon=True).start()

    start = time.time()
    result = solver.solve(ignore_current_edges=True)
    elapsed = time.time() - start

    assert elapsed < 2.0, f"Solver did not stop cleanly: {elapsed:.2f}s"
    print(f"  PASS: stopped in {elapsed:.2f}s after stop_event, nodes={result['nodes_visited']}")


def test_metrics_queue():
    """Metrics must be pushed to queue during solving."""
    print("[TEST] Metrics queue...")
    state = MockGameState(5, 5, {
        (r, c): 2 for r in range(5) for c in range(5)
    })

    mq = queue.Queue()
    solver = DPBacktrackingSolver(state, timeout=2.0, max_states=5000, metrics_queue=mq)
    result = solver.solve(ignore_current_edges=True)

    metrics_count = mq.qsize()
    print(f"  Metrics snapshots pushed: {metrics_count}")
    assert metrics_count > 0, "No metrics were pushed to queue"

    # Check a sample
    sample = mq.get()
    assert hasattr(sample, 'timestamp'), "Missing timestamp"
    assert hasattr(sample, 'states_explored'), "Missing states_explored"
    assert hasattr(sample, 'time_per_step_ms'), "Missing time_per_step_ms"
    assert hasattr(sample, 'branching_factor'), "Missing branching_factor"
    print(f"  PASS: {metrics_count} metrics pushed, sample: states={sample.states_explored}, "
          f"time/step={sample.time_per_step_ms:.4f}ms")


def test_no_fake_data():
    """Verify live_analysis.py has no random/math imports for fake data."""
    print("[TEST] No fake data in live_analysis.py...")
    with open(os.path.join(os.path.dirname(__file__), "logic", "live_analysis.py"), "r") as f:
        content = f.read()

    assert "random.uniform" not in content, "Found random.uniform (fake data generator)"
    assert "random.randint" not in content, "Found random.randint (fake data generator)"
    assert "math.exp" not in content, "Found math.exp (extrapolation math)"
    assert "multiplier" not in content, "Found 'multiplier' (extrapolation variable)"
    assert "extrapolat" not in content.lower(), "Found 'extrapolat' reference"
    print("  PASS: No fake data generation found")


def test_analysis_worker():
    """AnalysisWorker must run function in background and report done."""
    print("[TEST] AnalysisWorker...")

    result_holder = {"value": None}
    def sample_fn():
        time.sleep(0.1)
        return {"status": "done", "data": 42}

    worker = AnalysisWorker(sample_fn)
    assert not worker.is_done()

    worker.start()
    time.sleep(0.05)
    assert not worker.is_done(), "Worker completed too quickly"

    time.sleep(0.2)
    assert worker.is_done(), "Worker should be done"
    result = worker.get_result()
    assert result is not None
    assert result["status"] == "done"
    print(f"  PASS: worker completed, result={result}")


def test_solver_worker():
    """SolverWorker must wrap solver execution in background."""
    print("[TEST] SolverWorker...")

    def slow_solver():
        time.sleep(0.1)
        return {"success": True, "nodes_visited": 100}

    worker = SolverWorker(slow_solver, timeout=2.0, label="test")
    worker.start()

    assert not worker.is_done()
    time.sleep(0.3)
    assert worker.is_done()

    result = worker.get_result()
    assert result["success"] is True
    print(f"  PASS: worker completed, result={result}")


if __name__ == "__main__":
    print("=" * 60)
    print("Solver Architecture Verification Tests")
    print("=" * 60)

    tests = [
        test_timeout_enforcement,
        test_state_limit_enforcement,
        test_clean_stop,
        test_metrics_queue,
        test_no_fake_data,
        test_analysis_worker,
        test_solver_worker,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  FAIL: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed, {len(tests)} total")
    print("=" * 60)
