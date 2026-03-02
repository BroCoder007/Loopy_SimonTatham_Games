"""
Tests for Constraint Propagation + Backtracking Solver
======================================================
Verifies the ConstraintPropagator deduction engine and
ConstraintBacktrackingSolver integration.

Run with: python tests/test_constraint_propagation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from logic.constraint_propagation import (
    ConstraintPropagator, ConstraintBacktrackingSolver,
    ON, OFF, UNKNOWN
)
from logic.graph import Graph


# ── Helper: minimal game state mock ──────────────────────────────────
class MockGameState:
    def __init__(self, rows, cols, clues, edges=None):
        self.rows = rows
        self.cols = cols
        self.clues = clues
        self.graph = Graph(rows, cols)
        self.game_mode = "vs_cpu"
        self.turn = "Player 1 (Human)"
        self.message = ""
        self.edge_weights = {}
        self.solution_edges = set()
        self.live_analysis_table = []
        self.last_cpu_move_info = None
        if edges:
            for u, v in edges:
                self.graph.add_edge(u, v)


results = []


def run_test(name, fn):
    try:
        ok = fn()
        tag = "PASS" if ok else "FAIL"
        results.append(tag)
        print(f"  {tag} — {name}")
    except Exception as ex:
        results.append("FAIL")
        print(f"  FAIL — {name}: {ex}")


# ── Test 1: Clue 0 forces all 4 edges OFF ────────────────────────────
def test_clue_zero():
    """Cell with clue 0 → all surrounding edges must be OFF."""
    prop = ConstraintPropagator(3, 3, {(1, 1): 0})
    valid = prop.propagate()
    if not valid:
        return False
    cell_edges = prop.cell_edges[(1, 1)]
    return all(prop.edge_state[e] == OFF for e in cell_edges)


# ── Test 2: Clue 4 forces all 4 edges ON ─────────────────────────────
def test_clue_four():
    """Cell with clue 4 → all surrounding edges must be ON."""
    prop = ConstraintPropagator(3, 3, {(1, 1): 4})
    valid = prop.propagate()
    if not valid:
        return False
    cell_edges = prop.cell_edges[(1, 1)]
    return all(prop.edge_state[e] == ON for e in cell_edges)


# ── Test 3: Partial clue deduction ────────────────────────────────────
def test_partial_clue():
    """Cell with clue 3 and 3 edges already ON → 4th edge forced OFF."""
    edges_on = set()
    cell_edges_for_1_1 = [
        tuple(sorted(((1, 1), (1, 2)))),   # top
        tuple(sorted(((2, 1), (2, 2)))),   # bottom
        tuple(sorted(((1, 1), (2, 1)))),   # left
    ]
    edges_on.update(cells for cells in cell_edges_for_1_1)
    prop = ConstraintPropagator(3, 3, {(1, 1): 3}, edges_on)
    valid = prop.propagate()
    if not valid:
        return False
    right_edge = tuple(sorted(((1, 2), (2, 2))))
    return prop.edge_state[right_edge] == OFF


# ── Test 4: Vertex degree constraint ─────────────────────────────────
def test_vertex_degree():
    """Vertex at degree 2 → all other edges forced OFF."""
    edges_on = {
        tuple(sorted(((1, 1), (1, 2)))),
        tuple(sorted(((1, 1), (2, 1)))),
    }
    prop = ConstraintPropagator(3, 3, {}, edges_on)
    valid = prop.propagate()
    if not valid:
        return False
    # Vertex (1,1) has degree 2, so other edges at (1,1) must be OFF
    up = tuple(sorted(((0, 1), (1, 1))))
    left = tuple(sorted(((1, 0), (1, 1))))
    return (prop.edge_state[up] == OFF and prop.edge_state[left] == OFF)


# ── Test 5: Contradiction detection ──────────────────────────────────
def test_contradiction():
    """Clue already exceeded → propagation returns False."""
    # Place 3 edges around a cell with clue 1 — contradiction
    edges_on = {
        tuple(sorted(((1, 1), (1, 2)))),
        tuple(sorted(((2, 1), (2, 2)))),
        tuple(sorted(((1, 1), (2, 1)))),
    }
    prop = ConstraintPropagator(3, 3, {(1, 1): 1}, edges_on)
    valid = prop.propagate()
    return valid == False  # Should detect contradiction


# ── Test 6: Solver produces a valid move ──────────────────────────────
def test_solver_move():
    """Solver on a small board with clue 4 should return a forced edge."""
    gs = MockGameState(3, 3, {(1, 1): 4})
    solver = ConstraintBacktrackingSolver(gs)
    move = solver.solve()
    if move is None:
        return False
    # The move should be one of the 4 edges around cell (1,1)
    cell_edges = [
        tuple(sorted(((1, 1), (1, 2)))),
        tuple(sorted(((2, 1), (2, 2)))),
        tuple(sorted(((1, 1), (2, 1)))),
        tuple(sorted(((1, 2), (2, 2)))),
    ]
    return move in cell_edges


# ── Test 7: Hint generation returns proper dict ───────────────────────
def test_hint_format():
    """generate_hint() returns a dict with move, strategy, explanation."""
    gs = MockGameState(3, 3, {(1, 1): 4})
    solver = ConstraintBacktrackingSolver(gs)
    hint = solver.generate_hint()
    if not isinstance(hint, dict):
        return False
    required_keys = {"move", "strategy", "explanation"}
    return required_keys.issubset(hint.keys())


# ── Test 8: Integration — GameState instantiation ─────────────────────
def test_integration():
    """GameState with solver_strategy='constraint_propagation' creates correct solver."""
    try:
        from logic.game_state import GameState
        gs = GameState(rows=3, cols=3, difficulty="Easy",
                       game_mode="vs_cpu",
                       solver_strategy="constraint_propagation")
        return isinstance(gs.cpu, ConstraintBacktrackingSolver)
    except Exception as ex:
        print(f"    Integration error: {ex}")
        return False


# ── Run all tests ────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Constraint Propagation + Backtracking Tests")
    print("=" * 50)

    run_test("T1: Clue 0 forces all edges OFF", test_clue_zero)
    run_test("T2: Clue 4 forces all edges ON", test_clue_four)
    run_test("T3: Partial clue deduction (3/3 ON → 4th OFF)", test_partial_clue)
    run_test("T4: Vertex degree 2 → others OFF", test_vertex_degree)
    run_test("T5: Contradiction detection", test_contradiction)
    run_test("T6: Solver returns valid move", test_solver_move)
    run_test("T7: Hint format correct", test_hint_format)
    run_test("T8: GameState integration", test_integration)

    print("=" * 50)
    passed = results.count("PASS")
    failed = results.count("FAIL")
    print(f"TOTAL: {passed} passed, {failed} failed out of {len(results)}")
    sys.exit(0 if failed == 0 else 1)
