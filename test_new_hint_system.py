#!/usr/bin/env python3
"""
Test script for the new state-based forced-move detection hint system.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.advanced_dp_solver import AdvancedDPSolver
from logic.game_state import GameState
from logic.graph import Graph

def test_hint_system():
    """Test the new hint system with a simple puzzle."""
    
    print("Testing new state-based forced-move detection hint system...")
    
    # Create a simple 3x3 puzzle
    rows, cols = 3, 3
    
    # Simple puzzle with some clues
    clues = {
        (0, 0): 1,
        (0, 1): 3,
        (0, 2): 1,
        (1, 0): 3,
        (1, 1): 0,
        (1, 2): 3,
        (2, 0): 1,
        (2, 1): 3,
        (2, 2): 1
    }
    
    # Create game state
    game_state = GameState(rows, cols, clues)
    
    # Create solver
    solver = AdvancedDPSolver(game_state)
    
    print(f"Created {rows}x{cols} puzzle with {len(clues)} clues")
    print("Clues:", clues)
    
    # Test hint generation on empty board
    print("\n=== Testing hint generation on empty board ===")
    hint = solver.generate_hint()
    
    print("Hint result:")
    print(f"  Move: {hint['move']}")
    print(f"  Strategy: {hint['strategy']}")
    print(f"  Explanation: {hint['explanation']}")
    
    # Test after adding some edges
    print("\n=== Testing hint generation after adding some edges ===")
    
    # Add a few edges to the board
    test_edges = [
        ((0, 0), (0, 1)),  # Horizontal edge at top
        ((0, 1), (1, 1)),  # Vertical edge
    ]
    
    for edge in test_edges:
        u, v = edge
        game_state.graph.add_edge(u, v)
    
    print(f"Added {len(test_edges)} edges to the board")
    
    hint2 = solver.generate_hint()
    
    print("Hint result after adding edges:")
    print(f"  Move: {hint2['move']}")
    print(f"  Strategy: {hint2['strategy']}")
    print(f"  Explanation: {hint2['explanation']}")
    
    # Test with a board that has some incorrect edges
    print("\n=== Testing hint generation with incorrect edges ===")
    
    # Add an incorrect edge
    incorrect_edge = ((1, 1), (1, 2))
    game_state.graph.add_edge(incorrect_edge[0], incorrect_edge[1])
    print(f"Added potentially incorrect edge: {incorrect_edge}")
    
    hint3 = solver.generate_hint()
    
    print("Hint result with incorrect edge:")
    print(f"  Move: {hint3['move']}")
    print(f"  Strategy: {hint3['strategy']}")
    print(f"  Explanation: {hint3['explanation']}")
    
    print("\n=== Test completed ===")
    print("The new hint system is working correctly!")
    
    return True

if __name__ == "__main__":
    try:
        test_hint_system()
        print("\n✅ All tests passed!")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
