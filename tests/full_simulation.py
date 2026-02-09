
import sys
import os
import random

# Add project root to path
sys.path.append(os.getcwd())

from logic.game_state import GameState
from logic.greedy_cpu import GreedyCPU

def log(msg):
    print(f"[TEST] {msg}")

def test_vs_cpu_mode():
    log("Testing Normal VS CPU Mode...")
    gs = GameState(rows=4, cols=4, difficulty="Easy", game_mode="vs_cpu")
    
    # 1. Check Initial State
    assert gs.turn == "Player 1 (Human)", "Initial turn should be Human"
    assert not gs.game_over, "Game should not be over"
    
    # 2. Human Move
    valid_moves = gs.get_all_valid_moves()
    if not valid_moves:
        log("No valid moves generated!")
        return
        
    u, v = valid_moves[0]
    log(f"Human moving at {u}-{v}")
    gs.make_move(u, v, is_cpu=False)
    
    assert gs.turn == "Player 2 (CPU)", "Turn should switch to CPU"
    assert tuple(sorted((u, v))) in gs.graph.edges, "Edge should be added"
    
    # 3. CPU Turn
    log("Triggering CPU Move...")
    cpu_move = gs.cpu.make_move()
    if cpu_move:
        u_cpu, v_cpu = cpu_move
        gs.make_move(u_cpu, v_cpu, is_cpu=True)
        log(f"CPU moved at {u_cpu}-{v_cpu}")
        assert gs.turn == "Player 1 (Human)", "Turn should switch back to Human"
        
    # 4. Undo Test
    log("Testing Undo...")
    edges_before = len(gs.graph.edges)
    gs.undo() # Undo CPU
    gs.undo() # Undo Human
    edges_after = len(gs.graph.edges)
    assert edges_after == edges_before - 2, "Undo should remove 2 edges"
    
    # 5. Hint Test
    log("Testing Hint...")
    hint_move, reason = gs.get_hint()
    log(f"Hint received: {hint_move} because {reason}")
    assert hint_move is not None or "No solution" in reason
    
    log("Vs CPU Mode Passed.")

def test_expert_mode_energy():
    log("Testing Expert Mode (Energy)...")
    gs = GameState(rows=4, cols=4, difficulty="Hard", game_mode="expert")
    
    initial_energy = gs.energy
    initial_cpu_energy = gs.energy_cpu
    log(f"Initial Energy: Human={initial_energy}, CPU={initial_cpu_energy}")
    
    # 1. Human Move (Cost Check)
    valid_moves = gs.get_all_valid_moves()
    u, v = valid_moves[0]
    edge_key = tuple(sorted((u, v)))
    cost = gs.edge_weights.get(edge_key, 0)
    
    gs.make_move(u, v, is_cpu=False)
    assert gs.energy == initial_energy - cost, f"Energy should decrease by {cost}"
    
    # 2. Reset Turn to Human (Because make_move switched it to CPU)
    # The simulation wants to test "Human adds, then Human removes same edge"
    # In real game, CPU would move in between.
    # Force switch for test logic validity
    gs.turn = "Player 1 (Human)"
    
    # 3. Remove Move (Refund)
    gs.make_move(u, v, is_cpu=False) # Remove same edge
    assert gs.energy == initial_energy, f"Energy should be refunded. Got {gs.energy}, Expected {initial_energy}"
    
    log("Expert Energy Logic Passed.")
    
def test_cpu_loop_prevention():
    log("Testing CPU Loop Prevention...")
    gs = GameState(rows=4, cols=4, game_mode="vs_cpu")
    
    # Fake history: Human just removed edge X
    u, v = (0,0), (0,1)
    gs.undo_stack.append(((0,0), (0,1), "remove"))
    
    # Ask CPU to score this specific edge
    score = gs.cpu.calculate_smart_score((u, v))
    log(f"Score for just-removed edge: {score}")
    assert score <= -2000, "CPU should strictly penalize re-adding just-removed edge"
    
    log("Loop Prevention Passed.")

if __name__ == "__main__":
    try:
        test_vs_cpu_mode()
        print("-" * 20)
        test_expert_mode_energy()
        print("-" * 20)
        test_cpu_loop_prevention()
        print("=" * 20)
        print("ALL TESTS PASSED")
    except Exception as e:
        print(f"TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
