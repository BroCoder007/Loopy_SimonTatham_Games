import sys
import os
import random
# Add project root to path
sys.path.append(os.getcwd())

from logic.game_state import GameState

def log(msg, passed=True):
    status = "[PASS]" if passed else "[FAIL]"
    print(f"{status} {msg}", flush=True)

def test_vs_cpu_mode():
    print("\n--- Testing VS CPU Mode ---")
    game = GameState(rows=4, cols=4, game_mode="vs_cpu")
    
    # 1. Check Initial State
    if game.turn == "Player 1 (Human)":
        log("Initial turn is Human")
    else:
        log(f"Initial turn is {game.turn}", False)
        
    # 2. Human Move
    moves = game.get_all_valid_moves()
    if not moves:
        log("No valid moves found", False)
        return
    
    u, v = moves[0]
    success = game.make_move(u, v, is_cpu=False)
    if success and game.turn == "Player 2 (CPU)":
        log("Human move successful, turn switched to CPU")
    else:
        log(f"Human move failed or turn didn't switch. Success: {success}, Turn: {game.turn}", False)

    # 3. CPU Move
    if game.cpu:
        candidates, best_move = game.cpu.decide_move()
        if best_move:
            u, v = best_move
            success = game.make_move(u, v, is_cpu=True)
            if success and game.turn == "Player 1 (Human)":
                log("CPU move successful, turn switched to Human")
            else:
                 log(f"CPU move failed. Success: {success}, Turn: {game.turn}", False)
        else:
            log("CPU could not decide a move", False)
    else:
        log("CPU object not initialized in vs_cpu mode", False)

def test_expert_mode():
    print("\n--- Testing Expert Mode ---")
    game = GameState(rows=4, cols=4, game_mode="expert")
    
    # 1. Check Energy
    initial_energy = game.energy
    initial_cpu_energy = game.energy_cpu
    
    if initial_energy == initial_cpu_energy and initial_energy > 0:
        log(f"Initial Energy Set Correctly: {initial_energy}")
    else:
        log(f"Initial Energy Mismatch or Zero. Human: {initial_energy}, CPU: {initial_cpu_energy}", False)
        
    # 2. Human Move & Energy Deduction
    moves = game.get_all_valid_moves()
    u, v = moves[0]
    edge_key = tuple(sorted((u, v)))
    cost = game.edge_weights.get(edge_key, 0)
    
    success = game.make_move(u, v, is_cpu=False)
    
    # Expected energy
    if success:
        if game.energy == initial_energy - cost:
            log(f"Human Energy deducted correctly (Cost {cost})")
        else:
            log(f"Human Energy Error. Expected {initial_energy - cost}, Got {game.energy}", False)
            
        # 3. Undo & Refund
        game.undo()
        if game.energy == initial_energy:
            log("Energy Refunded on Undo")
        else:
            log(f"Energy Refund Failed. Expected {initial_energy}, Got {game.energy}", False)
            
    # 4. CPU Energy
    # Redo the move to switch turn
    game.redo()
    
    # CPU Turn
    candidates, best_move = game.cpu.decide_move() # Expert CPU logic uses energy
    if best_move:
        u_cpu, v_cpu = best_move
        edge_key_cpu = tuple(sorted((u_cpu, v_cpu)))
        cost_cpu = game.edge_weights.get(edge_key_cpu, 0)
        
        success = game.make_move(u_cpu, v_cpu, is_cpu=True)
        if game.energy_cpu == initial_cpu_energy - cost_cpu:
             log(f"CPU Energy deducted correctly (Cost {cost_cpu})")
        else:
             log(f"CPU Energy Error. Expected {initial_cpu_energy - cost_cpu}, Got {game.energy_cpu}", False)

def test_pvp_mode():
    print("\n--- Testing PvP Mode ---")
    game = GameState(rows=4, cols=4, game_mode="two_player")
    
    if game.turn == "Player 1":
        log("Initial turn is Player 1")
    
    moves = game.get_all_valid_moves()
    game.make_move(moves[0][0], moves[0][1])
    
    if game.turn == "Player 2":
        log("Turn switched to Player 2")
    else:
        log(f"Turn Switch Failed: {game.turn}", False)
        
    if game.cpu is None:
        log("CPU is None in PvP")
    else:
        log("CPU initialized in PvP (Unexpected)", False)

def test_save_load():
    print("\n--- Testing Save/Load ---")
    game = GameState(rows=4, cols=4, game_mode="vs_cpu")
    moves = game.get_all_valid_moves()
    u, v = moves[0]
    game.make_move(u, v, is_cpu=False) # Human Move
    
    # Save
    save_file = "qa_test_save.bin"
    if os.path.exists(save_file):
        os.remove(save_file)
        
    game.save_game(save_file)
    
    # New Game & Load
    new_game = GameState()
    success = new_game.load_game(save_file)
    
    if success:
        log("Game loaded successfully")
        # Check integrity
        if new_game.rows == 4 and new_game.cols == 4:
             log("Board dimensions preserved")
        else:
             log("Board dimensions mismatch", False)
             
        if len(new_game.undo_stack) == 1:
            log("Undo stack preserved")
        else:
            log(f"Undo stack mismatch. Expected 1, Got {len(new_game.undo_stack)}", False)
            
        if new_game.turn == "Player 2 (CPU)":
            log("Turn state preserved")
        else:
            log(f"Turn state mismatch. Got {new_game.turn}", False)
            
    else:
        log("Load Failed", False)
    
    # Cleanup
    if os.path.exists(save_file):
        os.remove(save_file)

def test_hints():
    print("\n--- Testing Hints ---")
    game = GameState(rows=5, cols=5, game_mode="vs_cpu")
    # Just check if it crashes
    try:
        move, reason = game.get_hint()
        log(f"Hint function returned: {reason}")
    except Exception as e:
        log(f"Hint function crashed: {e}", False)

if __name__ == "__main__":
    try:
        test_vs_cpu_mode()
        test_pvp_mode()
        test_expert_mode()
        test_save_load()
        test_hints()
        print("\nQA Tests Completed.")
    except Exception as e:
        print(f"\nCRITICAL ERROR IN TEST SCRIPT: {e}")
        import traceback
        traceback.print_exc()
