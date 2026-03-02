import time
from logic.game_state import GameState

def test_headless_ai_vs_ai():
    print("Testing AI vs AI using Greedy (P1) vs DP Backtracking (P2)")
    
    # Initialize GameState with AI vs AI
    gs = GameState(
        rows=3, 
        cols=3, 
        difficulty="Easy", 
        game_mode="ai_vs_ai", 
        solver_strategy={"p1": "greedy", "p2": "dp_backtracking"},
        generator_type="dp"
    )
    
    print(f"Goal: {len(gs.solution_edges)} edges to form loop.")
    print(f"Initial edges: {len(gs.graph.edges)}")

    moves = 0
    max_moves = 50

    while not gs.game_over and moves < max_moves:
        current_turn = gs.turn
        move, source_str, solver, fallback = gs.get_next_cpu_move()
        
        if move is None:
            print(f"Turn {moves}: {current_turn} ({source_str}) has NO MOVES")
            break
            
        print(f"Turn {moves}: {current_turn} ({source_str}) selected move {move}")
        success = gs.make_move(move[0], move[1], is_cpu=True)
        if hasattr(solver, "register_move"):
            solver.register_move(move)
            
        if not success:
            print("Move failed!")
            break
            
        moves += 1

    print(f"Game over status: {gs.game_over}, Winner: {gs.winner}")

if __name__ == '__main__':
    test_headless_ai_vs_ai()
