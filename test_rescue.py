import sys
from logic.game_state import GameState

with open("rescue_log.txt", "w") as f:
    f.write("Starting test...\n")
    game = GameState(rows=3, cols=3, game_mode="expert", generator_type="dp")
    
    # Let's inspect D&C explicitly
    from logic.solvers.divide_conquer_solver import DivideConquerSolver
    solver = DivideConquerSolver(game)

    f.write(f"Solution size: {len(game.solution_edges)}\n")

    bad_edge = None
    all_edges = []
    for r in range(4):
        for c in range(3):
            all_edges.append(tuple(sorted(((r, c), (r, c+1)))))
    for r in range(3):
        for c in range(4):
            all_edges.append(tuple(sorted(((r, c), (r+1, c)))))

    for e in all_edges:
        if e not in game.solution_edges:
            bad_edge = e
            break

    f.write(f"Adding bad edge: {bad_edge}\n")
    game.make_move(bad_edge[0], bad_edge[1], is_cpu=False)
    
    # call _run_divide_and_conquer directly to see what it returns
    move, reason = solver._run_divide_and_conquer()
    
    f.write(f"D&C returned: move={move}, reason={reason}\n")
    
    # Debug DPBacktrackingSolver directly
    from logic.solvers.dp_backtracking_solver import DPBacktrackingSolver
    bk_solver = DPBacktrackingSolver(game)
    res1 = bk_solver.solve()
    f.write(f"First solve: success={res1['success']}\n")
    res2 = bk_solver.solve(ignore_current_edges=True)
    f.write(f"Second solve (ignore edges): success={res2['success']}, size={len(res2['edges']) if 'edges' in res2 else 0}\n")
