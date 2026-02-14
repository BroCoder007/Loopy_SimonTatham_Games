#!/usr/bin/env python3
"""
Debug why solution_edges attribute is not being detected
"""

from logic.game_state import GameState

def debug_solution_attr():
    print('Checking solution edges attribute...')
    game = GameState(rows=4, cols=4, difficulty='Easy', game_mode='vs_cpu', solver_strategy='dynamic_programming')
    
    print(f'hasattr solution_edges: {hasattr(game, "solution_edges")}')
    print(f'solution_edges exists: {hasattr(game, "solution_edges") and game.solution_edges}')
    print(f'solution_edges type: {type(getattr(game, "solution_edges", None))}')
    print(f'solution_edges value: {getattr(game, "solution_edges", None)}')
    
    solution_edges = getattr(game, 'solution_edges', None)
    if solution_edges is None:
        print('solution_edges is None')
    elif isinstance(solution_edges, set) and len(solution_edges) == 0:
        print('solution_edges is empty set')
    else:
        print(f'solution_edges has {len(solution_edges)} elements')

if __name__ == '__main__':
    debug_solution_attr()
