#!/usr/bin/env python3
"""
Debug when solution_edges gets set/checked
"""

from logic.game_state import GameState

def debug_timing():
    print('Checking solution edges timing...')
    game = GameState(rows=4, cols=4, difficulty='Easy', game_mode='vs_cpu', solver_strategy='dynamic_programming')
    
    print(f'After game init, solution_edges: {len(getattr(game, "solution_edges", set()))}')
    
    # Check what happens during DP solver init
    print('\nDuring DP solver init:')
    print(f'solution_edges before DP init: {len(getattr(game, "solution_edges", set()))}')
    
    # This should trigger the DP solver init
    dp_solver = game.cpu
    print(f'solution_edges after DP init: {len(getattr(game, "solution_edges", set()))}')

if __name__ == '__main__':
    debug_timing()
