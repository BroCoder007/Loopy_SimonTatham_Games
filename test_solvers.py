import time
from logic.graph import Graph
from logic.validators import check_win_condition
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver
from logic.solvers.advanced_dp_solver import AdvancedDPSolver

class MockUIState:
    def __init__(self, rows, cols, clues):
        self.rows = rows
        self.cols = cols
        self.clues = clues
        self.graph = Graph(rows, cols)
        self.game_mode = "expert"
        self.turn = "Player 1 (Human)"
        self.last_cpu_move_info = {}
        self.message = ""

def test_solver(name, SolverClass, state):
    print(f"\n--- Testing {name} ---")
    start = time.time()
    solver = SolverClass(state)
    
    # Run the solve action
    if hasattr(solver, "decide_move"):
        solver.decide_move()
    elif hasattr(solver, "solve"):
        solver.solve()
        
    duration = time.time() - start
    print(f"{name} took {duration:.4f} seconds.")

if __name__ == "__main__":
    # Create a 4x4 mock board (standard grid)
    rows, cols = 4, 4
    clues = {
        (0, 0): 3, (0, 1): 2, (0, 3): 1,
        (1, 1): 3, (1, 2): 2,
        (2, 0): 2, (2, 3): 3,
        (3, 1): 1, (3, 2): 2
    }
    
    state = MockUIState(rows, cols, clues)
    
    test_solver("Divide & Conquer", DivideConquerSolver, state)
    test_solver("Dynamic Programming", DynamicProgrammingSolver, state)
    test_solver("Advanced DP", AdvancedDPSolver, state)
