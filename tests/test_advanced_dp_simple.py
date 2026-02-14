
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.solvers.advanced_dp_solver import AdvancedDPSolver

def test_simple_case():
    print("Initializing Simple 4x4 Test...")
    game = GameState(rows=4, cols=4, difficulty="Easy", game_mode="vs_cpu", solver_strategy="advanced_dp")
    
    # Overwrite clues to force a simple 2x2 box loop in the center
    # Box: (1,1)-(1,2)-(2,2)-(2,1)-(1,1)
    # The cells are (1,1), (1,2), (2,1), (2,2) ? 
    # Vertices (1,1) to (1,2) is Top edge of Cell (1,1).
    # Vertices (1,2) to (2,2) is Right edge of Cell (1,1) and Left of (1,2)? No.
    # Vertices:
    # (1,1) -- (1,2) -- (1,3)
    #   |        |        |
    # (2,1) -- (2,2) -- (2,3)
    
    # Let's define the loop around the single CENTERAL INTERSECTION?
    # No, let's define a loop around the central 4 cells (2x2 cells).
    # Cells: (1,1), (1,2), (2,1), (2,2).
    # Loop Edges:
    # Top: (1,1)-(1,2), (1,2)-(1,3) -> Row 1.
    # Bottom: (3,1)-(3,2), (3,2)-(3,3) -> Row 3.
    # Left: (1,1)-(2,1), (2,1)-(3,1) -> Col 1.
    # Right: (1,3)-(2,3), (2,3)-(3,3) -> Col 3.
    
    # Clues for this:
    # (1,1): 2 (Top, Left)
    # (1,2): 2 (Top, Right)
    # (2,1): 2 (Bottom, Left)
    # (2,2): 2 (Bottom, Right)
    
    # Wait. This describes a 2x2 block of cells.
    # Cells (1,1), (1,2), (2,1), (2,2).
    # We want a loop encapsulating them.
    # Edges: around the outer boundary of this block.
    # Inner edges (Between 1,1 and 1,2 etc) are OFF.
    # Clues:
    # (1,1): Top and Left ON. Bottom/Right OFF. -> 2
    # (1,2): Top and Right ON. Bottom/Left OFF. -> 2
    # (2,1): Left and Bottom ON. Top/Right OFF. -> 2
    # (2,2): Right and Bottom ON. Top/Left OFF. -> 2
    
    # Valid Loop: Spanning (1,1)-(1,3)-(3,3)-(3,1)
    # Cells (1,1), (1,2), (2,1), (2,2) form the distinct corners?
    # No, Cells (1,1), (1,2), (2,1), (2,2) have 2 segments each.
    # Center edges are OFF. Outer edges are ON.
    clues = {
        # Inner 2x2 block
        (1,1): 2, (1,2): 2,
        (2,1): 2, (2,2): 2,
        
        # Outer Ring
        (0,1): 1, (0,2): 1,
        (1,0): 1, (1,3): 1,
        (2,0): 1, (2,3): 1,
        (3,1): 1, (3,2): 1,
        
        # Corners (Empty)
        (0,0): 0, (0,3): 0,
        (3,0): 0, (3,3): 0
    }
    
    game.clues = clues
    print(f"Set Clues: {clues}")
    
    print("Solving...")
    game.cpu._compute_full_solution()
    
    sol = game.cpu._final_solution_edges
    if not sol:
        print("FAIL: No solution found for simple case.")
    else:
        print(f"PASS: Found solution with {len(sol)} edges.")
        print(sol)

if __name__ == "__main__":
    with open("test_simple.log", "w", encoding="utf-8") as f:
        sys.stdout = f
        test_simple_case()
