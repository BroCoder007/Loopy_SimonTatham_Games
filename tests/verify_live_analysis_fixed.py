
import sys
import os

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from logic.game_state import GameState
from logic.live_analysis import LiveAnalysisService
from logic.solvers.greedy_solver import GreedySolver

def test_live_analysis_greedy_integration():
    """
    Verify that Greedy solver runs correctly within the LiveAnalysisService.
    """
    print("Initializing GameState...")
    # Create a simple 3x3 board
    game_state = GameState(rows=3, cols=3)
    
    # Set a simple configuration where Greedy should find a move
    # e.g., a "0" clue in top-left (0,0) -> edges around it must be OFF
    game_state.clues[(0, 0)] = 0
    
    print("Initializing LiveAnalysisService...")
    service = LiveAnalysisService(game_state)
    
    print("Running analysis...")
    results = service.run_analysis()
    
    print("Analysis Results:", results)
    
    # Assertions
    assert results is not None, "Results should not be None"
    assert results["greedy_move"] != "N/A", "Greedy move should not be N/A"
    assert results["greedy_move"] != "Error", "Greedy solver returned an Error"
    
    # D&C and DP might timeout or return N/A on empty/complex boards without proper setup, 
    # but for a small 3x3 with one clue, they should ideally return something or at least not crash.
    # We mainly care about Greedy for this specific task.
    
    if results["greedy_move"] != "None":
        print("SUCCESS: Greedy solver found a move or successfully determined no move possible (but ran).")
    else:
        print("SUCCESS: Greedy solver ran but found no move (which is a valid result).")

    print("Verification Passed!")

if __name__ == "__main__":
    test_live_analysis_greedy_integration()
