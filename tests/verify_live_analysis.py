import sys
import os
import tkinter as tk
import unittest

# Ensure project root is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from logic.game_state import GameState
from logic.live_analysis import LiveAnalysisService
from ui.analysis_panel import LiveAnalysisPanel

class TestLiveAnalysis(unittest.TestCase):
    def setUp(self):
        # Create a 4x4 easy game state
        self.game_state = GameState(rows=4, cols=4, difficulty="Easy", game_mode="vs_cpu")
        # Ensure some clues exist (it usually auto-generates in init)
        
    def test_gamestate_clone(self):
        print("\nTesting GameState cloning...")
        clone = self.game_state.clone_for_simulation()
        
        self.assertNotEqual(id(self.game_state), id(clone))
        self.assertNotEqual(id(self.game_state.graph), id(clone.graph))
        self.assertNotEqual(id(self.game_state.clues), id(clone.clues))
        
        # Verify essential data matches
        self.assertEqual(self.game_state.rows, clone.rows)
        self.assertEqual(len(self.game_state.graph.edges), len(clone.graph.edges))
        
        # Verify clean slate
        self.assertIsNone(clone.cpu)
        print("Clone verification passed.")

    def test_service_execution(self):
        print("\nTesting LiveAnalysisService execution (this may take a second)...")
        service = LiveAnalysisService(self.game_state)
        result = service.run_analysis()
        
        print("Result:", result)
        
        # Check structure
        self.assertIn("move_number", result)
        self.assertIn("greedy_move", result)
        self.assertIn("dp_move", result)
        self.assertIn("advanced_move", result)
        
        # Check integrity of GameState history
        self.assertEqual(len(self.game_state.live_analysis_table), 1)
        self.assertEqual(self.game_state.live_analysis_table[0], result)
        print("Service execution passed.")

    def test_ui_panel_instantiation(self):
        print("\nTesting UI Panel instantiation...")
        root = tk.Tk()
        # Hide it
        root.withdraw()
        
        try:
            panel = LiveAnalysisPanel(root, self.game_state)
            
            # Populate some fake data to test rendering
            fake_dat = {
                "move_number": 1,
                "greedy_move": "((0,0),(0,1))", "greedy_time": 10.5, "greedy_states": 50,
                "dp_move": "((0,0),(0,1))", "dp_time": 15.2, "dp_states": 100,
                "advanced_move": "((0,0),(0,1))", "advanced_time": 25.0, "advanced_states": 200
            }
            self.game_state.live_analysis_table.append(fake_dat)
            
            panel.update_data()
            print("Panel update_data() ran without error.")
            
        except Exception as e:
            self.fail(f"UI Panel crashed: {e}")
        finally:
            root.destroy()

if __name__ == "__main__":
    unittest.main()
