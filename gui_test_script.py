import unittest
import tkinter as tk
import sys
import os

# Add project root to path
sys.path.append(os.getcwd())

from ui.main_window import MainWindow
from ui.pages import HomePage, GamePage

class TestGUI(unittest.TestCase):
    def setUp(self):
        self.app = MainWindow()
        # Do not call mainloop()
        
    def tearDown(self):
        self.app.destroy()
        
    def test_full_flow(self):
        print("Starting GUI Flow Test...", flush=True)
        
        # 1. Verify Home Page
        children = self.app.content_area.winfo_children()
        self.assertTrue(len(children) > 0, "No content in main window")
        home_page = children[0]
        self.assertIsInstance(home_page, HomePage, "Initial page is not HomePage")
        print("[PASS] application launched to HomePage", flush=True)
        
        # 2. Select Expert Mode
        # The buttons are attributes of HomePage
        # expert button is btn_greedy
        self.assertTrue(hasattr(home_page, 'btn_greedy'), "Expert button not found")
        home_page.btn_greedy.invoke()
        print("[PASS] Clicked Expert Mode button", flush=True)
        
        # Verify mode selection (internal state of HomePage)
        self.assertEqual(home_page.selected_mode, "expert", "Mode not set to expert")
        
        # 3. Select Difficulty (Medium) to Start Game
        # The difficulty buttons are packed in diff_card but not assigned to self attributes in a handy way
        # But we can find them via children of diff_card
        diff_card = home_page.diff_card
        # We need to find the button with text "Medium (5x5)"
        med_btn = None
        for child in diff_card.winfo_children():
            if isinstance(child, tk.Button):
                if "Medium" in child.cget("text"):
                    med_btn = child
                    break
        
        self.assertIsNotNone(med_btn, "Medium difficulty button not found")
        med_btn.invoke()
        print("[PASS] Clicked Medium Difficulty button", flush=True)
        
        # 4. Verify Game Page Transition
        # content_area should have been cleared and repopulated
        self.app.update() # Process events
        children = self.app.content_area.winfo_children()
        self.assertTrue(len(children) > 0)
        game_page = children[0]
        self.assertIsInstance(game_page, GamePage, "Did not transition to GamePage")
        print("[PASS] Transitioned to GamePage", flush=True)
        
        # 5. Verify Game State
        self.assertEqual(game_page.game_state.game_mode, "expert")
        self.assertEqual(game_page.game_state.rows, 5)
        print("[PASS] Game State initialized correctly", flush=True)
        
        # 6. Simulate Human Move
        # GamePage has canvas. We can call on_move directly or try to simulate click?
        # Simulating click on canvas is hard (coordinates). calling on_move is safer for logic test.
        # Find a valid move first
        moves = game_page.game_state.get_all_valid_moves()
        u, v = moves[0]
        
        print(f"Simulating move at {u}-{v}", flush=True)
        game_page.on_move(u, v)
        
        # 7. Verify UI Update
        # Turn label should change
        # Expert mode: Turn switches to CPU
        self.assertIn("CPU", game_page.lbl_turn.cget("text"))
        print("[PASS] UI updated turn to CPU", flush=True)
        
        # 8. Test CPU Thinking (Mocking after?)
        # CPU move is scheduled with self.after(500, ...)
        # We can force it by calling the callback directly if we want, or verify it's registered
        # But for this test, we proved the UI logic connected the click to the GameState update.
        
        print("GUI Logic Test Completed Successfully", flush=True)

if __name__ == '__main__':
    unittest.main()
