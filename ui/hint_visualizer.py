from ui.styles import APPLE_RED, APPLE_GREEN

class HintVisualizer:
    """
    Phase 3: Visual Feedback Engine for the Hint System.
    Connects the DP & Backtracking logic calculations to Tkinter canvas effects.
    """
    def __init__(self, page, canvas):
        self.page = page
        self.canvas = canvas

    def render_hint(self, hint_data):
        is_error = hint_data.get("is_error", False)
        move = hint_data.get("move")
        path = hint_data.get("highlight_path", [])
        
        if is_error:
            self._flash_error()
        elif path:
            self._draw_dp_path(path)
            
        if move and not is_error:
            self.canvas.show_hint(move, color=APPLE_GREEN)
            
        # Update UI text readout
        reason = hint_data.get("strategy", "Hint")
        explanation = hint_data.get("explanation", "")
        
        self.page.game_state.message = f"Hint: {reason}"
        self.page._last_hint_explanation = explanation
        self.page._render_hint_explanation()
        self.page.update_ui()

    def _flash_error(self):
        """Flashes the board red to highlight backtrack validation logic failures."""
        original_bg = self.canvas["bg"]
        self.canvas.config(bg=APPLE_RED)
        self.canvas.after(150, lambda: self.canvas.config(bg=original_bg))
        self.canvas.after(300, lambda: self.canvas.config(bg=APPLE_RED))
        self.canvas.after(450, lambda: self.canvas.config(bg=original_bg))

    def _draw_dp_path(self, path_edges):
        """Highlights the entire optimal DP path in a soft green temporarily."""
        # Clears up on next user action automatically via the canvas draw routine
        for edge in path_edges:
            self.canvas.show_hint(edge, color="#A8E6CF") 
