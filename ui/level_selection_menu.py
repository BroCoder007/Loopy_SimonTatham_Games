import tkinter as tk
from ui.styles import *
from ui.components import CardFrame, HoverButton

class LevelSelectionMenu(tk.Frame):
    """
    Phase 3: Level Selection Menu UI
    Allows the user to select the DP Generator and their difficulty level.
    """
    def __init__(self, master, on_start):
        super().__init__(master, bg=CARD_BG)
        self.on_start = on_start
        
        tk.Label(self, text="Step 2: Setup Generator & Difficulty", font=FONT_BODY, bg=CARD_BG, fg=TEXT_DIM).pack(pady=(0, 15))
        
        # Generator Strategy Selection
        self.generator_var = tk.StringVar(value="dp")
        
        gen_frame = tk.Frame(self, bg=CARD_BG)
        gen_frame.pack(pady=(0, 15))
        
        # Emphasize the new Phase 3 DP Generator
        tk.Radiobutton(gen_frame, text="✅ DP/Backtracking Generator (Phase 3)", variable=self.generator_var, value="dp", 
                       bg=CARD_BG, fg=APPLE_GREEN, selectcolor=CARD_BG, activebackground=CARD_BG, font=FONT_SMALL).pack(anchor="w", pady=2)
                       
        tk.Radiobutton(gen_frame, text="Divide & Conquer Generator (Phase 2)", variable=self.generator_var, value="dnc", 
                       bg=CARD_BG, fg=TEXT_DIM, selectcolor=CARD_BG, activebackground=CARD_BG, font=FONT_SMALL).pack(anchor="w", pady=2)
                       
        tk.Radiobutton(gen_frame, text="Prim's MST Generator (Phase 1)", variable=self.generator_var, value="prim", 
                       bg=CARD_BG, fg=TEXT_DIM, selectcolor=CARD_BG, activebackground=CARD_BG, font=FONT_SMALL).pack(anchor="w", pady=2)
        
        # Difficulty Buttons (These trigger the on_start callback)
        HoverButton(self, text="Easy (4x4)", command=lambda: self._select(4, 4, "Easy"), width=24, fg=APPLE_GREEN).pack(pady=5)
        HoverButton(self, text="Medium (5x5)", command=lambda: self._select(5, 5, "Medium"), width=24, fg=APPLE_BLUE).pack(pady=5)
        HoverButton(self, text="Hard (7x7)", command=lambda: self._select(7, 7, "Hard"), width=24, fg=APPLE_RED).pack(pady=5)
        HoverButton(self, text="Extreme (10x10)", command=lambda: self._select(10, 10, "Extreme"), width=24, fg=APPLE_ORANGE).pack(pady=5)

    def _select(self, rows, cols, difficulty):
        generator_type = self.generator_var.get()
        self.on_start(rows, cols, difficulty, generator_type)
