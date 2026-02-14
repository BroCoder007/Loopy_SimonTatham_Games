"""
Main Window
===========
The shell application window.
"""

import tkinter as tk
from ui.styles import *
from ui.pages import HomePage, GamePage, StatsPage
from logic.game_state import GameState

class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Loopy DAA - Slitherlink AI Duel")
        self.geometry("900x700")
        self.configure(bg=BG_COLOR)
        
        # Layout
        self.sidebar = tk.Frame(self, bg=SIDEBAR_COLOR, width=SIDEBAR_WIDTH)
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)
        self.sidebar.pack_propagate(False)
        
        self.content_area = tk.Frame(self, bg=BG_COLOR)
        self.content_area.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)
        
        self.setup_sidebar()
        self.show_home()
        
    def setup_sidebar(self):
        # App Title in Sidebar
        tk.Label(self.sidebar, text="LOOPY\nDAA", font=("Segoe UI", 20, "bold"), bg=SIDEBAR_COLOR, fg="white").pack(pady=30)
        
        # Nav Buttons
        self.create_nav_btn("Home", self.show_home)
        self.create_nav_btn("Statistics", self.show_stats)
        self.create_nav_btn("Exit", self.quit)
        
    def create_nav_btn(self, text, command):
        btn = tk.Button(self.sidebar, text=text, command=command,
                        bg=SIDEBAR_COLOR, fg=TEXT_DIM, font=FONT_BODY,
                        bd=0, activebackground=ACCENT_COLOR, activeforeground="white",
                        cursor="hand2", anchor="w", padx=20)
        btn.pack(fill=tk.X, pady=2)
        
        # Simple hover
        btn.bind("<Enter>", lambda e: btn.config(bg=ACCENT_COLOR))
        btn.bind("<Leave>", lambda e: btn.config(bg=SIDEBAR_COLOR))

    def clear_content(self):
        for widget in self.content_area.winfo_children():
            widget.destroy()

    def show_home(self):
        self.clear_content()
        HomePage(self.content_area, self.start_game, self.show_help).pack(expand=True, fill=tk.BOTH)

    def show_stats(self):
        self.clear_content()
        StatsPage(self.content_area, self.show_home).pack(expand=True, fill=tk.BOTH)

    def start_game(self, rows, cols, difficulty, game_mode, solver_strategy=None, generator_type="prim"):
        self.clear_content()
        # UI wiring only: store/forward selected strategy.
        # For now, all strategies still use Greedy internally.
        game_state = GameState(rows, cols, difficulty, game_mode, solver_strategy=solver_strategy, generator_type=generator_type)
        GamePage(self.content_area, game_state, self.show_home).pack(expand=True, fill=tk.BOTH)

    def show_help(self):
        self.clear_content()
        from ui.pages import HelpPage
        HelpPage(self.content_area, self.show_home).pack(expand=True, fill=tk.BOTH)
