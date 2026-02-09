"""
UI Pages
========
Content pages for the main window.
Includes Game Over handling.
"""

import tkinter as tk
from tkinter import messagebox
from ui.styles import *
from ui.components import HoverButton, CardFrame
from ui.board_canvas import BoardCanvas
from logic.game_state import GameState
from ui.audio import play_sound

class HomePage(tk.Frame):
    def __init__(self, master, on_start_game, on_show_help):
        super().__init__(master, bg=BG_COLOR)
        self.on_start_game = on_start_game
        self.on_show_help = on_show_help
        self.selected_mode = None  # No default mode initially
        
        # Hero Section
        tk.Label(self, text="Loopy.", font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(80, 10))
        tk.Label(self, text="The Slitherlink Duel.", font=FONT_HEADER, bg=BG_COLOR, fg=TEXT_DIM).pack(pady=(0, 40))
        
        # Game Mode Selection
        mode_card = CardFrame(self, padx=40, pady=30)
        mode_card.pack(pady=10)
        
        tk.Label(mode_card, text="Step 1: Select Game Mode", font=FONT_BODY, bg=CARD_BG, fg=TEXT_DIM).pack(pady=(0, 15))
        
        mode_frame = tk.Frame(mode_card, bg=CARD_BG)
        mode_frame.pack()
        
        self.btn_vs_cpu = HoverButton(mode_frame, text="Normal (vs CPU)", command=lambda: self.set_mode("vs_cpu"), width=20, fg=APPLE_PURPLE)
        self.btn_vs_cpu.pack(side=tk.LEFT, padx=5)
        
        self.btn_2p = HoverButton(mode_frame, text="Normal (PvP)", command=lambda: self.set_mode("two_player"), width=20, fg=APPLE_TEAL)
        self.btn_2p.pack(side=tk.LEFT, padx=5)
        
        self.btn_greedy = HoverButton(mode_frame, text="Expert Level", command=lambda: self.set_mode("expert"), width=20, fg=APPLE_ORANGE)
        self.btn_greedy.pack(side=tk.LEFT, padx=5)
        
        # Help Button
        HoverButton(self, text="How to Play", command=self.on_show_help, width=15, fg=TEXT_COLOR, bg=BG_COLOR).pack(pady=(20, 0))
        
        # Difficulty Selection (Initially Hidden)
        self.diff_card = CardFrame(self, padx=40, pady=30)
        # Don't pack immediately
        
        tk.Label(self.diff_card, text="Step 2: Select Difficulty", font=FONT_BODY, bg=CARD_BG, fg=TEXT_DIM).pack(pady=(0, 15))
        
        HoverButton(self.diff_card, text="Easy (4x4)", command=lambda: self.start_with_mode(4, 4, "Easy"), width=24, fg=APPLE_GREEN).pack(pady=5)
        HoverButton(self.diff_card, text="Medium (5x5)", command=lambda: self.start_with_mode(5, 5, "Medium"), width=24, fg=APPLE_BLUE).pack(pady=5)
        HoverButton(self.diff_card, text="Hard (7x7)", command=lambda: self.start_with_mode(7, 7, "Hard"), width=24, fg=APPLE_RED).pack(pady=5)
    
    def set_mode(self, mode):
        self.selected_mode = mode
        
        # Highlight selected button
        self.btn_vs_cpu.config(bg=SIDEBAR_COLOR, fg=APPLE_PURPLE)
        self.btn_2p.config(bg=SIDEBAR_COLOR, fg=APPLE_TEAL)
        self.btn_greedy.config(bg=SIDEBAR_COLOR, fg=APPLE_ORANGE)
        
        if mode == "vs_cpu":
            self.btn_vs_cpu.config(bg=ACCENT_COLOR, fg="white")
        elif mode == "two_player":
            self.btn_2p.config(bg=ACCENT_COLOR, fg="white")
        elif mode == "expert":
            self.btn_greedy.config(bg=ACCENT_COLOR, fg="white")
            
        # Show difficulty card if not already shown
        if not self.diff_card.winfo_ismapped():
            self.diff_card.pack(pady=10)
    
    def start_with_mode(self, rows, cols, difficulty):
        self.on_start_game(rows, cols, difficulty, self.selected_mode)

class HelpPage(tk.Frame):
    def __init__(self, master, on_back):
        super().__init__(master, bg=BG_COLOR)
        self.on_back = on_back
        
        # Header
        header = tk.Frame(self, bg=BG_COLOR)
        header.pack(fill=tk.X, pady=20, padx=40)
        
        tk.Label(header, text="How to Play", font=FONT_HEADER, bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        HoverButton(header, text="Back", command=on_back, width=10, fg=APPLE_RED).pack(side=tk.RIGHT)
        
        # Content Scrollable? For simplicity, just a frame for now.
        content = tk.Frame(self, bg=BG_COLOR)
        content.pack(expand=True, fill=tk.BOTH, padx=40)
        
        # Section 1: Basics
        frame_basics = CardFrame(content, padx=20, pady=20)
        frame_basics.pack(fill=tk.X, pady=10)
        
        tk.Label(frame_basics, text="The Rules", font=FONT_BODY, bg=CARD_BG, fg=ACCENT_COLOR).pack(anchor="w")
        
        try:
            self.img_basics = tk.PhotoImage(file="assets/help_basics.png").subsample(3, 3)
            tk.Label(frame_basics, image=self.img_basics, bg=CARD_BG).pack(side=tk.LEFT, padx=10)
        except Exception as e:
            print(f"Error loading image: {e}")
            
        text_basics = """
        1. Connect adjacent dots with vertical or horizontal lines.
        2. The goal is to form a SINGLE continuous loop.
        3. Numbers indicate how many lines surround that cell.
        4. Empty cells can have any number of lines (0-3).
        5. Lines cannot cross or branch.
        """
        tk.Label(frame_basics, text=text_basics, font=FONT_SMALL, bg=CARD_BG, fg=TEXT_DIM, justify=tk.LEFT).pack(side=tk.LEFT, padx=10)

        # Section 2: Expert Mode
        frame_greedy = CardFrame(content, padx=20, pady=20)
        frame_greedy.pack(fill=tk.X, pady=10)
        
        tk.Label(frame_greedy, text="Expert Level (Greedy)", font=FONT_BODY, bg=CARD_BG, fg=APPLE_ORANGE).pack(anchor="w")
        
        try:
            self.img_greedy = tk.PhotoImage(file="assets/help_greedy.png").subsample(3, 3)
            tk.Label(frame_greedy, image=self.img_greedy, bg=CARD_BG).pack(side=tk.LEFT, padx=10)
        except:
            pass
            
        text_greedy = """
        1. You have limited ENERGY (Knapsack Capacity).
        2. Each edge has a WEIGHT (Cost) shown next to it.
        3. Adding an edge consumes Energy. Removing it refunds it.
        4. Solve the puzzle without running out of Energy!
        5. Tip: Prioritize low-weight edges (Green) like Kruskal's Algorithm.
        """
        tk.Label(frame_greedy, text=text_greedy, font=FONT_SMALL, bg=CARD_BG, fg=TEXT_DIM, justify=tk.LEFT).pack(side=tk.LEFT, padx=10)

class GamePage(tk.Frame):
    def __init__(self, master, game_state, on_back):
        super().__init__(master, bg=BG_COLOR)
        self.game_state = game_state
        self.on_back = on_back
        
        # Top Bar (Info)
        self.info_bar = tk.Frame(self, bg=BG_COLOR)
        self.info_bar.pack(fill=tk.X, pady=20, padx=40)
        
        self.lbl_turn = tk.Label(self.info_bar, text="", font=FONT_HEADER, bg=BG_COLOR, fg=ACCENT_COLOR)
        self.lbl_turn.pack(side=tk.LEFT)
        
        self.lbl_status = tk.Label(self.info_bar, text="", font=FONT_BODY, bg=BG_COLOR, fg=TEXT_DIM)
        self.lbl_status.pack(side=tk.RIGHT)
        
        # Energy Bar (Expert Mode)
        if self.game_state.game_mode == "expert":
            self.energy_frame = tk.Frame(self, bg=BG_COLOR)
            self.energy_frame.pack(fill=tk.X, padx=40, pady=(0, 10))
            
            tk.Label(self.energy_frame, text="Energy:", font=FONT_BODY, bg=BG_COLOR, fg=TEXT_DIM).pack(side=tk.LEFT)
            self.lbl_energy = tk.Label(self.energy_frame, text="", font=FONT_HEADER, bg=BG_COLOR, fg=APPLE_ORANGE)
            self.lbl_energy.pack(side=tk.LEFT, padx=10)
        
        # Board Area
        self.board_frame = CardFrame(self, padx=20, pady=20)
        self.board_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=(0, 20))
        
        self.canvas = BoardCanvas(self.board_frame, game_state, self.on_move)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Controls
        controls = tk.Frame(self, bg=BG_COLOR)
        controls.pack(fill=tk.X, pady=20, padx=40)
        
        HoverButton(controls, text="Undo", command=self.undo, fg=WARNING_COLOR).pack(side=tk.LEFT, padx=5)
        HoverButton(controls, text="Redo", command=self.redo, fg=WARNING_COLOR).pack(side=tk.LEFT, padx=5)
        HoverButton(controls, text="Hint", command=self.hint, fg=SUCCESS_COLOR).pack(side=tk.LEFT, padx=5)
        HoverButton(controls, text="End Game", command=on_back, fg=ERROR_COLOR).pack(side=tk.RIGHT, padx=5)
        
        self.update_ui()

    def on_move(self, u, v):
        success = self.game_state.make_move(u, v)
        # Always update UI to show messages (e.g. Invalid move, Low energy)
        self.update_ui()
        
        if success:
            play_sound("move")
        else:
            play_sound("error")
            
        self.check_game_over()
        
        # Only trigger CPU move in vs_cpu mode OR expert mode
        if (not self.game_state.game_over and 
            self.game_state.game_mode in ["vs_cpu", "expert"] and 
            self.game_state.turn == "Player 2 (CPU)"):
            
            # Start CPU thinking process (Visualization)
            self.after(500, self.cpu_move)

    def cpu_move(self):
        # 1. Decide
        candidates, best_move = self.game_state.cpu.decide_move()
        
        if best_move:
            # 2. Execute directly (Hidden thinking)
            self.finalize_cpu_move(best_move)
        else:
            # Stalemate for CPU?
            self.check_game_over()

    def finalize_cpu_move(self, move):
        u, v = move
        self.game_state.make_move(u, v, is_cpu=True)
        play_sound("move")
        self.update_ui()
        self.check_game_over()

    def check_game_over(self):
        if self.game_state.game_over:
            winner = self.game_state.winner
            if winner == "Stalemate":
                msg = "Game Over!\nStalemate: No valid moves left."
            else:
                msg = f"Game Over!\nWinner: {winner}"
                # Victory Animation
                if winner == "Player 1 (Human)":
                     self.canvas.play_victory_animation()
                     play_sound("win")
                     
            messagebox.showinfo("Game Over", msg)
            # Auto-exit to homepage after 3 seconds
            self.after(3000, self.on_back)

    def undo(self):
        if self.game_state.undo():
            self.update_ui()

    def redo(self):
        if self.game_state.redo():
            self.update_ui()

    def hint(self):
        move, reason = self.game_state.get_hint()
        if move:
            color = APPLE_GREEN
            if "Remove" in reason:
                 color = APPLE_RED
                 
            self.canvas.show_hint(move, color=color)
            
            self.game_state.message = f"Hint: {reason}"
            self.update_ui()
        else:
            self.game_state.message = "No hints available."
            self.update_ui()

    def update_ui(self):
        self.canvas.draw()
        self.lbl_turn.config(text=f"Turn: {self.game_state.turn}")
        self.lbl_status.config(text=self.game_state.message)
        
        if self.game_state.game_mode == "expert":
            self.lbl_energy.config(text=f"You: {self.game_state.energy} | CPU: {self.game_state.energy_cpu}")

class StatsPage(tk.Frame):
    def __init__(self, master, on_back):
        super().__init__(master, bg=BG_COLOR)
        
        tk.Label(self, text="Statistics", font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=40)
        
        card = CardFrame(self, padx=40, pady=40)
        card.pack(pady=20)
        
        # Load stats
        from logic.statistics import StatisticsManager
        stats = StatisticsManager().stats
        
        tk.Label(card, text=f"Games Played: {stats.get('games_played', 0)}", font=FONT_BODY, bg=CARD_BG, fg=TEXT_COLOR).pack(anchor="w", pady=5)
        tk.Label(card, text=f"Wins vs CPU: {stats.get('wins_vs_cpu', 0)}", font=FONT_BODY, bg=CARD_BG, fg=APPLE_GREEN).pack(anchor="w", pady=5)
        tk.Label(card, text=f"Losses: {stats.get('losses', 0)}", font=FONT_BODY, bg=CARD_BG, fg=APPLE_RED).pack(anchor="w", pady=5)
        
        HoverButton(self, text="Back", command=on_back).pack(pady=20)
