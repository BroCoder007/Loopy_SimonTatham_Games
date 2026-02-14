"""
UI Pages
========
Content pages for the main window.
Includes Game Over handling.
"""

import tkinter as tk
import re
from tkinter import messagebox
from typing import Any, Dict, List, Optional, Tuple
from ui.styles import *
from ui.components import HoverButton, CardFrame
from ui.board_canvas import BoardCanvas
from logic.game_state import GameState
from ui.audio import play_sound
from ui.strategy_store import strategy_store
from logic.live_analysis import LiveAnalysisService
from ui.analysis_panel import LiveAnalysisPanel

class HomePage(tk.Frame):
    def __init__(self, master, on_start_game, on_show_help):
        super().__init__(master, bg=BG_COLOR)
        self.on_start_game = on_start_game
        self.on_show_help = on_show_help
        self.selected_mode = None  # No default mode initially
        self.selected_strategy = None  # Selected after difficulty via modal
        
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
        
        # Generator Selection (New)
        self.var_use_dnc = tk.BooleanVar(value=False)
        chk_dnc = tk.Checkbutton(self.diff_card, text="Use D&C Generator", variable=self.var_use_dnc, 
                                 bg=CARD_BG, fg=TEXT_COLOR, selectcolor=CARD_BG, activebackground=CARD_BG)
        chk_dnc.pack(pady=(10, 5))

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
        # After difficulty selection, ask for solving strategy.
        # Default is Greedy if user closes the dialog.
        generator_type = "dnc" if self.var_use_dnc.get() else "prim"
        self._show_strategy_modal(
            on_selected=lambda strategy: self.on_start_game(rows, cols, difficulty, self.selected_mode, strategy, generator_type)
        )

    def _show_strategy_modal(self, on_selected):
        """
        Modal dialog shown after difficulty selection.
        Stores selection globally in `ui.strategy_store`.
        """
        dlg = tk.Toplevel(self)
        dlg.title("Select Solving Strategy")
        dlg.configure(bg=BG_COLOR)
        dlg.resizable(False, False)
        dlg.transient(self.winfo_toplevel())
        dlg.grab_set()

        # Center-ish relative to parent
        try:
            dlg.update_idletasks()
            x = self.winfo_rootx() + 120
            y = self.winfo_rooty() + 120
            dlg.geometry(f"+{x}+{y}")
        except Exception:
            pass

        card = CardFrame(dlg, padx=30, pady=25)
        card.pack(padx=20, pady=20)

        tk.Label(card, text="Select Solving Strategy", font=FONT_HEADER, bg=CARD_BG, fg=TEXT_COLOR).pack(pady=(0, 10))
        tk.Label(
            card,
            text="Choose one CPU solving strategy.",
            font=FONT_SMALL,
            bg=CARD_BG,
            fg=TEXT_DIM,
            justify=tk.LEFT,
        ).pack(pady=(0, 15))

        choice = tk.StringVar(value=strategy_store.get_strategy())

        options = [
            ("Greedy Solver", "greedy"),
            ("Divide & Conquer Solver", "divide_conquer"),
            ("Dynamic Programming Solver", "dynamic_programming"),
        ]

        for label, value in options:
            row = tk.Frame(card, bg=CARD_BG)
            row.pack(fill=tk.X, pady=4)
            tk.Radiobutton(
                row,
                text=label,
                variable=choice,
                value=value,
                bg=CARD_BG,
                fg=TEXT_COLOR,
                selectcolor=CARD_BG,
                activebackground=CARD_BG,
                activeforeground=TEXT_COLOR,
                font=FONT_BODY,
                anchor="w",
            ).pack(fill=tk.X, padx=6)

        btns = tk.Frame(card, bg=CARD_BG)
        btns.pack(fill=tk.X, pady=(15, 0))

        def commit_and_close():
            selected = choice.get() or "greedy"
            self.selected_strategy = selected
            strategy_store.set_strategy(selected)
            try:
                dlg.grab_release()
            except Exception:
                pass
            dlg.destroy()
            on_selected(selected)

        def cancel_and_close():
            # Keep current global (default Greedy) and proceed without crashing.
            selected = strategy_store.get_strategy() or "greedy"
            self.selected_strategy = selected
            try:
                dlg.grab_release()
            except Exception:
                pass
            dlg.destroy()
            on_selected(selected)

        HoverButton(btns, text="Back", command=cancel_and_close, width=10, fg=WARNING_COLOR).pack(side=tk.LEFT)
        HoverButton(btns, text="Start", command=commit_and_close, width=10, fg=SUCCESS_COLOR).pack(side=tk.RIGHT)

        dlg.protocol("WM_DELETE_WINDOW", cancel_and_close)
        dlg.wait_window()

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
        self._last_hint_explanation: str = ""
        
        # Sidebar for Analysis (Optional: could be separate window)
        # Using a PanedWindow might be better but let's stick to packing below board for now 
        # as requested in "Table UI: Add dynamic table panel"
        
        # Main Container to scroll if needed? 
        # For simplicity, we just pack it.
        
        # Top Bar (Info)
        self.info_bar = tk.Frame(self, bg=BG_COLOR)
        self.info_bar.pack(fill=tk.X, pady=10, padx=40)
        
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
        
        # Controls & Analysis Toggle
        controls = tk.Frame(self, bg=BG_COLOR)
        controls.pack(fill=tk.X, pady=10, padx=40)
        
        HoverButton(controls, text="Undo", command=self.undo, fg=WARNING_COLOR).pack(side=tk.LEFT, padx=5)
        HoverButton(controls, text="Redo", command=self.redo, fg=WARNING_COLOR).pack(side=tk.LEFT, padx=5)
        HoverButton(controls, text="Hint", command=self.hint, fg=SUCCESS_COLOR).pack(side=tk.LEFT, padx=5)
        HoverButton(controls, text="Explain CPU", command=self.explain_cpu_move, width=12, fg=APPLE_BLUE).pack(side=tk.LEFT, padx=5)
        
        self.var_live_analysis = tk.BooleanVar(value=False)
        chk_analysis = tk.Checkbutton(controls, text="Enable Live Analysis", variable=self.var_live_analysis,
                                      bg=BG_COLOR, fg=TEXT_COLOR, selectcolor=BG_COLOR, activebackground=BG_COLOR,
                                      command=self.toggle_analysis_panel)
        chk_analysis.pack(side=tk.RIGHT, padx=5)
        
        HoverButton(controls, text="End Game", command=self.end_game, fg=ERROR_COLOR).pack(side=tk.RIGHT, padx=5)

        # Board Area
        self.board_frame = CardFrame(self, padx=20, pady=20)
        self.board_frame.pack(expand=True, fill=tk.BOTH, padx=40, pady=(0, 20))
        
        self.canvas = BoardCanvas(self.board_frame, game_state, self.on_move)
        self.canvas.pack(expand=True, fill=tk.BOTH)
        
        # Hint explanation panel
        self.hint_explain_card = CardFrame(self, padx=20, pady=15)
        self.hint_explain_card.pack(fill=tk.X, padx=40, pady=(0, 10))

        tk.Label(self.hint_explain_card, text="Why this hint?", font=FONT_BODY, bg=CARD_BG, fg=TEXT_COLOR, anchor="w").pack(fill=tk.X)
        self.lbl_hint_explain = tk.Label(self.hint_explain_card, text="", font=FONT_SMALL, bg=CARD_BG, fg=TEXT_DIM, justify=tk.LEFT, wraplength=760, anchor="w")
        self.lbl_hint_explain.pack(fill=tk.X, pady=(8, 0))
        
        # Live Analysis Panel (Initially Hidden)
        self.analysis_panel = LiveAnalysisPanel(self, self.game_state)
        # Do not pack initially
        
        self.update_ui()

    def toggle_analysis_panel(self):
        if self.var_live_analysis.get():
            self.analysis_panel.pack(fill=tk.BOTH, expand=True, padx=40, pady=(0, 20))
            self.analysis_panel.update_data()
        else:
            self.analysis_panel.pack_forget()

    def explain_cpu_move(self):
        """
        Show a popup explaining the last move made by the CPU.
        """
        move_info = getattr(self.game_state, "last_cpu_move_info", None)
        if not move_info:
             messagebox.showinfo("CPU Explanation", "No CPU move has been made yet.")
             return
             
        move = move_info.get("move")
        explanation = move_info.get("explanation", "No explanation available.")
        strategy = move_info.get("strategy", "Unknown")
        
        msg = f"Last Move: {move}\nStrategy: {strategy}\n\nReasoning:\n{explanation}"
        messagebox.showinfo("CPU Explanation", msg)

    def on_move(self, u, v):
        success = self.game_state.make_move(u, v)
        self.update_ui()
        
        if success:
            play_sound("move")
            if hasattr(self.game_state, "cpu") and hasattr(self.game_state.cpu, "register_move"):
                move = tuple(sorted((u, v)))
                self.game_state.cpu.register_move(move)
        else:
            play_sound("error")
            
        self.check_game_over()
        
        if (not self.game_state.game_over and 
            self.game_state.game_mode in ["vs_cpu", "expert"] and 
            self.game_state.turn == "Player 2 (CPU)"):
            
            # Start CPU thinking process
            self.after(500, self.cpu_move)

    def cpu_move(self):
        # 0. Live Analysis Hook
        if self.var_live_analysis.get():
             self.game_state.message = "Running Comparative Analysis..."
             self.update_ui()
             self.update_idletasks() # Force UI update before freezing for threads
             
             service = LiveAnalysisService(self.game_state)
             service.run_analysis()
             self.analysis_panel.update_data()
             self.game_state.message = "Analysis Complete. CPU deciding..."
             self.update_ui()

        # 1. Decide through hierarchical strategy controller.
        best_move, strategy_source, solver_used, fallback_message = self.game_state.get_next_cpu_move()

        if best_move:
            # 2. Execute directly (Hidden thinking)
            success = self.finalize_cpu_move(
                best_move,
                strategy_override=strategy_source,
                solver_used=solver_used,
                fallback_message=fallback_message,
            )
            if not success:
                # Move failed, try again with next move
                self.after(500, self.cpu_move)

        else:
            # No moves available, switch turn back to human
            if not self.game_state.game_over:
                if strategy_source == "No moves available":
                    self.game_state.message = "No CPU move available for this board state."
                elif strategy_source == "DP (No deterministic move)":
                    self.game_state.message = "DP has no deterministic move on this board state."
                self.game_state.switch_turn()
                self.update_ui()
            self.check_game_over()

    def finalize_cpu_move(self, move, register_solver_move=True, strategy_override=None, solver_used=None, fallback_message=None):
        u, v = move
        success = self.game_state.make_move(u, v, is_cpu=True)
        
        if not success:
            # Move failed, return False so CPU can try another move
            return False

        active_solver = solver_used if solver_used is not None else self.game_state.cpu

        # Ensure move is registered for solver-side state tracking.
        if register_solver_move and hasattr(active_solver, "register_move"):
            active_solver.register_move(move)

        # Store CPU move information for explanation.
        explanation = "No explanation available."
        if hasattr(active_solver, "explain_last_move"):
            explanation = active_solver.explain_last_move()

        strategy_label = strategy_override or self._get_strategy_display_name()
        self.game_state.last_cpu_move_info = {
            "move": move,
            "explanation": explanation,
            "strategy": strategy_label
        }
            
        play_sound("move")

        if fallback_message and not self.game_state.game_over:
            self.game_state.message = fallback_message

        self.update_ui()
        self.check_game_over()
        return True

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

    def end_game(self):
        self.on_back()

    def undo(self):
        if self.game_state.undo():
            self.update_ui()

    def redo(self):
        if self.game_state.redo():
            self.update_ui()

    def _get_strategy_display_name(self):
        strategy = getattr(self.game_state, "solver_strategy", "greedy")
        if strategy == "divide_conquer":
            return "Divide & Conquer"
        if strategy == "dynamic_programming":
            return "Dynamic Programming"
        return "Greedy Solver"

    def hint(self):
        # Prefer solver output format when available, but remain backward-compatible.
        move = None
        reason = ""
        explanation = ""

        hint_obj = None
        try:
            if getattr(self.game_state, "cpu", None) is not None and hasattr(self.game_state.cpu, "generate_hint"):
                hint_obj = self.game_state.cpu.generate_hint(self.game_state)
        except Exception:
            hint_obj = None

        if isinstance(hint_obj, dict):
            move = hint_obj.get("move")
            # Keep old UI behavior: message continues to show the old "reason"-style text.
            # If solver didn't provide it, fall back to something safe.
            explanation = hint_obj.get("explanation") or ""
            strategy = hint_obj.get("strategy") or ""
            if strategy:
                # Provide a stable short reason for the status bar.
                reason = f"{strategy} hint"
            else:
                reason = "Hint"
        else:
            # Legacy (move, reason) tuple format from GameState.get_hint()
            move, reason = self.game_state.get_hint()
            explanation = ""

        if move:
            color = APPLE_GREEN
            if isinstance(reason, str) and "Remove" in reason:
                 color = APPLE_RED
                 
            self.canvas.show_hint(move, color=color)
            
            self.game_state.message = f"Hint: {reason}"
            self._last_hint_explanation = explanation if isinstance(explanation, str) else ""
            self._render_hint_explanation()
            self.update_ui()
        else:
            self.game_state.message = "No hints available."
            self._last_hint_explanation = explanation if isinstance(explanation, str) else ""
            self._render_hint_explanation()
            self.update_ui()

    def _render_hint_explanation(self):
        """
        Render the hint explanation panel safely.
        If explanation is missing, show a friendly backup message (must never crash gameplay).
        """
        try:
            text = (self._last_hint_explanation or "").strip()
            if not text:
                text = "No explanation available for this hint."
            self.lbl_hint_explain.config(text=text)
        except Exception:
            # Absolutely do not let UI rendering break gameplay.
            pass

    def update_ui(self):
        self.canvas.draw()
        self.lbl_turn.config(text=f"Turn: {self.game_state.turn}")
        self.lbl_status.config(text=self.game_state.message)
        
        if self.game_state.game_mode == "expert":
            self.lbl_energy.config(text=f"You: {self.game_state.energy} | CPU: {self.game_state.energy_cpu}")

    def destroy(self):
        super().destroy()

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
