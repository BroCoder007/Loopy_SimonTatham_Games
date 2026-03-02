"""
Solver Control Panel
====================
Phase 3: DP & Backtracking Solver Control Panel
A dedicated control window to watch the algorithm run in real-time.

Now supports:
- Clean stop via stop_event
- Timeout display when solver hits limits
- Max states limit enforcement
"""

import tkinter as tk
import time
import threading
from ui.styles import *
from ui.components import HoverButton, CardFrame
from logic.solvers.dp_backtracking_solver import DPBacktrackingSolver


class SolverControlPanel(tk.Toplevel):
    """
    Phase 3: DP & Backtracking Solver Control Panel (Person 1 / Leader)
    A dedicated control window to watch the algorithm run in real-time.
    """
    def __init__(self, master, game_page):
        super().__init__(master, bg=BG_COLOR)
        self.game_page = game_page
        self.game_state = game_page.game_state
        self.canvas = game_page.canvas
        self._stop_event = threading.Event()
        self.solver = DPBacktrackingSolver(
            self.game_state,
            stop_event=self._stop_event,
            timeout=30.0,       # 30s for interactive auto-solve
            max_states=10_000_000,
        )

        self.title("DP & Backtracking Auto-Solver")
        self.geometry("400x580")
        self.resizable(False, False)
        self.transient(master)

        # Keep on top
        self.attributes("-topmost", True)

        self._setup_ui()
        self.is_running = False
        self._start_time = 0.0

    def _setup_ui(self):
        tk.Label(self, text="Phase 3 Auto-Solver", font=FONT_HEADER, bg=BG_COLOR, fg=TEXT_COLOR).pack(pady=(20, 5))
        tk.Label(self, text="Watch Backtracking combined with DP Memoization", font=FONT_SMALL, bg=BG_COLOR, fg=TEXT_DIM).pack(pady=(0, 20))

        # Config Panel
        config_card = CardFrame(self, padx=20, pady=20)
        config_card.pack(fill=tk.X, padx=20, pady=10)

        tk.Label(config_card, text="Animation Speed", font=FONT_BODY, bg=CARD_BG, fg=ACCENT_COLOR).pack(anchor="w")

        self.speed_var = tk.DoubleVar(value=0.01)
        speed_scale = tk.Scale(config_card, variable=self.speed_var, from_=0.0, to=0.5, resolution=0.01,
                               orient=tk.HORIZONTAL, bg=CARD_BG, fg=TEXT_COLOR, highlightthickness=0)
        speed_scale.pack(fill=tk.X, pady=(5, 10))

        tk.Label(config_card, text="Algorithm uses DP to prune repeated dead-end states.",
                 font=FONT_SMALL, bg=CARD_BG, fg=TEXT_DIM, justify=tk.LEFT, wraplength=300).pack(anchor="w")

        # Live Stats Panel
        self.stats_card = CardFrame(self, padx=20, pady=20)
        self.stats_card.pack(fill=tk.X, padx=20, pady=10)

        self.lbl_nodes = tk.Label(self.stats_card, text="Nodes Visited: 0", font=FONT_BODY, bg=CARD_BG, fg=TEXT_COLOR)
        self.lbl_nodes.pack(anchor="w")

        self.lbl_cache = tk.Label(self.stats_card, text="DP Cache Hits: 0", font=FONT_BODY, bg=CARD_BG, fg=APPLE_GREEN)
        self.lbl_cache.pack(anchor="w", pady=5)

        self.lbl_time = tk.Label(self.stats_card, text="Time Elapsed: 0.0s", font=FONT_BODY, bg=CARD_BG, fg=APPLE_PURPLE)
        self.lbl_time.pack(anchor="w")

        self.lbl_status = tk.Label(self.stats_card, text="Status: Ready", font=FONT_BODY, bg=CARD_BG, fg=TEXT_DIM)
        self.lbl_status.pack(anchor="w", pady=5)

        # Controls
        ctrl_frame = tk.Frame(self, bg=BG_COLOR)
        ctrl_frame.pack(fill=tk.X, padx=20, pady=20)

        self.btn_solve = HoverButton(ctrl_frame, text="Start DP Solve", command=self._start_solve_thread, width=15, fg=SUCCESS_COLOR)
        self.btn_solve.pack(side=tk.LEFT, padx=5)

        self.btn_stop = HoverButton(ctrl_frame, text="Stop", command=self._stop_solver, width=10, fg=WARNING_COLOR)
        self.btn_stop.pack(side=tk.LEFT, padx=5)
        self.btn_stop.config(state=tk.DISABLED)

        HoverButton(ctrl_frame, text="Close", command=self._on_close, width=15, fg=WARNING_COLOR).pack(side=tk.RIGHT, padx=5)

    def _ui_update_callback(self, current_edges):
        """Called by solver during backtracking to update UI."""
        if not self.winfo_exists(): return

        # 1. Update Canvas
        self.game_state.graph.edges = set(current_edges)
        self.canvas.draw()

        # 2. Update Live Stats
        try:
            nodes = getattr(self.solver, "nodes_visited", 0)
            cache = getattr(self.solver, "cache_hits", 0)
            elapsed = time.perf_counter() - self._start_time

            self.lbl_nodes.config(text=f"Nodes Visited: {nodes:,}")
            self.lbl_cache.config(text=f"DP Cache Hits: {cache:,}")
            self.lbl_time.config(text=f"Time Elapsed: {elapsed:.2f}s")
            self.lbl_status.config(text="Status: Solving...", fg=APPLE_ORANGE)
        except Exception:
            pass

    def _start_solve_thread(self):
        if self.is_running: return
        self.is_running = True
        self._stop_event.clear()
        self._start_time = time.perf_counter()
        self.btn_solve.config(state=tk.DISABLED, text="Solving...")
        self.btn_stop.config(state=tk.NORMAL)
        self.lbl_status.config(text="Status: Solving...", fg=APPLE_ORANGE)

        def run():
            delay = self.speed_var.get()
            result = self.solver.solve(ui_callback=self._ui_update_callback, delay=delay)

            # Revert to main thread updates for completion
            if self.winfo_exists():
                self.after(0, self._solve_complete, result)

        threading.Thread(target=run, daemon=True).start()

    def _stop_solver(self):
        """Cleanly stop the solver."""
        self._stop_event.set()
        self.lbl_status.config(text="Status: Stopping...", fg=WARNING_COLOR)

    def _solve_complete(self, result):
        self.is_running = False
        self.btn_solve.config(state=tk.NORMAL, text="Solve Again")
        self.btn_stop.config(state=tk.DISABLED)

        status = result.get("status", "Unknown")

        if result.get("success", False):
            # Commit solution
            self.game_state.graph.edges = result["edges"]
            self.canvas.draw()
            self.lbl_time.config(text=f"Time Elapsed: {result['time_taken']:.3f}s")
            self.lbl_status.config(text="Status: ✅ Solved!", fg=APPLE_GREEN)
            self.game_page.check_game_over()
        elif status == "Timeout" or result.get("timed_out", False):
            self.lbl_status.config(text=f"Status: ⏱️ Timeout ({result.get('nodes_visited', 0):,} nodes)", fg=WARNING_COLOR)
            self.lbl_time.config(text=f"Time Elapsed: {result['time_taken']:.3f}s")
        else:
            self.lbl_status.config(text="Status: ❌ No solution found", fg=ERROR_COLOR)
            tk.messagebox.showwarning("Unsolvable", "The current board is mathematically unsolvable!")

    def _on_close(self):
        """Stop solver and close panel."""
        if self.is_running:
            self._stop_event.set()
        self.destroy()
