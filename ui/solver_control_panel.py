    import tkinter as tk
import time
from threading import Thread
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
        self.solver = DPBacktrackingSolver(self.game_state)
        
        self.title("DP & Backtracking Auto-Solver")
        self.geometry("400x530")
        self.resizable(False, False)
        self.transient(master)
        
        # Keep on top
        self.attributes("-topmost", True)
        
        self._setup_ui()
        self.is_running = False

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

        # Controls
        ctrl_frame = tk.Frame(self, bg=BG_COLOR)
        ctrl_frame.pack(fill=tk.X, padx=20, pady=20)
        
        self.btn_solve = HoverButton(ctrl_frame, text="Start DP Solve", command=self._start_solve_thread, width=15, fg=SUCCESS_COLOR)
        self.btn_solve.pack(side=tk.LEFT, padx=5)
        
        HoverButton(ctrl_frame, text="Close", command=self.destroy, width=15, fg=WARNING_COLOR).pack(side=tk.RIGHT, padx=5)

    def _ui_update_callback(self, current_edges):
        """Called by solver during backtracking to update UI."""
        if not self.winfo_exists(): return
        
        # 1. Update Canvas
        self.game_state.graph.edges = set(current_edges)
        self.canvas.draw()
        
        # 2. Update Live Stats
        # Safely wrap since this is called from a thread
        try:
            nodes = getattr(self.solver, "nodes_visited", 0)
            cache = len([v for v in self.solver.dp_cache.values() if v is False])
            
            self.lbl_nodes.config(text=f"Nodes Visited: {nodes}")
            self.lbl_cache.config(text=f"DP Cache Hits (Pruned): {cache}")
        except Exception:
            pass

    def _start_solve_thread(self):
        if self.is_running: return
        self.is_running = True
        self.btn_solve.config(state=tk.DISABLED, text="Solving...")
        
        def run():
            delay = self.speed_var.get()
            result = self.solver.solve(ui_callback=self._ui_update_callback, delay=delay)
            
            # Revert to main thread updates for completion
            if self.winfo_exists():
                self.after(0, self._solve_complete, result)
        
        Thread(target=run, daemon=True).start()

    def _solve_complete(self, result):
        self.is_running = False
        self.btn_solve.config(state=tk.NORMAL, text="Solve Again")
        
        if result["success"]:
            # Commit solution
            self.game_state.graph.edges = result["edges"]
            self.canvas.draw()
            self.lbl_time.config(text=f"Time Elapsed: {result['time_taken']:.3f}s")
            self.game_page.check_game_over()
        else:
            tk.messagebox.showwarning("Unsolvable", "The current board is mathematically unsolvable!")
