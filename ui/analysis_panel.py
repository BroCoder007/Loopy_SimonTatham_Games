"""
Live Analysis Panel
===================
Visually dramatic comparative analysis window.
Two large, clear graphs demonstrate WHY backtracking is superior:

1. Execution Time   — Wide bar chart showing per-move time gap
2. Speedup Factor   — Color-coded bars showing how many times faster
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.gridspec as gridspec
import numpy as np
from ui.styles import *


# Colors
C_LEGACY = "#FF453A"        # Red — Without Backtracking
C_CURRENT = "#30D158"       # Green — With Backtracking
C_SPEEDUP_HOT = "#00E676"
C_SPEEDUP_MED = "#5AC8FA"
C_SPEEDUP_LOW = "#FFCC00"


class LiveAnalysisPanel(tk.Toplevel):
    """
    Premium visual comparison: Legacy vs Current.
    Two clear, spacious graphs — no clutter.
    """

    def __init__(self, master, game_state):
        super().__init__(master)
        self.game_state = game_state
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Window Setup ───────────────────────────────────────
        self.title("\U0001f4ca Live Comparative Analysis")
        self.geometry("1200x750")
        self.minsize(950, 600)
        self.configure(bg=BG_COLOR)

        # ── Header ─────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_COLOR)
        header.pack(fill=tk.X, padx=20, pady=(15, 5))

        tk.Label(
            header, text="Live Comparative Analysis",
            font=("Segoe UI", 18, "bold"), bg=BG_COLOR, fg=TEXT_COLOR,
        ).pack(side=tk.LEFT)

        tk.Label(
            header, text="Why Backtracking Wins",
            font=("Segoe UI", 11, "italic"), bg=BG_COLOR, fg="#FFCC00",
        ).pack(side=tk.LEFT, padx=15)

        # Legend
        legend_frame = tk.Frame(header, bg=BG_COLOR)
        legend_frame.pack(side=tk.RIGHT)
        for label, color in [("Without Backtracking (Legacy)", "#FF3B30"), ("With Backtracking (Current)", "#30D158")]:
            tk.Label(legend_frame, text="●", font=("Segoe UI", 12), bg=BG_COLOR, fg=color).pack(side=tk.LEFT, padx=(8, 2))
            tk.Label(legend_frame, text=label, font=("Segoe UI", 9), bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)

        # Complexity Info
        self.complexity_label = tk.Label(
            header,
            text="",
            font=("Consolas", 10, "italic"),
            bg=BG_COLOR,
            fg="#FFCC00"  # Yellowish to make it pop
        )
        self.complexity_label.pack(side=tk.LEFT, padx=15)

        # ── Graphs Area ────────────────────────────────────────
        graph_frame = tk.Frame(self, bg=BG_COLOR)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)

        self.fig = Figure(figsize=(14, 5), dpi=100, facecolor=BG_COLOR)
        gs = gridspec.GridSpec(1, 2, width_ratios=[3, 2], wspace=0.25)

        self.ax1 = self.fig.add_subplot(gs[0])  # Execution Time — WIDE
        self.ax2 = self.fig.add_subplot(gs[1])  # Speedup Factor

        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Table Area ─────────────────────────────────────────
        table_frame = tk.Frame(self, bg=BG_COLOR)
        table_frame.pack(fill=tk.X, padx=20, pady=(0, 12))

        tk.Label(
            table_frame, text="Move-by-Move Data",
            font=("Segoe UI", 11, "bold"), bg=BG_COLOR, fg=TEXT_DIM,
        ).pack(anchor="w", pady=(0, 4))

        columns = (
            "Move #",
            "Strategy",
            "Legacy Move", "Legacy ms", "Legacy States",
            "Current Move", "Current ms", "Current States",
        )

        style = ttk.Style()
        style.configure(
            "Analysis.Treeview",
            background=CARD_BG, foreground=TEXT_COLOR,
            fieldbackground=CARD_BG, rowheight=26,
            font=("Consolas", 9),
        )
        style.configure(
            "Analysis.Treeview.Heading",
            background="#1A3A5C", foreground=TEXT_COLOR,
            font=("Segoe UI", 9, "bold"), relief="flat",
        )
        style.map("Analysis.Treeview", background=[("selected", ACCENT_COLOR)])

        self.tree = ttk.Treeview(
            table_frame, columns=columns, show="headings",
            height=5, style="Analysis.Treeview",
        )

        col_widths = {
            "Move #": 50,
            "Strategy": 110,
            "Legacy Move": 110, "Legacy ms": 85, "Legacy States": 95,
            "Current Move": 110, "Current ms": 85, "Current States": 95,
        }
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_widths.get(col, 100), anchor="center")

        scrollbar = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.pack(side=tk.LEFT, fill=tk.X, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        self._is_open = True

    # ── Graph Styling ──────────────────────────────────────────

    def _style_axis(self, ax, title, ylabel=""):
        ax.set_title(title, fontsize=13, color=TEXT_COLOR, fontweight="bold", pad=12)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.set_facecolor("#0D1B2A")
        ax.set_xlabel("Move #", fontsize=9, color=TEXT_DIM)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color=TEXT_DIM)
        for spine in ax.spines.values():
            spine.set_color("#1B2838")
        ax.grid(True, color="#1B2838", linestyle="-", linewidth=0.4, alpha=0.5, axis="y")

    # ── Data Update ────────────────────────────────────────────

    def update_data(self):
        """Refresh graphs and table from game_state.live_analysis_table."""
        if not self._is_open:
            return

        data = self.game_state.live_analysis_table
        if not data:
            return

        # 1. Update Table & Complexity Label
        strategy = self.game_state.solver_strategy
        if strategy == "dynamic_programming":
            self.complexity_label.config(text="Theory: Legacy DP is O(2^E) vs Current DP is O(V * 2^(E_rem))")
        elif strategy == "divide_conquer":
            self.complexity_label.config(text="Theory: Legacy D&C is O(2^(E/4) * 4) vs Current D&C is O(V * 2^(E_rem))")
        elif strategy == "advanced_dp":
            self.complexity_label.config(text="Theory: Legacy Adv DP is O(2^(E/4) * N) vs Current Adv DP is O(V * 2^(E_rem))")
        else:
            self.complexity_label.config(text="")

        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree.tag_configure("greedy_fail", background="#3E2723", foreground="#FFCCBC") # Dark reddish background

        # ─── Extract data ───
        moves = [r.get("move_number") for r in data]
        n = len(moves)

        legacy_times = [self._safe_float(r.get("legacy_time")) for r in data]
        current_times = [self._safe_float(r.get("current_time")) for r in data]

        legacy_states = [self._safe_int(r.get("legacy_states")) for r in data]
        current_states = [self._safe_int(r.get("current_states")) for r in data]

        for row in data:
            speedup_val = self._safe_float(row.get("speedup", 1.0))
            speedup_str = "{:.0f}x".format(speedup_val) if speedup_val >= 10 else "{:.1f}x".format(speedup_val)

            values = (
                row.get("move_number"),
                row.get("strategy"),
                row.get("legacy_move"), row.get("legacy_time"), row.get("legacy_states"),
                row.get("current_move"), row.get("current_time"), row.get("current_states"),
            )
            
            # Highlight timeouts or errors in legacy
            l_move = str(row.get("legacy_move"))
            
            tags = ()
            if l_move in ("None", "N/A", "Timeout", "Error"):
                 tags = ("legacy_fail",)
                 
            self.tree.insert("", "end", values=values, tags=tags)

        import itertools
        legacy_cum = list(itertools.accumulate(legacy_times))
        current_cum = list(itertools.accumulate(current_times))

        # 3. Clear and redraw
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        c_legacy = "#FF3B30"
        c_current = "#30D158"

        # Graph 1: Execution Time
        self._plot_line(self.ax1, moves, legacy_times, c_legacy, "Without Backtracking")
        self._plot_line(self.ax1, moves, current_times, c_current, "With Backtracking")
        self._style_axis(self.ax1, "Execution Time (ms)")
        self.ax1.set_yscale('symlog', linthresh=1.0)
        self.ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#555555", labelcolor=TEXT_COLOR)

        # Graph 2: States / Depth
        self._plot_line(self.ax2, moves, legacy_states, c_legacy, "Without Backtracking")
        self._plot_line(self.ax2, moves, current_states, c_current, "With Backtracking")
        self._style_axis(self.ax2, "States Explored")
        self.ax2.set_yscale('symlog', linthresh=1.0)
        self.ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#555555", labelcolor=TEXT_COLOR)

        # Graph 3: Cumulative Time
        self._plot_line(self.ax3, moves, legacy_cum, c_legacy, "Without Backtracking")
        self._plot_line(self.ax3, moves, current_cum, c_current, "With Backtracking")
        self._style_axis(self.ax3, "Cumulative Time (ms)")
        self.ax3.set_yscale('symlog', linthresh=1.0)
        self.ax3.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#555555", labelcolor=TEXT_COLOR)

        self.canvas.draw()

    # ── Helpers ─────────────────────────────────────────────────

    def _safe_float(self, val):
        try:
            return float(val)
        except Exception:
            return 0.0

    def _safe_int(self, val):
        try:
            return int(val)
        except Exception:
            return 0

    def _on_close(self):
        self._is_open = False
        self.withdraw()

    def show(self):
        self._is_open = True
        self.deiconify()
        self.lift()
        self.focus_force()

class AIVsAIAnalysisPanel(tk.Toplevel):
    """
    Live comparison of two AIs playing against each other.
    """
    def __init__(self, master, game_state):
        super().__init__(master)
        self.game_state = game_state
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.title("\U0001f916 AI vs AI Live Stats")
        self.geometry("1100x650")
        self.minsize(900, 500)
        self.configure(bg=BG_COLOR)

        header = tk.Frame(self, bg=BG_COLOR)
        header.pack(fill=tk.X, padx=20, pady=(15, 5))

        tk.Label(
            header, text="AI vs AI Performance",
            font=("Segoe UI", 18, "bold"), bg=BG_COLOR, fg=TEXT_COLOR,
        ).pack(side=tk.LEFT)
        
        # Legend
        legend_frame = tk.Frame(header, bg=BG_COLOR)
        legend_frame.pack(side=tk.RIGHT)

        tk.Label(legend_frame, text="\u25a0", font=("Segoe UI", 14), bg=BG_COLOR, fg="#FF9F0A").pack(side=tk.LEFT, padx=(8, 3))
        tk.Label(legend_frame, text="Player 1 (CPU)", font=("Segoe UI", 9, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="\u25a0", font=("Segoe UI", 14), bg=BG_COLOR, fg="#0A84FF").pack(side=tk.LEFT, padx=(14, 3))
        tk.Label(legend_frame, text="Player 2 (CPU)", font=("Segoe UI", 9, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)

        # Labels for strategy names
        self.lbl_p1_strat = tk.Label(self, text=f"P1: {self.game_state.solver_strategy_p1}", font=("Consolas", 11, "italic"), bg=BG_COLOR, fg="#FF9F0A")
        self.lbl_p1_strat.pack(anchor="w", padx=25)
        
        self.lbl_p2_strat = tk.Label(self, text=f"P2: {self.game_state.solver_strategy_p2}", font=("Consolas", 11, "italic"), bg=BG_COLOR, fg="#0A84FF")
        self.lbl_p2_strat.pack(anchor="w", padx=25, pady=(0, 10))

        # Graph
        graph_frame = tk.Frame(self, bg=BG_COLOR)
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=8)

        self.fig = Figure(figsize=(10, 4), dpi=100, facecolor=BG_COLOR)
        gs = gridspec.GridSpec(1, 1)
        self.ax1 = self.fig.add_subplot(gs[0])

        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self._is_open = True

    def _style_axis(self, ax, title, ylabel=""):
        ax.set_title(title, fontsize=13, color=TEXT_COLOR, fontweight="bold", pad=12)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.set_facecolor("#0D1B2A")
        ax.set_xlabel("Move # (Pairs)", fontsize=9, color=TEXT_DIM)
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=9, color=TEXT_DIM)
        for spine in ax.spines.values():
            spine.set_color("#1B2838")
        ax.grid(True, color="#1B2838", linestyle="-", linewidth=0.4, alpha=0.5, axis="y")

    def update_data(self):
        if not self._is_open: return
        stats = getattr(self.game_state, "ai_vs_ai_stats", [])
        if not stats: return

        # Group by rounds (pairs of moves)
        p1_times = []
        p2_times = []
        
        for s in stats:
            t = s["time"] * 1000  # ms
            if "Player 1" in s["player"]:
                p1_times.append(t)
            else:
                p2_times.append(t)

        n = max(len(p1_times), len(p2_times))
        
        # Pad with 0s if uneven
        while len(p1_times) < n: p1_times.append(0)
        while len(p2_times) < n: p2_times.append(0)

        self.ax1.clear()
        
        x = np.arange(n)
        bar_w = 0.35

        self.ax1.bar(
            x - bar_w / 2, p1_times, bar_w,
            color="#FF9F0A", alpha=0.85, label="Player 1",
            edgecolor="#CC7A00", linewidth=0.3, zorder=3,
        )
        self.ax1.bar(
            x + bar_w / 2, p2_times, bar_w,
            color="#0A84FF", alpha=0.85, label="Player 2",
            edgecolor="#005A9E", linewidth=0.3, zorder=3,
        )

        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels([str(i+1) for i in x], fontsize=8)
        self._style_axis(self.ax1, "Execution Time per Move (ms)", "milliseconds (ms)")
        self.ax1.set_ylim(bottom=0)

        self.ax1.legend(
            fontsize=8, facecolor="#0D1B2A", edgecolor="#333",
            labelcolor=TEXT_COLOR, loc="upper left", framealpha=0.9,
        )

        try:
            self.fig.tight_layout(pad=1.5)
        except:
            pass
        self.canvas.draw()

    def _on_close(self):
        self._is_open = False
        self.withdraw()

    def show(self):
        self._is_open = True
        self.deiconify()
        self.lift()
        self.focus_force()

