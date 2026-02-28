"""
Live Analysis Panel
===================
Opens as a separate, resizable Toplevel window with large graphs and a data table.
Shows real-time comparative performance of Greedy, D&C, and DP solvers.
"""

import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ui.styles import *


class LiveAnalysisPanel(tk.Toplevel):
    """
    Separate window for Live Comparative Analysis.
    Opens independently of the main game window.
    """

    def __init__(self, master, game_state):
        super().__init__(master)
        self.game_state = game_state
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        # ── Window Setup ───────────────────────────────────────
        self.title("📊 Live Comparative Analysis")
        self.geometry("1100x750")
        self.minsize(900, 600)
        self.configure(bg=BG_COLOR)

        # ── Header ─────────────────────────────────────────────
        header = tk.Frame(self, bg=BG_COLOR)
        header.pack(fill=tk.X, padx=20, pady=(15, 5))

        tk.Label(
            header,
            text="Live Comparative Analysis",
            font=("Segoe UI", 18, "bold"),
            bg=BG_COLOR,
            fg=TEXT_COLOR,
        ).pack(side=tk.LEFT)

        tk.Label(
            header,
            text="Real-time solver performance comparison",
            font=("Segoe UI", 10),
            bg=BG_COLOR,
            fg=TEXT_DIM,
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
        graph_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)

        self.fig = Figure(figsize=(12, 5), dpi=100, facecolor=BG_COLOR)
        self.fig.subplots_adjust(left=0.06, right=0.97, top=0.90, bottom=0.12, wspace=0.3)

        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)

        self._style_axis(self.ax1, "Execution Time (ms)")
        self._style_axis(self.ax2, "States / Depth")
        self._style_axis(self.ax3, "Cumulative Time (ms)")

        self.canvas = FigureCanvasTkAgg(self.fig, graph_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # ── Table Area ─────────────────────────────────────────
        table_frame = tk.Frame(self, bg=BG_COLOR)
        table_frame.pack(fill=tk.X, padx=20, pady=(0, 15))

        tk.Label(
            table_frame,
            text="Move History",
            font=("Segoe UI", 11, "bold"),
            bg=BG_COLOR, fg=TEXT_DIM,
        ).pack(anchor="w", pady=(0, 5))

        columns = (
            "Move #",
            "Strategy",
            "Legacy Move", "Legacy ms", "Legacy States",
            "Current Move", "Current ms", "Current States",
        )

        style = ttk.Style()
        style.configure(
            "Analysis.Treeview",
            background=CARD_BG,
            foreground=TEXT_COLOR,
            fieldbackground=CARD_BG,
            rowheight=28,
            font=("Consolas", 9),
        )
        style.configure(
            "Analysis.Treeview.Heading",
            background="#1A3A5C",
            foreground=TEXT_COLOR,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
        )
        style.map("Analysis.Treeview", background=[("selected", ACCENT_COLOR)])

        self.tree = ttk.Treeview(
            table_frame,
            columns=columns,
            show="headings",
            height=7,
            style="Analysis.Treeview",
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

        # Track open state
        self._is_open = True

    # ── Styling ────────────────────────────────────────────────

    def _style_axis(self, ax, title):
        ax.set_title(title, fontsize=11, color=TEXT_COLOR, fontweight="bold", pad=10)
        ax.tick_params(colors=TEXT_DIM, labelsize=9)
        ax.set_facecolor(CARD_BG)
        ax.set_xlabel("Move #", fontsize=9, color=TEXT_DIM)
        for spine in ax.spines.values():
            spine.set_color("#333333")
        ax.grid(True, color="#333333", linestyle="--", linewidth=0.5, alpha=0.7)

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

        # 2. Extract data series
        moves = [r.get("move_number") for r in data]

        legacy_times = [self._safe_float(r.get("legacy_time")) for r in data]
        current_times = [self._safe_float(r.get("current_time")) for r in data]

        legacy_states = [self._safe_int(r.get("legacy_states")) for r in data]
        current_states = [self._safe_int(r.get("current_states")) for r in data]

        for row in data:
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

    def _plot_line(self, ax, x, y, color, label=None):
        ax.plot(x, y, marker='o', markersize=5, color=color, label=label,
                linewidth=2.0, markeredgecolor="white", markeredgewidth=0.5)

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
        """Handle window close — hide instead of destroy so we can reopen."""
        self._is_open = False
        self.withdraw()

    def show(self):
        """Show or reopen the window."""
        self._is_open = True
        self.deiconify()
        self.lift()
        self.focus_force()
