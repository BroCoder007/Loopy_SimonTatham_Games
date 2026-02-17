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
        for label, color in [("Greedy", "#30D158"), ("D&C", "#FF9F0A"), ("DP", "#0A84FF")]:
            tk.Label(legend_frame, text="●", font=("Segoe UI", 12), bg=BG_COLOR, fg=color).pack(side=tk.LEFT, padx=(8, 2))
            tk.Label(legend_frame, text=label, font=("Segoe UI", 9), bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)

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
            "Greedy Move", "Greedy ms", "Greedy States",
            "D&C Move", "D&C ms", "D&C Depth",
            "DP Move", "DP ms", "DP States",
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
            "Move #": 55,
            "Greedy Move": 120, "Greedy ms": 75, "Greedy States": 85,
            "D&C Move": 120, "D&C ms": 75, "D&C Depth": 75,
            "DP Move": 120, "DP ms": 75, "DP States": 85,
        }
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_widths.get(col, 80), anchor="center")

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

        # 1. Update Table
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree.tag_configure("greedy_fail", background="#3E2723", foreground="#FFCCBC") # Dark reddish background

        # 2. Extract data series
        moves = [r.get("move_number") for r in data]

        greedy_times = [self._safe_float(r.get("greedy_time")) for r in data]
        dnc_times = [self._safe_float(r.get("dnc_time")) for r in data]
        dp_times = [self._safe_float(r.get("dp_time")) for r in data]

        greedy_states = [self._safe_int(r.get("greedy_states")) for r in data]
        dnc_states = [self._safe_int(r.get("dnc_states")) for r in data]
        dp_states = [self._safe_int(r.get("dp_states")) for r in data]

        for row in data:
            values = (
                row.get("move_number"),
                row.get("greedy_move"), row.get("greedy_time"), row.get("greedy_states"),
                row.get("dnc_move"), row.get("dnc_time"), row.get("dnc_states"),
                row.get("dp_move"), row.get("dp_time"), row.get("dp_states"),
            )
            
            # Check for Greedy Failure vs Others Success
            g_move = str(row.get("greedy_move"))
            d_move = str(row.get("dnc_move"))
            dp_move = str(row.get("dp_move"))
            
            tags = ()
            if g_move in ("None", "N/A", "Timeout") and (d_move not in ("None", "N/A", "Timeout") or dp_move not in ("None", "N/A", "Timeout")):
                 tags = ("greedy_fail",)
                 
            self.tree.insert("", "end", values=values, tags=tags)

        import itertools
        greedy_cum = list(itertools.accumulate(greedy_times))
        dnc_cum = list(itertools.accumulate(dnc_times))
        dp_cum = list(itertools.accumulate(dp_times))

        # 3. Clear and redraw
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()

        c_greedy = "#30D158"
        c_dnc = "#FF9F0A"
        c_dp = "#0A84FF"

        # Graph 1: Execution Time
        self._plot_line(self.ax1, moves, greedy_times, c_greedy, "Greedy")
        self._plot_line(self.ax1, moves, dnc_times, c_dnc, "D&C")
        self._plot_line(self.ax1, moves, dp_times, c_dp, "DP")
        self._style_axis(self.ax1, "Execution Time (ms)")
        self.ax1.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#555555", labelcolor=TEXT_COLOR)

        # Graph 2: States / Depth
        self._plot_line(self.ax2, moves, greedy_states, c_greedy, "Greedy")
        self._plot_line(self.ax2, moves, dnc_states, c_dnc, "D&C")
        self._plot_line(self.ax2, moves, dp_states, c_dp, "DP")
        self._style_axis(self.ax2, "States / Depth")
        self.ax2.legend(fontsize=8, facecolor=CARD_BG, edgecolor="#555555", labelcolor=TEXT_COLOR)

        # Graph 3: Cumulative Time
        self._plot_line(self.ax3, moves, greedy_cum, c_greedy, "Greedy")
        self._plot_line(self.ax3, moves, dnc_cum, c_dnc, "D&C")
        self._plot_line(self.ax3, moves, dp_cum, c_dp, "DP")
        self._style_axis(self.ax3, "Cumulative Time (ms)")
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
