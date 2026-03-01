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

        tk.Label(legend_frame, text="\u25a0", font=("Segoe UI", 14), bg=BG_COLOR, fg=C_LEGACY).pack(side=tk.LEFT, padx=(8, 3))
        tk.Label(legend_frame, text="Without Backtracking", font=("Segoe UI", 9, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)
        tk.Label(legend_frame, text="\u25a0", font=("Segoe UI", 14), bg=BG_COLOR, fg=C_CURRENT).pack(side=tk.LEFT, padx=(14, 3))
        tk.Label(legend_frame, text="With Backtracking", font=("Segoe UI", 9, "bold"), bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT)

        # ── Speedup Banner ─────────────────────────────────────
        self.speedup_frame = tk.Frame(self, bg="#0A2540")
        self.speedup_frame.pack(fill=tk.X, padx=20, pady=(5, 2))

        self.lbl_speedup_banner = tk.Label(
            self.speedup_frame,
            text="\u26a1 Play moves to see the comparison!",
            font=("Segoe UI", 14, "bold"),
            bg="#0A2540", fg=C_SPEEDUP_MED,
        )
        self.lbl_speedup_banner.pack(pady=8)

        # ── Complexity Info ────────────────────────────────────
        self.complexity_label = tk.Label(
            self, text="", font=("Consolas", 9, "italic"),
            bg=BG_COLOR, fg="#666666",
        )
        self.complexity_label.pack(anchor="w", padx=25)

        # ── Graphs Area (2 graphs: 60/40 split) ───────────────
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
            "Move #", "Strategy",
            "Legacy Move", "Legacy ms",
            "Current Move", "Current ms",
            "Speedup",
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
            "Move #": 60, "Strategy": 110,
            "Legacy Move": 120, "Legacy ms": 90,
            "Current Move": 120, "Current ms": 90,
            "Speedup": 80,
        }
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=col_widths.get(col, 90), anchor="center")

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

        # ─── Complexity label ───
        strategy = self.game_state.solver_strategy
        complexity_map = {
            "dynamic_programming": "Legacy: O(2^E) brute-force  vs  Current: O(V \u00b7 2^E_rem) pruned",
            "divide_conquer": "Legacy: O(2^(E/4) \u00b7 4) naive  vs  Current: O(V \u00b7 2^E_rem) pruned",
            "advanced_dp": "Legacy: O(2^(E/4) \u00b7 N) region DP  vs  Current: O(V \u00b7 2^E_rem) pruned",
        }
        self.complexity_label.config(text=complexity_map.get(strategy, ""))

        # ─── Extract data ───
        moves = [r.get("move_number") for r in data]
        n = len(moves)

        legacy_times = [self._safe_float(r.get("legacy_time")) for r in data]
        current_times = [self._safe_float(r.get("current_time")) for r in data]
        speedups = [self._safe_float(r.get("speedup", 1.0)) for r in data]

        # ─── Speedup Banner ───
        avg_speedup = sum(speedups) / len(speedups) if speedups else 1.0
        max_speedup = max(speedups) if speedups else 1.0

        if max_speedup >= 10:
            banner_text = "\u26a1 Backtracking is {:.0f}\u00d7 faster on average! (peak: {:.0f}\u00d7)".format(avg_speedup, max_speedup)
            banner_color = C_CURRENT
        elif max_speedup >= 2:
            banner_text = "\u26a1 Backtracking is {:.1f}\u00d7 faster on average".format(avg_speedup)
            banner_color = C_SPEEDUP_MED
        else:
            banner_text = "\u26a1 Both solvers comparable \u2014 try larger boards!"
            banner_color = C_SPEEDUP_LOW
        self.lbl_speedup_banner.config(text=banner_text, fg=banner_color)

        # ─── Update Table ───
        for item in self.tree.get_children():
            self.tree.delete(item)

        self.tree.tag_configure("timeout_row", background="#3E2723", foreground="#FFCCBC")
        self.tree.tag_configure("fast_row", background="#1B3A2A", foreground="#A5D6A7")

        for row in data:
            speedup_val = self._safe_float(row.get("speedup", 1.0))
            speedup_str = "{:.0f}x".format(speedup_val) if speedup_val >= 10 else "{:.1f}x".format(speedup_val)

            values = (
                row.get("move_number"), row.get("strategy"),
                row.get("legacy_move"), row.get("legacy_time"),
                row.get("current_move"), row.get("current_time"),
                speedup_str,
            )

            l_move = str(row.get("legacy_move", ""))
            if l_move in ("Timeout", "Error", "Skipped (>5x5)"):
                tags = ("timeout_row",)
            elif speedup_val >= 10:
                tags = ("fast_row",)
            else:
                tags = ()
            self.tree.insert("", "end", values=values, tags=tags)

        # ═══════════════════════════════════════════════════════
        #  DRAW TWO GRAPHS
        # ═══════════════════════════════════════════════════════
        self.ax1.clear()
        self.ax2.clear()

        x = np.arange(n)
        bar_w = 0.35

        # ──────────────────────────────────────────────────────
        #  GRAPH 1: Execution Time — WIDE BAR CHART
        #  Takes 60% of the width for maximum clarity
        # ──────────────────────────────────────────────────────
        self.ax1.bar(
            x - bar_w / 2, legacy_times, bar_w,
            color=C_LEGACY, alpha=0.85, label="Without Backtracking",
            edgecolor="#992222", linewidth=0.3, zorder=3,
        )
        self.ax1.bar(
            x + bar_w / 2, current_times, bar_w,
            color=C_CURRENT, alpha=0.85, label="With Backtracking",
            edgecolor="#1B8A3A", linewidth=0.3, zorder=3,
        )

        # Timeout annotation: mark legacy bars that hit 3000ms
        for i, lt in enumerate(legacy_times):
            if lt >= 2900:
                self.ax1.text(
                    i - bar_w / 2, lt * 1.02, "TIMEOUT",
                    ha="center", va="bottom", fontsize=6,
                    color="#FF8A80", fontweight="bold", rotation=90,
                )

        self.ax1.set_xticks(x)
        self.ax1.set_xticklabels([str(m) for m in moves], fontsize=8)
        self._style_axis(self.ax1, "Execution Time per Move", "milliseconds (ms)")
        self.ax1.set_ylim(bottom=0)

        # Smart scale: use log only when gap is massive
        max_t = max(max(legacy_times, default=0), max(current_times, default=0))
        min_t = min((t for t in current_times if t > 0), default=1.0)
        if max_t > 0 and max_t / max(min_t, 0.01) > 15:
            self.ax1.set_yscale("symlog", linthresh=1.0)
            # Add horizontal reference lines
            for ref in [1, 10, 100, 1000]:
                if ref < max_t:
                    self.ax1.axhline(y=ref, color="#333", linestyle=":", linewidth=0.5, alpha=0.4)
                    self.ax1.text(
                        -0.6, ref, "{}ms".format(ref),
                        fontsize=6, color="#555", va="center",
                    )

        self.ax1.legend(
            fontsize=8, facecolor="#0D1B2A", edgecolor="#333",
            labelcolor=TEXT_COLOR, loc="upper left",
            framealpha=0.9,
        )

        # ──────────────────────────────────────────────────────
        #  GRAPH 2: Speedup Factor — COLOR-CODED BARS
        #  Shows "150x", "200x" etc.
        # ──────────────────────────────────────────────────────
        colors = []
        for s in speedups:
            if s >= 50:
                colors.append(C_SPEEDUP_HOT)
            elif s >= 10:
                colors.append("#69F0AE")
            elif s >= 5:
                colors.append(C_SPEEDUP_MED)
            elif s >= 2:
                colors.append(C_SPEEDUP_LOW)
            else:
                colors.append("#666666")

        self.ax2.bar(
            x, speedups, width=0.55,
            color=colors, alpha=0.9,
            edgecolor="#FFFFFF22", linewidth=0.3, zorder=3,
        )

        # Add speedup labels on top of each bar
        max_spd = max(speedups, default=1)
        for i, s in enumerate(speedups):
            label = "{:.0f}\u00d7".format(s) if s >= 10 else "{:.1f}\u00d7".format(s)
            y_pos = s + max_spd * 0.02
            self.ax2.text(
                i, y_pos, label,
                ha="center", va="bottom", fontsize=9,
                color="white", fontweight="bold",
            )

        # Reference line at 1x
        self.ax2.axhline(y=1, color="#444444", linestyle="--", linewidth=0.8, alpha=0.6)

        self.ax2.set_xticks(x)
        self.ax2.set_xticklabels([str(m) for m in moves], fontsize=8)
        self._style_axis(self.ax2, "Speedup Factor", "times faster (\u00d7)")
        self.ax2.set_ylim(bottom=0, top=max_spd * 1.2 if max_spd > 1 else 5)

        if max_spd > 50:
            self.ax2.set_yscale("symlog", linthresh=1.0)
            self.ax2.set_ylim(top=max_spd * 1.3)

        try:
            self.fig.tight_layout(pad=1.5)
        except Exception:
            pass
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
