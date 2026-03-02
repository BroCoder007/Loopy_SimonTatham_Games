"""
CPU Reasoning Panel
===================
Cognitive Visualization Layer - Sidebar Component.
Displays mode-specific reasoning, scope, and decision metrics.
Designed to fit in the left sidebar of MainWindow.
"""

import tkinter as tk
from tkinter import ttk
from typing import Any, Dict, List, Optional

from ui.styles import BG_COLOR, SIDEBAR_COLOR, TEXT_COLOR, TEXT_DIM, ACCENT_COLOR, FONT_BODY, FONT_MONO
from logic.solvers.solver_interface import MoveExplanation


class CPUReasoningPanel(tk.Frame):
    """
    Live panel that updates after every CPU move.
    Shows: Mode badge, scope, decision summary, mode-specific metrics,
    and a scrollable move history log.
    """

    def __init__(self, parent: tk.Widget, **kwargs):
        super().__init__(parent, bg=SIDEBAR_COLOR, height=350, **kwargs)
        # Allow the frame to maintain its requested height
        self.pack_propagate(True)

        # ── Header ──────────────────────────────────────────────
        header_frame = tk.Frame(self, bg=SIDEBAR_COLOR)
        header_frame.pack(fill=tk.X, padx=10, pady=(10, 2))

        tk.Label(
            header_frame,
            text="🧠 CPU REASONING",
            font=("Segoe UI", 10, "bold"),
            bg=SIDEBAR_COLOR,
            fg=ACCENT_COLOR,
        ).pack(side=tk.LEFT)

        # ── Mode Badge (e.g. "Greedy • Local") ─────────────────
        self.mode_badge = tk.Label(
            self,
            text="Waiting...",
            font=("Segoe UI", 9, "bold"),
            bg="#444444",
            fg="white",
            padx=8,
            pady=2,
        )
        self.mode_badge.pack(anchor="w", padx=10, pady=4)

        # ── Decision Summary ───────────────────────────────────
        tk.Label(
            self, text="Decision:", font=("Segoe UI", 9, "bold"),
            bg=SIDEBAR_COLOR, fg=TEXT_DIM
        ).pack(anchor="w", padx=10)

        self.summary_text = tk.Text(
            self,
            height=3,
            bg="#132F4C",
            fg=TEXT_COLOR,
            font=FONT_BODY,
            wrap=tk.WORD,
            bd=1,
            relief=tk.FLAT,
            state=tk.DISABLED,
        )
        self.summary_text.pack(fill=tk.X, padx=10, pady=(2, 5))

        # ── Metrics Container ──────────────────────────────────
        self.metrics_frame = tk.Frame(self, bg=SIDEBAR_COLOR)
        self.metrics_frame.pack(fill=tk.X, padx=10)

        # Pre-built widgets (packed/unpacked dynamically)
        self.certainty_label = tk.Label(self.metrics_frame, text="Certainty:", bg=SIDEBAR_COLOR, fg=TEXT_DIM, font=FONT_BODY)
        self.certainty_bar = ttk.Progressbar(self.metrics_frame, orient=tk.HORIZONTAL, length=100, mode='determinate')
        self.certainty_val = tk.Label(self.metrics_frame, text="0%", bg=SIDEBAR_COLOR, fg=ACCENT_COLOR, font=("Segoe UI", 10, "bold"))

        self.detail_label_1 = tk.Label(self.metrics_frame, text="", bg=SIDEBAR_COLOR, fg=TEXT_DIM, font=FONT_BODY, anchor="w")
        self.detail_val_1 = tk.Label(self.metrics_frame, text="", bg=SIDEBAR_COLOR, fg=TEXT_COLOR, font=FONT_MONO, anchor="w")
        self.detail_label_2 = tk.Label(self.metrics_frame, text="", bg=SIDEBAR_COLOR, fg=TEXT_DIM, font=FONT_BODY, anchor="w")
        self.detail_val_2 = tk.Label(self.metrics_frame, text="", bg=SIDEBAR_COLOR, fg=TEXT_COLOR, font=FONT_MONO, anchor="w")

        # ── Move History Log ───────────────────────────────────
        tk.Label(
            self, text="Move Log:", font=("Segoe UI", 9, "bold"),
            bg=SIDEBAR_COLOR, fg=TEXT_DIM
        ).pack(anchor="w", padx=10, pady=(8, 0))

        self.history_text = tk.Text(
            self,
            height=6,
            bg="#132F4C",
            fg=TEXT_DIM,
            font=("Consolas", 8),
            wrap=tk.WORD,
            bd=1,
            relief=tk.FLAT,
            state=tk.DISABLED,
        )
        self.history_text.pack(fill=tk.BOTH, padx=10, pady=(2, 10), expand=True)

        self._move_count = 0

    # ── Public API ─────────────────────────────────────────────

    def update_explanation(self, explanation: Optional[MoveExplanation]):
        """Update the panel with new reasoning data."""
        if not explanation:
            self._set_waiting_state()
            return

        self._move_count += 1

        # 1. Update Mode Badge
        scope_color = {
            "Local": "#28a745",     # Green
            "Regional": "#17a2b8",  # Blue
            "Global": "#6f42c1",    # Purple
        }.get(explanation.scope, "#444444")

        badge_text = f"{explanation.mode}  •  {explanation.scope}"
        self.mode_badge.config(text=badge_text, bg=scope_color)

        # 2. Update Summary
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", explanation.decision_summary)
        self.summary_text.config(state=tk.DISABLED)

        # 3. Update Metrics based on Mode
        self._clear_metrics()

        if explanation.mode == "Dynamic Programming":
            self._show_dp_metrics(explanation.reasoning_data)
        elif explanation.mode == "Divide & Conquer":
            self._show_dnc_metrics(explanation.reasoning_data)
        elif explanation.mode == "Greedy":
            self._show_greedy_metrics(explanation.reasoning_data)

        # 4. Append to history log
        self._append_history(explanation)

    # ── History Log ────────────────────────────────────────────

    def _append_history(self, expl: MoveExplanation):
        edge_str = ""
        if expl.highlight_edges:
            edge_str = str(expl.highlight_edges[0])

        entry = f"#{self._move_count} [{expl.mode[:3].upper()}] {edge_str} — {expl.decision_summary[:60]}\n"
        self.history_text.config(state=tk.NORMAL)
        self.history_text.insert(tk.END, entry)
        self.history_text.see(tk.END)
        self.history_text.config(state=tk.DISABLED)

    # ── Internal Helpers ───────────────────────────────────────

    def _set_waiting_state(self):
        self.mode_badge.config(text="Waiting for CPU...", bg="#444444")
        self.summary_text.config(state=tk.NORMAL)
        self.summary_text.delete("1.0", tk.END)
        self.summary_text.insert("1.0", "CPU has not moved yet.")
        self.summary_text.config(state=tk.DISABLED)
        self._clear_metrics()

    def _clear_metrics(self):
        for widget in self.metrics_frame.winfo_children():
            widget.pack_forget()

    def _show_dp_metrics(self, data: Dict[str, Any]):
        certainty = data.get("certainty", 0.0) * 100
        solutions = data.get("total_solutions", 0)
        freq = data.get("edge_frequency", 0)

        self.certainty_label.pack(anchor="w")
        self.certainty_bar.pack(fill=tk.X, pady=(2, 0))
        self.certainty_bar["value"] = certainty
        self.certainty_val.config(text=f"{certainty:.1f}%")
        self.certainty_val.pack(anchor="e")

        self.detail_label_1.config(text=f"Total Solutions Enumerated: {solutions}")
        self.detail_label_1.pack(anchor="w", pady=(5, 0))

        self.detail_label_2.config(text=f"Edge Frequency: {freq}/{solutions}")
        self.detail_label_2.pack(anchor="w")

    def _show_dnc_metrics(self, data: Dict[str, Any]):
        depth = data.get("recursion_depth", 0)
        stage = data.get("merge_stage", "unknown")
        region = data.get("region_id", "N/A")

        self.detail_label_1.config(text=f"Recursion Depth: {depth}")
        self.detail_label_1.pack(anchor="w")

        self.detail_val_1.config(text=f"Stage: {stage}")
        self.detail_val_1.pack(anchor="w")

        self.detail_label_2.config(text=f"Region: {region}")
        self.detail_label_2.pack(anchor="w")

    def _show_greedy_metrics(self, data: Dict[str, Any]):
        rule = data.get("rule", "Union Rule")

        self.detail_label_1.config(text="Rule Applied:")
        self.detail_label_1.pack(anchor="w")

        self.detail_val_1.config(text=rule)
        self.detail_val_1.pack(anchor="w", pady=(2, 0))
