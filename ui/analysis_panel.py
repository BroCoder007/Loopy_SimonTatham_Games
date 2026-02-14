import tkinter as tk
from tkinter import ttk
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from ui.styles import *

class LiveAnalysisPanel(tk.Frame):
    def __init__(self, master, game_state):
        super().__init__(master, bg=BG_COLOR)
        self.game_state = game_state
        
        # Main Layout: Top (Graphs), Bottom (Table)
        self.top_frame = tk.Frame(self, bg=BG_COLOR)
        self.top_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, pady=10)
        
        self.bottom_frame = tk.Frame(self, bg=BG_COLOR)
        self.bottom_frame.pack(side=tk.BOTTOM, fill=tk.X, expand=False, pady=10)
        
        # --- Graphs ---
        self.setup_graphs()
        
        # --- Table ---
        self.setup_table()

    def setup_graphs(self):
        # 3 Subplots: Execution Time, States Explored, Cumulative Time
        self.fig = Figure(figsize=(8, 3), dpi=100, facecolor=BG_COLOR)
        self.ax1 = self.fig.add_subplot(131)
        self.ax2 = self.fig.add_subplot(132)
        self.ax3 = self.fig.add_subplot(133)
        
        self.fig.tight_layout(pad=2.0)
        
        # Initial empty plots
        self._plot_empty(self.ax1, "Exec Time (ms)")
        self._plot_empty(self.ax2, "States Explored")
        self._plot_empty(self.ax3, "Cumulative Time (ms)")
        
        self.canvas = FigureCanvasTkAgg(self.fig, self.top_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    def _plot_empty(self, ax, title):
        ax.set_title(title, fontsize=8, color=TEXT_COLOR)
        ax.tick_params(colors=TEXT_DIM, labelsize=7)
        ax.set_facecolor(CARD_BG)
        for spine in ax.spines.values():
            spine.set_color(TEXT_DIM)

    def setup_table(self):
        columns = ("Move #", "Greedy Move", "Greedy Time", "D&C Move", "D&C Time", "DP Move", "DP Time")
        self.tree = ttk.Treeview(self.bottom_frame, columns=columns, show="headings", height=5)
        
        # Styles for Treeview to match dark theme? 
        # Tkinter Treeview theming is notoriously hard to perfect without a theme engine, 
        # but we can try basic config.
        style = ttk.Style()
        style.configure("Treeview", 
                        background=CARD_BG,
                        foreground=TEXT_COLOR,
                        fieldbackground=CARD_BG,
                        rowheight=25)
        style.configure("Treeview.Heading", 
                        background=ACCENT_COLOR, 
                        foreground="black",
                        relief="flat")
        
        for col in columns:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=90, anchor="center")
            
        self.tree.pack(fill=tk.X, expand=True)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(self.bottom_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscroll=scrollbar.set)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

    def update_data(self):
        """
        Refresh graphs and table from game_state.live_analysis_table
        """
        data = self.game_state.live_analysis_table
        if not data:
            return

        # 1. Update Table (Clear and Switch)
        for item in self.tree.get_children():
            self.tree.delete(item)
            
        for row in data:
            values = (
                row.get("move_number"),
                row.get("greedy_move"), row.get("greedy_time"),
                row.get("dnc_move"), row.get("dnc_time"),
                row.get("dp_move"), row.get("dp_time")
            )
            self.tree.insert("", "end", values=values)
            
        # 2. Update Graphs
        moves = [r.get("move_number") for r in data]
        
        # Time
        greedy_times = [self._safe_float(r.get("greedy_time")) for r in data]
        dnc_times = [self._safe_float(r.get("dnc_time")) for r in data]
        dp_times = [self._safe_float(r.get("dp_time")) for r in data]
        
        # States
        greedy_states = [self._safe_int(r.get("greedy_states")) for r in data]
        dnc_states = [self._safe_int(r.get("dnc_states")) for r in data]
        dp_states = [self._safe_int(r.get("dp_states")) for r in data]
        
        # Cumulative Time
        import itertools
        greedy_cum = list(itertools.accumulate(greedy_times))
        dnc_cum = list(itertools.accumulate(dnc_times))
        dp_cum = list(itertools.accumulate(dp_times))
        
        self.ax1.clear()
        self.ax2.clear()
        self.ax3.clear()
        
        # Colors: Use default Matplotlib cycle (C0, C1, C2...)
        c_greedy = "C2" # Green-ish in default cycle usually
        c_dnc = "C3"    # Red-ish
        c_dp = "C0"     # Blue-ish

        
        # Graph 1: Time
        self._plot_line(self.ax1, moves, greedy_times, c_greedy, "Greedy")
        self._plot_line(self.ax1, moves, dnc_times, c_dnc, "D&C")
        self._plot_line(self.ax1, moves, dp_times, c_dp, "DP")
        self.ax1.set_title("Exec Time (ms)", color=TEXT_COLOR, fontsize=8)
        self.ax1.legend(fontsize=6)
        
        # Graph 2: States
        self._plot_line(self.ax2, moves, greedy_states, c_greedy)
        self._plot_line(self.ax2, moves, dnc_states, c_dnc)
        self._plot_line(self.ax2, moves, dp_states, c_dp)
        self.ax2.set_title("States Explored", color=TEXT_COLOR, fontsize=8)

        # Graph 3: Cumulative
        self._plot_line(self.ax3, moves, greedy_cum, c_greedy)
        self._plot_line(self.ax3, moves, dnc_cum, c_dnc)
        self._plot_line(self.ax3, moves, dp_cum, c_dp)
        self.ax3.set_title("Cumulative Time (ms)", color=TEXT_COLOR, fontsize=8)

        # Re-apply styling
        for ax in [self.ax1, self.ax2, self.ax3]:
            ax.tick_params(colors=TEXT_DIM, labelsize=7)
            ax.set_facecolor(CARD_BG)
            for spine in ax.spines.values():
                spine.set_color(TEXT_DIM)
            ax.grid(True, color="#444444", linestyle="--", linewidth=0.5)

        self.canvas.draw()

    def _plot_line(self, ax, x, y, color, label=None):
        ax.plot(x, y, marker='o', markersize=3, color=color, label=label, linewidth=1.5)

    def _safe_float(self, val):
        try:
            return float(val)
        except:
            return 0.0

    def _safe_int(self, val):
        try:
            return int(val)
        except:
            return 0
