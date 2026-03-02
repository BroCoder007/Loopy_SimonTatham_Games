"""
Statistics Dashboard UI
======================
Provides a `StatisticsDashboard` Frame that renders summary stats and simple charts.
Falls back to textual rendering if `matplotlib` is not available.
"""

import tkinter as tk
from ui.components import CardFrame, HoverButton
from ui.styles import *
from logic.statistics import StatisticsManager

try:
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    MATPLOTLIB_AVAILABLE = True
except Exception:
    MATPLOTLIB_AVAILABLE = False

class StatisticsDashboard(tk.Frame):
    def __init__(self, master, **kwargs):
        super().__init__(master, bg=BG_COLOR, **kwargs)
        self.stats_mgr = StatisticsManager()
        self._build_ui()
        self.update_data()

    def _build_ui(self):
        header = tk.Frame(self, bg=BG_COLOR)
        header.pack(fill=tk.X, pady=(20, 10))
        tk.Label(header, text="Player Statistics", font=FONT_TITLE, bg=BG_COLOR, fg=TEXT_COLOR).pack(side=tk.LEFT, padx=20)
        HoverButton(header, text="Refresh", command=self.update_data).pack(side=tk.RIGHT, padx=20)

        self.container = tk.Frame(self, bg=BG_COLOR)
        self.container.pack(expand=True, fill=tk.BOTH, padx=20, pady=10)

        # Summary card
        self.summary_card = CardFrame(self.container, padx=20, pady=20)
        self.summary_card.pack(fill=tk.X, pady=(0, 10))

        self.lbl_games = tk.Label(self.summary_card, text="Games Played: 0", font=FONT_BODY, bg=CARD_BG, fg=TEXT_COLOR)
        self.lbl_games.pack(anchor="w")
        self.lbl_wins = tk.Label(self.summary_card, text="Wins: 0", font=FONT_BODY, bg=CARD_BG, fg=APPLE_GREEN)
        self.lbl_wins.pack(anchor="w")
        self.lbl_losses = tk.Label(self.summary_card, text="Losses: 0", font=FONT_BODY, bg=CARD_BG, fg=APPLE_RED)
        self.lbl_losses.pack(anchor="w")
        self.lbl_fastest = tk.Label(self.summary_card, text="Fastest Win: N/A", font=FONT_BODY, bg=CARD_BG, fg=TEXT_DIM)
        self.lbl_fastest.pack(anchor="w")

        # Charts area
        self.charts_card = CardFrame(self.container, padx=10, pady=10)
        self.charts_card.pack(expand=True, fill=tk.BOTH)

        if MATPLOTLIB_AVAILABLE:
            self.fig = Figure(figsize=(6, 3), dpi=100)
            self.ax1 = self.fig.add_subplot(131)
            self.ax2 = self.fig.add_subplot(132)
            self.ax3 = self.fig.add_subplot(133)
            self.canvas = FigureCanvasTkAgg(self.fig, master=self.charts_card)
            self.canvas.get_tk_widget().pack(expand=True, fill=tk.BOTH)
        else:
            self.plain_text = tk.Text(self.charts_card, height=15, bg=CARD_BG, fg=TEXT_DIM)
            self.plain_text.pack(expand=True, fill=tk.BOTH)

        # Recent games list
        self.recent_card = CardFrame(self.container, padx=10, pady=10)
        self.recent_card.pack(fill=tk.X, pady=(10, 0))
        tk.Label(self.recent_card, text="Recent Games", font=FONT_BODY, bg=CARD_BG, fg=TEXT_COLOR).pack(anchor="w")
        self.lst_recent = tk.Listbox(self.recent_card, height=6, bg=CARD_BG, fg=TEXT_DIM, borderwidth=0, highlightthickness=0)
        self.lst_recent.pack(fill=tk.X, pady=(8,0))

    def update_data(self):
        s = self.stats_mgr.stats
        games = s.get("games", [])

        self.lbl_games.config(text=f"Games Played: {s.get('games_played', 0)}")
        self.lbl_wins.config(text=f"Wins: {s.get('wins_vs_cpu', 0)}")
        self.lbl_losses.config(text=f"Losses: {s.get('losses', 0)}")
        fastest = s.get('fastest_win')
        self.lbl_fastest.config(text=f"Fastest Win: {fastest:.2f}s" if fastest else "Fastest Win: N/A")

        # Recent list
        self.lst_recent.delete(0, tk.END)
        recent = list(reversed(games))[:20]
        for g in recent:
            ts = g.get('ts')
            import time
            timestr = time.strftime('%Y-%m-%d %H:%M', time.localtime(ts)) if ts else 'N/A'
            t = g.get('time')
            diff = g.get('difficulty') or 'unknown'
            mv = g.get('moves')
            winner = g.get('winner')
            txt = f"{timestr} | {diff} | {winner} | time: {t if t is not None else 'N/A'}s | moves: {mv if mv is not None else 'N/A'}"
            self.lst_recent.insert(tk.END, txt)

        if MATPLOTLIB_AVAILABLE:
            self._draw_charts(games)
        else:
            self._draw_text_fallback(games)

    def _draw_text_fallback(self, games):
        self.plain_text.delete('1.0', tk.END)
        # Provide simple aggregates
        by_diff = {}
        for g in games:
            d = g.get('difficulty') or 'unknown'
            by_diff.setdefault(d, []).append(g)

        lines = []
        lines.append('Win rate by difficulty:')
        for d, lst in by_diff.items():
            played = len(lst)
            wins = sum(1 for x in lst if x.get('winner') == 'Player 1 (Human)')
            rate = f"{wins}/{played} ({wins/played*100:.1f}% )" if played else 'N/A'
            lines.append(f" - {d}: {rate}")

        lines.append('\nAverage solve times (s) by difficulty:')
        for d, lst in by_diff.items():
            times = [x.get('time') for x in lst if x.get('time') is not None]
            if times:
                avg = sum(times)/len(times)
                lines.append(f" - {d}: {avg:.2f}s ({len(times)} samples)")
            else:
                lines.append(f" - {d}: N/A")

        self.plain_text.insert(tk.END, "\n".join(lines))

    def _draw_charts(self, games):
        # Clear axes
        self.ax1.clear(); self.ax2.clear(); self.ax3.clear()

        # 1) Win rate over time (cumulative)
        times = []
        cum_games = []
        cum_wins = []
        wins = 0
        for i, g in enumerate(games):
            times.append(i)
            if g.get('winner') == 'Player 1 (Human)':
                wins += 1
            cum_games.append(i+1)
            cum_wins.append(wins)
        if cum_games:
            rates = [w/g for w, g in zip(cum_wins, cum_games)]
            self.ax1.plot(rates, marker='o')
            self.ax1.set_title('Win Rate (cumulative)')
            self.ax1.set_ylim(0, 1)
        else:
            self.ax1.text(0.5, 0.5, 'No data', ha='center')

        # 2) Fastest win per difficulty (bar)
        fb = self.stats_mgr.stats.get('fastest_win_by_difficulty', {})
        labels = list(fb.keys())
        values = [fb[k] for k in labels]
        if labels:
            self.ax2.bar(labels, values, color='tab:green')
            self.ax2.set_title('Fastest Win by Difficulty (s)')
            self.ax2.tick_params(axis='x', rotation=45)
        else:
            self.ax2.text(0.5, 0.5, 'No data', ha='center')

        # 3) Average moves per difficulty
        by_diff = {}
        for g in games:
            d = g.get('difficulty') or 'unknown'
            by_diff.setdefault(d, []).append(g)
        labels2 = []
        avg_moves = []
        for d, lst in by_diff.items():
            moves = [x.get('moves') for x in lst if x.get('moves') is not None]
            if moves:
                labels2.append(d)
                avg_moves.append(sum(moves)/len(moves))
        if labels2:
            self.ax3.bar(labels2, avg_moves, color='tab:orange')
            self.ax3.set_title('Avg Moves by Difficulty')
            self.ax3.tick_params(axis='x', rotation=45)
        else:
            self.ax3.text(0.5, 0.5, 'No data', ha='center')

        self.fig.tight_layout()
        try:
            self.canvas.draw()
        except Exception:
            pass
