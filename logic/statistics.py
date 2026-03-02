"""
Statistics Manager
==================
Handles loading and saving game statistics to a JSON file.
"""

import json
import os

STATS_FILE = "data/statistics.json"

class StatisticsManager:
    def __init__(self):
        # Maintain per-game entries for richer analytics
        self.stats = {
            "games_played": 0,
            "wins_vs_cpu": 0,
            "losses": 0,
            "fastest_win": None,
            "fastest_win_by_difficulty": {},
            "games": []  # list of {ts,winner,time,difficulty,moves}
        }
        self.load_stats()

    def load_stats(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    data = json.load(f)
                    # Backwards-compat: if old format exists merge defaults
                    if isinstance(data, dict):
                        self.stats.update(data)
            except Exception:
                pass # Keep defaults if error

    def save_stats(self):
        os.makedirs("data", exist_ok=True)
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def record_game(self, winner, time_taken=None, difficulty=None, moves=None):
        """
        Record a finished game. Backwards-compatible: callers may pass only `winner`.
        Stores a per-game entry and updates summary counters.
        """
        import time

        self.stats["games_played"] = self.stats.get("games_played", 0) + 1

        if winner == "Player 1 (Human)":
            self.stats["wins_vs_cpu"] = self.stats.get("wins_vs_cpu", 0) + 1
        elif winner == "Player 2 (CPU)":
            self.stats["losses"] = self.stats.get("losses", 0) + 1

        # Per-game entry
        entry = {
            "ts": time.time(),
            "winner": winner,
            "time": time_taken,
            "difficulty": difficulty,
            "moves": moves,
        }
        self.stats.setdefault("games", []).append(entry)

        # Fastest win overall and by difficulty (only consider human wins with time)
        if winner == "Player 1 (Human)" and time_taken is not None:
            current_fastest = self.stats.get("fastest_win")
            if current_fastest is None or time_taken < current_fastest:
                self.stats["fastest_win"] = time_taken

            if difficulty:
                fb = self.stats.setdefault("fastest_win_by_difficulty", {})
                cur = fb.get(difficulty)
                if cur is None or time_taken < cur:
                    fb[difficulty] = time_taken

        # Persist
        self.save_stats()
