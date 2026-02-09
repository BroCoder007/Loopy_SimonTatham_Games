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
        self.stats = {
            "games_played": 0,
            "wins_vs_cpu": 0,
            "losses": 0,
            "fastest_win": None
        }
        self.load_stats()

    def load_stats(self):
        if os.path.exists(STATS_FILE):
            try:
                with open(STATS_FILE, 'r') as f:
                    self.stats = json.load(f)
            except:
                pass # Keep defaults if error

    def save_stats(self):
        os.makedirs("data", exist_ok=True)
        with open(STATS_FILE, 'w') as f:
            json.dump(self.stats, f, indent=4)

    def record_game(self, winner, time_taken=None):
        self.stats["games_played"] += 1
        if winner == "Player 1 (Human)":
            self.stats["wins_vs_cpu"] += 1
            if time_taken:
                current_fastest = self.stats.get("fastest_win")
                if current_fastest is None or time_taken < current_fastest:
                    self.stats["fastest_win"] = time_taken
        elif winner == "Player 2 (CPU)":
            self.stats["losses"] += 1
        self.save_stats()
