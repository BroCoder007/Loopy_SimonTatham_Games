import os
from logic.statistics import StatisticsManager

def test_record_game_appends_entry():
    mgr = StatisticsManager()
    before = len(mgr.stats.get('games', []))
    mgr.record_game('Player 1 (Human)', time_taken=1.23, difficulty='Easy', moves=10)
    mgr2 = StatisticsManager()
    after = len(mgr2.stats.get('games', []))
    assert after >= before + 1
