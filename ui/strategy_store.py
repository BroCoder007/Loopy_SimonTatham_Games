"""
Strategy Store (UI-only)
========================
Global storage for the user's selected solving strategy.

CRITICAL:
- UI wiring only; gameplay/solver behavior must remain unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class StrategyStore:
    selected_strategy: str = "greedy"
    
    def get_strategy(self):
        """Get the current strategy."""
        return self.selected_strategy
    
    def set_strategy(self, strategy):
        """Set the strategy."""
        self.selected_strategy = strategy


# Single global instance
strategy_store = StrategyStore()

