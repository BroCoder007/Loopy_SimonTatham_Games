"""
Solver safety errors and limits.
"""

from __future__ import annotations

import os
from typing import Any

DEFAULT_SAFE_LIMIT = 200000
STATE_SPACE_EXPLOSION_MESSAGE = "State space explosion — theoretical exponential behavior demonstrated."


class ControlledStateExplosionError(RuntimeError):
    """
    Raised when a solver crosses a configured safety bound for state/config growth.
    """

    def __init__(
        self,
        message: str = STATE_SPACE_EXPLOSION_MESSAGE,
        *,
        safe_limit: int | None = None,
        observed: int | None = None,
        context: str | None = None,
    ) -> None:
        super().__init__(message)
        self.safe_limit = safe_limit
        self.observed = observed
        self.context = context


def resolve_safe_limit(game_state: Any, solver_attr_name: str) -> int:
    """
    Resolve SAFE_LIMIT with per-solver override support.

    Priority:
    1) game_state.<solver_attr_name>
    2) game_state.state_space_safe_limit
    3) env LOOPY_SAFE_LIMIT
    4) DEFAULT_SAFE_LIMIT
    """
    raw = getattr(game_state, solver_attr_name, None)
    if raw is None:
        raw = getattr(game_state, "state_space_safe_limit", None)
    if raw is None:
        raw = os.getenv("LOOPY_SAFE_LIMIT")
    if raw is None:
        return DEFAULT_SAFE_LIMIT

    try:
        limit = int(raw)
        if limit > 0:
            return limit
    except (TypeError, ValueError):
        pass
    return DEFAULT_SAFE_LIMIT

