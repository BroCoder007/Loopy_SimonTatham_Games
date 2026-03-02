"""
Execution Trace Module
=====================
Unified research visualization layer for structured execution trace.

This module provides a global trace object that captures detailed execution
information from all solver strategies without modifying their core logic.

TRACE STRUCTURE:
execution_trace = [
  {
    step_id: int,
    strategy_name: str,
    move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]],
    explanation: str,
    recursion_depth: int,
    region_id: Optional[str],
    dp_state_count: Optional[int],
    boundary_state_info: Optional[str],
    timestamp: float
  }
]
"""

import time
import threading
from typing import Any, Dict, List, Optional, Tuple, Union

# Global storage for execution trace
# Structure: [ { step_id, strategy_name, move, explanation, recursion_depth, ... }, ... ]
execution_trace: List[Dict[str, Any]] = []

# Global counter for steps
_step_counter = 0
_trace_lock = threading.Lock()

def log_execution_step(
    strategy_name: str,
    move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None,
    explanation: str = "",
    recursion_depth: int = 0,
    region_id: Optional[str] = None,
    dp_state_count: Optional[int] = None,
    boundary_state_info: Optional[str] = None,
    merge_info: Optional[str] = None
):
    """
    Logs a single execution step (move or internal decision) to the global trace.

    Args:
        strategy_name: Name of the solver/generator (e.g., "Greedy", "AdvancedDPSolver").
        move: The edge being acted upon (optional for internal steps). Format: ((r1, c1), (r2, c2)).
        explanation: Human-readable description of the step.
        recursion_depth: Current depth of recursion (0 for flat solvers).
        region_id: Identifier for the current region (e.g., "0-5,0-5").
        dp_state_count: Number of DP states computed/considered (for Pure DP).
        boundary_state_info: Description of boundary profiles (for Advanced DP).
        merge_info: Description of merge operations (for Generator/Advanced DP).
    """
    global _step_counter
    
    # Use merge_info as boundary_state_info if provided
    if merge_info and not boundary_state_info:
        boundary_state_info = merge_info
    
    entry = {
        "step_id": _step_counter,
        "strategy_name": strategy_name,
        "move": move,
        "explanation": explanation,
        "recursion_depth": recursion_depth,
        "region_id": region_id,
        "dp_state_count": dp_state_count,
        "boundary_state_info": boundary_state_info,
        "merge_info": merge_info, # Helper for specific reqs
        "timestamp": time.time()
    }
    
    with _trace_lock:
        execution_trace.append(entry)
        _step_counter += 1

def clear_trace():
    """Clears the global execution trace."""
    global execution_trace, _step_counter
    with _trace_lock:
        execution_trace.clear()
        _step_counter = 0

def get_trace() -> List[Dict[str, Any]]:
    """Returns a copy of the current execution trace."""
    with _trace_lock:
        return execution_trace.copy()

def get_trace_summary() -> Dict[str, Any]:
    """
    Get a summary of the execution trace.
    
    Returns:
        Dictionary with summary statistics
    """
    with _trace_lock:
        if not execution_trace:
            return {
                "total_steps": 0,
                "strategies": {},
                "max_recursion_depth": 0,
                "total_dp_states": 0
            }
        
        strategy_counts = {}
        max_depth = 0
        total_dp_states = 0
        
        for entry in execution_trace:
            strategy = entry["strategy_name"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1
            max_depth = max(max_depth, entry["recursion_depth"])
            if entry["dp_state_count"]:
                total_dp_states += entry["dp_state_count"]
        
        return {
            "total_steps": len(execution_trace),
            "strategies": strategy_counts,
            "max_recursion_depth": max_depth,
            "total_dp_states": total_dp_states,
            "time_span": execution_trace[-1]["timestamp"] - execution_trace[0]["timestamp"] if len(execution_trace) > 1 else 0
        }

def get_strategy_trace(strategy_name: str) -> List[Dict[str, Any]]:
    """
    Get trace entries for a specific strategy.
    
    Args:
        strategy_name: Name of the strategy to filter by
        
    Returns:
        List of trace entries for the specified strategy
    """
    with _trace_lock:
        return [entry for entry in execution_trace if entry["strategy_name"] == strategy_name]

# Strategy-specific helper functions for cleaner integration

def log_greedy_move(move: Tuple[Tuple[int, int], Tuple[int, int]], explanation: str) -> None:
    """
    Log a greedy solver move.
    
    Args:
        move: The edge move being made
        explanation: Explanation of the greedy decision
    """
    log_execution_step(
        strategy_name="Greedy",
        move=move,
        explanation=explanation,
        recursion_depth=0,
        region_id=None
    )

def log_pure_dp_move(move: Tuple[Tuple[int, int], Tuple[int, int]], explanation: str, dp_state_count: int) -> None:
    """
    Log a pure DP solver move.
    
    Args:
        move: The edge move being made
        explanation: Explanation of the DP decision
        dp_state_count: Number of DP states explored
    """
    log_execution_step(
        strategy_name="Pure DP",
        move=move,
        explanation=explanation,
        recursion_depth=0,
        dp_state_count=dp_state_count
    )

def log_advanced_dp_move(move: Tuple[Tuple[int, int], Tuple[int, int]], explanation: str, 
                        recursion_depth: int, region_id: str, boundary_state_info: str = "") -> None:
    """
    Log an advanced DP solver move.
    
    Args:
        move: The edge move being made
        explanation: Explanation of the advanced DP decision
        recursion_depth: Current recursion depth
        region_id: Current region identifier
        boundary_state_info: Information about boundary states
    """
    log_execution_step(
        strategy_name="Advanced DP",
        move=move,
        explanation=explanation,
        recursion_depth=recursion_depth,
        region_id=region_id,
        boundary_state_info=boundary_state_info
    )

def log_generator_step(explanation: str, recursion_depth: int, region_id: str, 
                      merge_info: str = "", move: Optional[Tuple[Tuple[int, int], Tuple[int, int]]] = None) -> None:
    """
    Log a puzzle generator step.
    
    Args:
        explanation: Explanation of the generator step
        recursion_depth: Current recursion depth
        region_id: Current region identifier
        merge_info: Information about merge operations
        move: Optional move being made
    """
    log_execution_step(
        strategy_name="DnC Generator",
        move=move,
        explanation=explanation,
        recursion_depth=recursion_depth,
        region_id=region_id,
        merge_info=merge_info
    )
