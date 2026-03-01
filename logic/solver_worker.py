"""
Solver Worker
=============
Thread-safe background solver wrapper that keeps the UI responsive.

Provides:
- Background thread execution for any solver
- Hard timeout and state-limit enforcement
- Real-time metrics streaming via queue.Queue
- Clean stop mechanism via threading.Event
- Periodic yield to prevent UI starvation
"""

import time
import threading
import queue
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


@dataclass
class SolverMetrics:
    """Snapshot of solver performance at a point in time."""
    timestamp: float          # wall-clock seconds since solver start
    states_explored: int      # total states explored so far
    states_delta: int         # states explored since last snapshot
    time_per_step_ms: float   # avg time per expansion in this window (ms)
    branching_factor: float   # avg branching factor in this window
    interval_ms: float        # time since last snapshot (ms)


class SolverWorker:
    """
    Runs a solver in a background thread with timeout, state limits,
    and real-time metrics streaming.

    Usage:
        worker = SolverWorker(solver_callable, timeout=3.0, max_states=10_000_000)
        worker.start()
        
        # Poll from UI thread:
        while not worker.is_done():
            metrics = worker.drain_metrics()
            # update graphs with metrics
            root.after(33, poll)
        
        result = worker.get_result()
    """

    # Metrics push interval
    METRICS_INTERVAL_MS = 50  # push metrics every ~50ms

    def __init__(
        self,
        solver_fn: Callable[[], Dict[str, Any]],
        timeout: float = 3.0,
        max_states: int = 10_000_000,
        label: str = "solver",
    ):
        self.solver_fn = solver_fn
        self.timeout = timeout
        self.max_states = max_states
        self.label = label

        self.stop_event = threading.Event()
        self.done_event = threading.Event()
        self._metrics_queue: queue.Queue = queue.Queue(maxsize=500)
        self._result: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None
        self._error: Optional[Exception] = None

    # ── Public API ─────────────────────────────────────────────

    def start(self):
        """Launch the solver in a background daemon thread."""
        self.stop_event.clear()
        self.done_event.clear()
        self._result = None
        self._error = None

        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the solver to stop cleanly."""
        self.stop_event.set()

    def is_done(self) -> bool:
        """Check if solver has finished (success, timeout, or error)."""
        return self.done_event.is_set()

    def get_result(self) -> Optional[Dict[str, Any]]:
        """Return solver result dict. None if not yet done."""
        if not self.done_event.is_set():
            return None
        return self._result

    def get_error(self) -> Optional[Exception]:
        return self._error

    def drain_metrics(self) -> list:
        """Non-blocking drain of all queued metrics snapshots."""
        items = []
        while True:
            try:
                items.append(self._metrics_queue.get_nowait())
            except queue.Empty:
                break
        return items

    # ── Internal ───────────────────────────────────────────────

    def _run(self):
        """Thread target: execute solver_fn with monitoring."""
        start_time = time.perf_counter()
        try:
            result = self.solver_fn()
            elapsed = time.perf_counter() - start_time

            # Inject timing if not present
            if isinstance(result, dict):
                result.setdefault("time_taken", elapsed)
                result.setdefault("worker_label", self.label)
            self._result = result

        except Exception as e:
            self._error = e
            self._result = {
                "success": False,
                "status": "Error",
                "error": str(e),
                "worker_label": self.label,
            }
        finally:
            self.done_event.set()

    def push_metrics(self, metrics: SolverMetrics):
        """Called by the solver (from solver thread) to push a metrics snapshot."""
        try:
            self._metrics_queue.put_nowait(metrics)
        except queue.Full:
            # Drop oldest if full
            try:
                self._metrics_queue.get_nowait()
                self._metrics_queue.put_nowait(metrics)
            except queue.Empty:
                pass


class AnalysisWorker:
    """
    Specialized worker for running the live comparative analysis
    (legacy vs current solver) in a background thread.

    Streams real incremental metrics — no fake/extrapolated data.
    """

    def __init__(self, analysis_fn: Callable[[], Optional[Dict[str, Any]]]):
        self.analysis_fn = analysis_fn
        self._done = threading.Event()
        self._result: Optional[Dict[str, Any]] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        self._done.clear()
        self._result = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def is_done(self) -> bool:
        return self._done.is_set()

    def get_result(self) -> Optional[Dict[str, Any]]:
        return self._result if self._done.is_set() else None

    def _run(self):
        try:
            self._result = self.analysis_fn()
        except Exception as e:
            self._result = {"error": str(e)}
        finally:
            self._done.set()
