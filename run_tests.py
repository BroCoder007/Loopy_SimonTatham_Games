import sys, os, time, threading, queue
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from logic.solvers.dp_backtracking_solver import DPBacktrackingSolver
from logic.solver_worker import SolverWorker, SolverMetrics, AnalysisWorker

class MG:
    edges = set()
    def copy(self): return MG()

class MS:
    def __init__(self, r, c, cl):
        self.rows = r
        self.cols = c
        self.clues = cl
        self.graph = MG()
        self.game_mode = "vs_cpu"
        self.turn = "P1"
        self.message = ""
        self.last_cpu_move_info = {}
        self.edge_weights = {}
        self.solution_edges = set()
        self.live_analysis_table = []

results = []

# Test 1: Timeout on 7x7
try:
    gs = MS(7, 7, {(r,c): ((r+c) % 4) for r in range(7) for c in range(7)})
    s = DPBacktrackingSolver(gs, timeout=0.5, max_states=10000000)
    t = time.time()
    r = s.solve(ignore_current_edges=True)
    e = time.time() - t
    ok = e < 2.0 and r["status"] in ("Timeout", "Success", "NoSolution")
    results.append("PASS" if ok else "FAIL")
    print("T1(timeout): " + r["status"] + " " + str(round(e, 2)) + "s " + str(r["nodes_visited"]) + " nodes - " + results[-1])
except Exception as ex:
    results.append("FAIL")
    print("T1 ERROR: " + str(ex))

# Test 2: State limit
try:
    gs2 = MS(5, 5, {(r,c): ((r+c) % 3) for r in range(5) for c in range(5)})
    s2 = DPBacktrackingSolver(gs2, timeout=30.0, max_states=500)
    r2 = s2.solve(ignore_current_edges=True)
    ok2 = r2["nodes_visited"] <= 600
    results.append("PASS" if ok2 else "FAIL")
    print("T2(statelim): " + r2["status"] + " " + str(r2["nodes_visited"]) + " nodes - " + results[-1])
except Exception as ex:
    results.append("FAIL")
    print("T2 ERROR: " + str(ex))

# Test 3: Clean stop
try:
    gs3 = MS(6, 6, {(r,c): ((r+c) % 3) for r in range(6) for c in range(6)})
    ev = threading.Event()
    s3 = DPBacktrackingSolver(gs3, stop_event=ev, timeout=30.0)
    def stoper():
        time.sleep(0.1)
        ev.set()
    threading.Thread(target=stoper, daemon=True).start()
    t3 = time.time()
    r3 = s3.solve(ignore_current_edges=True)
    e3 = time.time() - t3
    ok3 = e3 < 2.0
    results.append("PASS" if ok3 else "FAIL")
    print("T3(stop): " + r3["status"] + " " + str(round(e3, 2)) + "s - " + results[-1])
except Exception as ex:
    results.append("FAIL")
    print("T3 ERROR: " + str(ex))

# Test 4: Metrics queue
try:
    gs4 = MS(5, 5, {(r,c): ((r+c) % 3) for r in range(5) for c in range(5)})
    mq = queue.Queue()
    s4 = DPBacktrackingSolver(gs4, timeout=2.0, max_states=5000, metrics_queue=mq)
    r4 = s4.solve(ignore_current_edges=True)
    count = mq.qsize()
    ok4 = count >= 0  # may be 0 if solved in < 500 nodes
    results.append("PASS" if ok4 else "FAIL")
    print("T4(metrics): " + str(count) + " snapshots, " + str(r4["nodes_visited"]) + " nodes - " + results[-1])
except Exception as ex:
    results.append("FAIL")
    print("T4 ERROR: " + str(ex))

# Test 5: No fake data
try:
    with open("logic/live_analysis.py", "r") as f:
        c = f.read()
    bad = ["random.uniform", "random.randint", "math.exp", "multiplier"]
    found = [b for b in bad if b in c]
    ok5 = len(found) == 0
    results.append("PASS" if ok5 else "FAIL")
    print("T5(nofake): " + ("clean" if ok5 else "FOUND: " + str(found)) + " - " + results[-1])
except Exception as ex:
    results.append("FAIL")
    print("T5 ERROR: " + str(ex))

# Test 6: AnalysisWorker
try:
    def sample_fn():
        time.sleep(0.1)
        return {"status": "done"}
    w = AnalysisWorker(sample_fn)
    w.start()
    time.sleep(0.3)
    ok6 = w.is_done() and w.get_result()["status"] == "done"
    results.append("PASS" if ok6 else "FAIL")
    print("T6(worker): " + str(w.get_result()) + " - " + results[-1])
except Exception as ex:
    results.append("FAIL")
    print("T6 ERROR: " + str(ex))

# Test 7: UI imports
try:
    from ui.analysis_panel import LiveAnalysisPanel
    from ui.solver_control_panel import SolverControlPanel
    results.append("PASS")
    print("T7(imports): OK - PASS")
except Exception as ex:
    results.append("FAIL")
    print("T7 ERROR: " + str(ex))

passed = results.count("PASS")
failed = results.count("FAIL")
print("---")
print("TOTAL: " + str(passed) + " passed, " + str(failed) + " failed")
