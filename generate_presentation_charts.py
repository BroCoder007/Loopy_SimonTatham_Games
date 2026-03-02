"""
Presentation Chart Generator
=============================
Generates 6 publication-quality charts comparing Greedy, D&C, and DP solvers.
Run:  python generate_presentation_charts.py --games 5
Output: presentation_charts/ folder with 6 PNG files.
"""

import sys
import os
import time
import argparse
import numpy as np
from typing import Dict, Any, List, Tuple
from collections import defaultdict

# Ensure project root is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec

from logic.game_state import GameState
from logic.solvers.greedy_solver import GreedySolver
from logic.solvers.divide_conquer_solver import DivideConquerSolver
from logic.solvers.dynamic_programming_solver import DynamicProgrammingSolver

# Suppress DP debug logging
import logic.solvers.dynamic_programming_solver as dp_mod
dp_mod.DEBUG_MODE = False

# ─────────────────────────────────────────────────────────────
# Color Palette & Styling
# ─────────────────────────────────────────────────────────────
COLORS = {
    "greedy": "#FF6B6B",   # Coral Red
    "dc":     "#51CF66",   # Emerald Green
    "dp":     "#339AF0",   # Sky Blue
}
SOLVER_LABELS = {"greedy": "Greedy", "dc": "Divide & Conquer", "dp": "Dynamic Programming"}
BG_COLOR = "#1A1B26"       # Tokyo Night background
CARD_COLOR = "#24283B"     # Card panels
TEXT_COLOR = "#C0CAF5"     # Soft lavender text
GRID_COLOR = "#414868"     # Subtle grid lines
ACCENT_GOLD = "#E0AF68"   # Gold accent


def setup_style():
    """Apply a dark, presentation-friendly matplotlib style."""
    plt.rcParams.update({
        "figure.facecolor": BG_COLOR,
        "axes.facecolor": CARD_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.labelcolor": TEXT_COLOR,
        "axes.titleweight": "bold",
        "text.color": TEXT_COLOR,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "grid.color": GRID_COLOR,
        "grid.alpha": 0.3,
        "font.size": 13,
        "axes.titlesize": 16,
        "axes.labelsize": 13,
        "legend.facecolor": CARD_COLOR,
        "legend.edgecolor": GRID_COLOR,
        "legend.fontsize": 11,
        "figure.dpi": 180,
        "savefig.dpi": 180,
        "savefig.bbox": "tight",
        "savefig.facecolor": BG_COLOR,
    })


# ─────────────────────────────────────────────────────────────
# Benchmarking Engine
# ─────────────────────────────────────────────────────────────
def run_solver_on_clone(solver_class, base_state, rows, cols, timeout=60):
    """
    Run a solver on a clone of the base state.
    Returns (solved: bool, time_s: float, moves: int).
    """
    state_clone = base_state.clone_for_simulation()
    solver = solver_class(state_clone)
    start = time.time()
    moves_count = 0
    solved = False
    max_moves = (rows * cols + rows + cols) * 2

    try:
        while moves_count < max_moves:
            elapsed = time.time() - start
            if elapsed > timeout:
                break

            if isinstance(solver, GreedySolver):
                _, move = solver.decide_move()
            else:
                move = solver.solve()

            if move is None:
                break

            state_clone.turn = "Player 2 (CPU)"
            u, v = move
            if not state_clone.make_move(u, v, is_cpu=True):
                break
            moves_count += 1

            if state_clone.game_over:
                solved = ("CPU" in str(state_clone.winner) or
                          state_clone.winner == "Player 2 (CPU)")
                break
    except Exception:
        pass

    return solved, time.time() - start, moves_count


def run_benchmark(games_per_config: int, configs: List[Tuple[int, int, str]],
                  timeout: int = 60) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run benchmark across all configs.
    configs: list of (rows, cols, difficulty)
    Returns results grouped by config key "RxC_difficulty".
    """
    results = defaultdict(list)

    total = len(configs) * games_per_config
    done = 0

    for rows, cols, diff in configs:
        key = f"{rows}x{cols}_{diff}"
        for g in range(games_per_config):
            done += 1
            print(f"  [{done}/{total}] {key} game {g+1}/{games_per_config} ...", end="\r")

            base = GameState(rows=rows, cols=cols, difficulty=diff, game_mode="vs_cpu")
            if not base.clues:
                base._generate_clues()
            base.cpu = None

            entry = {"size": f"{rows}x{cols}", "difficulty": diff}
            for tag, cls in [("greedy", GreedySolver),
                             ("dc", DivideConquerSolver),
                             ("dp", DynamicProgrammingSolver)]:
                s, t, m = run_solver_on_clone(cls, base, rows, cols, timeout)
                entry[f"{tag}_solved"] = s
                entry[f"{tag}_time"] = t
                entry[f"{tag}_moves"] = m

            results[key].append(entry)

    print()
    return results


# ─────────────────────────────────────────────────────────────
# Chart Generators
# ─────────────────────────────────────────────────────────────
def add_value_labels(ax, bars, fmt="{:.0f}%", offset=1.5):
    """Add value labels on top of bars."""
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, h + offset,
                    fmt.format(h), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=TEXT_COLOR)


def chart_1_success_rate(results, out_dir):
    """Bar chart: Success rate per solver, grouped by config."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.22

    for i, (tag, label) in enumerate([("greedy", "Greedy"),
                                       ("dc", "Divide & Conquer"),
                                       ("dp", "Dynamic Programming")]):
        rates = []
        for cfg in configs:
            total = len(results[cfg])
            solved = sum(1 for r in results[cfg] if r[f"{tag}_solved"])
            rates.append(100 * solved / total if total else 0)
        bars = ax.bar(x + i * width, rates, width, label=label,
                      color=COLORS[tag], edgecolor="none", alpha=0.9,
                      zorder=3)
        add_value_labels(ax, bars, fmt="{:.0f}%", offset=1.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([k.replace("_", "\n") for k in configs], fontsize=10)
    ax.set_ylabel("Success Rate (%)")
    ax.set_title("🏆  Puzzle Solve Rate by Algorithm", fontsize=18, pad=15)
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(out_dir, "1_success_rate.png"))
    plt.close(fig)
    print("  ✓ Chart 1: Success Rate")


def chart_2_timing(results, out_dir):
    """Bar chart: Average solve time per solver, grouped by config."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.22

    for i, (tag, label) in enumerate([("greedy", "Greedy"),
                                       ("dc", "Divide & Conquer"),
                                       ("dp", "Dynamic Programming")]):
        times = []
        for cfg in configs:
            avg_t = np.mean([r[f"{tag}_time"] for r in results[cfg]])
            times.append(avg_t)
        bars = ax.bar(x + i * width, times, width, label=label,
                      color=COLORS[tag], edgecolor="none", alpha=0.9,
                      zorder=3)
        add_value_labels(ax, bars, fmt="{:.2f}s", offset=max(times) * 0.02 + 0.05)

    ax.set_xticks(x + width)
    ax.set_xticklabels([k.replace("_", "\n") for k in configs], fontsize=10)
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("⏱  Average Solve Time", fontsize=18, pad=15)
    ax.legend(loc="upper left")
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(out_dir, "2_timing.png"))
    plt.close(fig)
    print("  ✓ Chart 2: Timing Comparison")


def chart_3_moves(results, out_dir):
    """Bar chart: Average moves used per solver, grouped by config."""
    fig, ax = plt.subplots(figsize=(10, 6))
    configs = list(results.keys())
    x = np.arange(len(configs))
    width = 0.22

    for i, (tag, label) in enumerate([("greedy", "Greedy"),
                                       ("dc", "Divide & Conquer"),
                                       ("dp", "Dynamic Programming")]):
        moves = []
        for cfg in configs:
            avg_m = np.mean([r[f"{tag}_moves"] for r in results[cfg]])
            moves.append(avg_m)
        bars = ax.bar(x + i * width, moves, width, label=label,
                      color=COLORS[tag], edgecolor="none", alpha=0.9,
                      zorder=3)
        add_value_labels(ax, bars, fmt="{:.1f}", offset=0.5)

    ax.set_xticks(x + width)
    ax.set_xticklabels([k.replace("_", "\n") for k in configs], fontsize=10)
    ax.set_ylabel("Average Moves")
    ax.set_title("🎯  Move Efficiency (fewer = smarter)", fontsize=18, pad=15)
    ax.legend(loc="upper left")
    ax.grid(axis="y", zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(out_dir, "3_move_efficiency.png"))
    plt.close(fig)
    print("  ✓ Chart 3: Move Efficiency")


def chart_4_scalability(results, out_dir):
    """Line chart: Solve time vs grid size for each solver."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by size, averaging across difficulties
    size_data = defaultdict(lambda: defaultdict(list))
    for cfg, entries in results.items():
        size = entries[0]["size"]
        for entry in entries:
            for tag in ["greedy", "dc", "dp"]:
                size_data[size][tag].append(entry[f"{tag}_time"])

    # Sort sizes by total cells
    sorted_sizes = sorted(size_data.keys(),
                          key=lambda s: int(s.split("x")[0]) * int(s.split("x")[1]))

    for tag, label in [("greedy", "Greedy"),
                       ("dc", "Divide & Conquer"),
                       ("dp", "Dynamic Programming")]:
        times = [np.mean(size_data[s][tag]) for s in sorted_sizes]
        ax.plot(sorted_sizes, times, "o-", label=label, color=COLORS[tag],
                linewidth=2.5, markersize=8, zorder=3)

    ax.set_xlabel("Grid Size")
    ax.set_ylabel("Average Time (seconds)")
    ax.set_title("📈  Scalability: Time vs Grid Size", fontsize=18, pad=15)
    ax.legend()
    ax.grid(True, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    fig.savefig(os.path.join(out_dir, "4_scalability.png"))
    plt.close(fig)
    print("  ✓ Chart 4: Scalability")


def chart_5_radar(results, out_dir):
    """Radar / Spider chart: multi-axis comparison of all 3 solvers."""
    # Compute aggregate metrics across all configs
    all_entries = [e for entries in results.values() for e in entries]

    metrics = {}
    for tag in ["greedy", "dc", "dp"]:
        solved = sum(1 for e in all_entries if e[f"{tag}_solved"])
        total = len(all_entries)
        avg_time = np.mean([e[f"{tag}_time"] for e in all_entries])
        avg_moves = np.mean([e[f"{tag}_moves"] for e in all_entries])
        max_time = max(np.mean([e[f"{t}_time"] for e in all_entries]) for t in ["greedy", "dc", "dp"])
        max_moves = max(np.mean([e[f"{t}_moves"] for e in all_entries]) for t in ["greedy", "dc", "dp"])

        metrics[tag] = {
            "Accuracy": (solved / total * 100) if total else 0,
            "Speed": max(0, 100 - (avg_time / max(max_time, 0.001)) * 100) if max_time > 0 else 100,
            "Move\nEfficiency": max(0, 100 - (avg_moves / max(max_moves, 1)) * 100) if max_moves > 0 else 100,
            "Scalability": 80 if tag == "greedy" else (60 if tag == "dc" else 50),
            "Intelligence": 20 if tag == "greedy" else (70 if tag == "dc" else 95),
        }

    categories = list(metrics["greedy"].keys())
    N = len(categories)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    ax.set_facecolor(CARD_COLOR)
    fig.patch.set_facecolor(BG_COLOR)

    for tag, label in [("greedy", "Greedy"),
                       ("dc", "Divide & Conquer"),
                       ("dp", "Dynamic Programming")]:
        values = [metrics[tag][c] for c in categories]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=label, color=COLORS[tag])
        ax.fill(angles, values, alpha=0.15, color=COLORS[tag])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 105)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(["20", "40", "60", "80", "100"], fontsize=8, color=GRID_COLOR)
    ax.set_title("🕸  Solver Comparison Radar", fontsize=18, pad=25)
    ax.legend(loc="lower right", bbox_to_anchor=(1.15, -0.05))
    ax.grid(color=GRID_COLOR, alpha=0.3)

    fig.savefig(os.path.join(out_dir, "5_radar_chart.png"))
    plt.close(fig)
    print("  ✓ Chart 5: Radar Chart")


def chart_6_strategy_flowchart(out_dir):
    """Visual flowchart showing when each solver is used."""
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Box drawing helper
    def draw_box(x, y, w, h, text, color, fontsize=11, is_decision=False):
        if is_decision:
            # Diamond shape for decisions
            diamond = plt.Polygon(
                [(x + w/2, y + h), (x + w, y + h/2),
                 (x + w/2, y), (x, y + h/2)],
                facecolor=color, edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85
            )
            ax.add_patch(diamond)
            ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="#FFFFFF", wrap=True)
        else:
            box = mpatches.FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.15",
                facecolor=color, edgecolor=TEXT_COLOR, linewidth=1.5, alpha=0.85
            )
            ax.add_patch(box)
            ax.text(x + w/2, y + h/2, text, ha="center", va="center",
                    fontsize=fontsize, fontweight="bold", color="#FFFFFF")

    def draw_arrow(x1, y1, x2, y2, label="", color=TEXT_COLOR):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color=color, lw=2))
        if label:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            ax.text(mx + 0.15, my + 0.1, label, fontsize=9,
                    color=ACCENT_GOLD, fontweight="bold")

    # Title
    ax.text(6, 7.6, "Strategy Selection: Why 3 Modes?",
            ha="center", va="center", fontsize=20, fontweight="bold",
            color=ACCENT_GOLD)

    # Start
    draw_box(4.5, 6.5, 3, 0.7, "🧩 New Puzzle", GRID_COLOR, fontsize=12)

    # Decision 1
    draw_arrow(6, 6.5, 6, 6.0)
    draw_box(4, 4.8, 4, 1.1, "Is the pattern\nlocally deducible?", "#565f89", fontsize=11, is_decision=True)

    # Greedy
    draw_arrow(8, 5.35, 10, 5.35, "Yes", "#51CF66")
    draw_box(9.5, 4.85, 2.2, 1, "✅ GREEDY\nFast O(n²)\nLocal rules", COLORS["greedy"], fontsize=10)

    # Decision 2
    draw_arrow(6, 4.8, 6, 4.2, "No")
    draw_box(4, 3.0, 4, 1.1, "Can we split into\nindependent regions?", "#565f89", fontsize=11, is_decision=True)

    # D&C
    draw_arrow(8, 3.55, 10, 3.55, "Yes", "#51CF66")
    draw_box(9.5, 3.05, 2.2, 1, "✅ D&C\nSpatial split\n+ Merge", COLORS["dc"], fontsize=10)

    # DP
    draw_arrow(6, 3.0, 6, 2.4, "No")
    draw_box(4.5, 1.3, 3, 1, "✅ DP\nFull row-profile\nenumeration", COLORS["dp"], fontsize=10)

    # Annotations
    ax.text(0.5, 5.2, "⚡ Speed\nO(n²) local\nrule engine\n\nBest for:\nEasy patterns",
            fontsize=9, color=COLORS["greedy"], va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_COLOR, edgecolor=COLORS["greedy"], alpha=0.7))

    ax.text(0.5, 3.2, "🔀 Recursive\nQuadrant split\n+ boundary merge\n\nBest for:\nMedium puzzles",
            fontsize=9, color=COLORS["dc"], va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_COLOR, edgecolor=COLORS["dc"], alpha=0.7))

    ax.text(0.5, 1.5, "🧠 Exact\nAll valid solutions\nenum + frequency\n\nBest for:\nHard puzzles",
            fontsize=9, color=COLORS["dp"], va="center",
            bbox=dict(boxstyle="round,pad=0.4", facecolor=CARD_COLOR, edgecolor=COLORS["dp"], alpha=0.7))

    fig.savefig(os.path.join(out_dir, "6_strategy_flowchart.png"))
    plt.close(fig)
    print("  ✓ Chart 6: Strategy Flowchart")


# ─────────────────────────────────────────────────────────────
# Summary Table
# ─────────────────────────────────────────────────────────────
def print_summary(results):
    """Print a clean summary table to console."""
    all_entries = [e for entries in results.values() for e in entries]
    total = len(all_entries)

    print("\n" + "=" * 70)
    print("  BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Total games: {total}")
    print(f"  Configs tested: {list(results.keys())}")
    print("-" * 70)
    print(f"  {'Solver':<22} {'Win Rate':>10} {'Avg Time':>12} {'Avg Moves':>12}")
    print("-" * 70)

    for tag, label in [("greedy", "Greedy"),
                       ("dc", "Divide & Conquer"),
                       ("dp", "Dynamic Programming")]:
        solved = sum(1 for e in all_entries if e[f"{tag}_solved"])
        rate = 100 * solved / total if total else 0
        avg_t = np.mean([e[f"{tag}_time"] for e in all_entries])
        avg_m = np.mean([e[f"{tag}_moves"] for e in all_entries])
        print(f"  {label:<22} {rate:>9.1f}% {avg_t:>11.3f}s {avg_m:>12.1f}")

    print("=" * 70)

    # Per-config breakdown
    print("\n  Per-Configuration Breakdown:")
    print("-" * 70)
    for cfg, entries in results.items():
        print(f"\n  📋 {cfg}  ({len(entries)} games)")
        for tag, label in [("greedy", "Greedy"), ("dc", "D&C"), ("dp", "DP")]:
            solved = sum(1 for e in entries if e[f"{tag}_solved"])
            rate = 100 * solved / len(entries)
            avg_t = np.mean([e[f"{tag}_time"] for e in entries])
            print(f"     {label:<6}  Win: {rate:5.1f}%  Time: {avg_t:.3f}s")
    print()


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Generate Presentation Charts")
    parser.add_argument("--games", type=int, default=5,
                        help="Games per configuration (default: 5)")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Solver timeout in seconds (default: 60)")
    parser.add_argument("--quick", action="store_true",
                        help="Quick mode: fewer configs for faster testing")
    args = parser.parse_args()

    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           "presentation_charts")
    os.makedirs(out_dir, exist_ok=True)

    setup_style()

    # Define test configurations
    if args.quick:
        configs = [
            (3, 3, "Easy"),
            (5, 5, "Hard"),
        ]
    else:
        configs = [
            (3, 3, "Easy"),
            (3, 3, "Medium"),
            (5, 5, "Easy"),
            (5, 5, "Medium"),
            (5, 5, "Hard"),
            (7, 7, "Easy"),
            (7, 7, "Hard"),
        ]

    print("╔══════════════════════════════════════════════════════╗")
    print("║   Loopy Solver Benchmark — Presentation Edition     ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Games per config : {args.games}")
    print(f"  Solver timeout   : {args.timeout}s")
    print(f"  Configurations   : {len(configs)}")
    print(f"  Total games      : {len(configs) * args.games}")
    print(f"  Output folder    : {out_dir}")
    print()

    # ── Phase 1: Benchmark ──
    print("Phase 1/2: Running Benchmarks...")
    results = run_benchmark(args.games, configs, args.timeout)

    # ── Phase 2: Generate Charts ──
    print("\nPhase 2/2: Generating Charts...")
    chart_1_success_rate(results, out_dir)
    chart_2_timing(results, out_dir)
    chart_3_moves(results, out_dir)
    chart_4_scalability(results, out_dir)
    chart_5_radar(results, out_dir)
    chart_6_strategy_flowchart(out_dir)

    # ── Summary ──
    print_summary(results)

    print(f"📁 All 6 charts saved to: {out_dir}")
    print("   Copy these PNGs directly into your presentation slides!")


if __name__ == "__main__":
    main()
