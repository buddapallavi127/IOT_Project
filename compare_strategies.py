#!/usr/bin/env python3
"""
compare_strategies.py  --  Person 3: Performance Comparison of Traffic Control Strategies
===========================================================================================

Runs all four controllers on the same scenario/seeds, collects KPI CSVs, and produces:
  - data/comparison/kpi_comparison.csv   (one row per run)
  - data/comparison/comparison_report.md (human-readable summary)

Usage
-----
  # Quick comparison (fixed + queue_adaptive on peak, seeds 1-2)
  python compare_strategies.py

  # Full comparison including ML and RL, more seeds
  python compare_strategies.py --controllers fixed queue_adaptive ml_adaptive rl emergency \\
                                --scenarios peak congestion emergency \\
                                --seeds 1 2 3 -j 2

  # Skip running sims (just re-analyse data/raw)
  python compare_strategies.py --no-sim

Requirements: traffic_sim.py + adaptive_controller.py in the same folder.
IMPORTANT: Always run with the conda/sumo-capable Python, e.g.:
  /opt/miniconda3/bin/python compare_strategies.py ...
"""

import argparse, csv, glob, json, os, subprocess, sys, time
from collections import defaultdict

import pandas as pd

ROOT = os.path.dirname(os.path.abspath(__file__))
RAW_DIR  = os.path.join(ROOT, "data", "raw")
COMP_DIR = os.path.join(ROOT, "data", "comparison")


# ---------------------------------------------------------------------------
# Auto-detect a Python interpreter whose SUMO binaries actually work.
# This fixes the "dyld: Library not loaded libxerces-c" problem that occurs
# when the script is accidentally run with the system Python 3.12 whose
# pip-installed SUMO has a broken dylib, while the conda Python works fine.
# ---------------------------------------------------------------------------
def _find_working_python() -> str:
    """Return path to the first Python interpreter where duarouter works."""
    candidates = [
        sys.executable,                          # whichever python ran this script
        "/opt/miniconda3/bin/python",
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    ]
    seen = []
    for py in candidates:
        if py in seen or not os.path.exists(py):
            continue
        seen.append(py)
        probe = subprocess.run(
            [py, "-c",
             "import sumolib; import subprocess, sys; "
             "r=subprocess.run([sumolib.checkBinary('duarouter'),'--version'],"
             "capture_output=True); sys.exit(r.returncode)"],
            capture_output=True, timeout=10
        )
        if probe.returncode == 0:
            return py
    # Fall back to sys.executable and hope for the best
    return sys.executable


PYTHON_BIN = _find_working_python()
if PYTHON_BIN != sys.executable:
    print(f"[compare] Using SUMO-capable Python: {PYTHON_BIN}")


# Map controller name -> traffic_sim.py --controller argument
CONTROLLER_MAP = {
    "fixed":          "fixed",
    "queue_adaptive": "adaptive_controller:QueueAdaptiveController",
    "ml_adaptive":    "adaptive_controller:MLAdaptiveController",
    "emergency":      "adaptive_controller:EmergencyController",
    "rl":             "adaptive_controller:RLController",
}

CONTROLLER_LABELS = {
    "fixed":          "Fixed-Time (baseline)",
    "queue_adaptive": "Queue-Adaptive",
    "ml_adaptive":    "ML-Adaptive (Person 1 API)",
    "emergency":      "Emergency Priority",
    "rl":             "Reinforcement Learning",
}

KPI_COLS = [
    "completed_trips",
    "mean_travel_time_s",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "emergency_mean_travel_time_s",
    "emergency_trips",
]


# ============================================================ simulation runner
def run_one(scenario: str, seed: int, controller_name: str) -> dict:
    """Run a single scenario/seed/controller and return KPI dict."""
    ctrl_arg = CONTROLLER_MAP[controller_name]
    out_dir  = os.path.join(RAW_DIR, f"{scenario}_seed{seed}_{controller_name}")

    if os.path.exists(os.path.join(out_dir, "run_meta.json")):
        print(f"  [skip] {scenario} seed={seed} ctrl={controller_name} (already done)")
    else:
        print(f"  [run ] {scenario} seed={seed} ctrl={controller_name} ...")
        cmd = [
            PYTHON_BIN, os.path.join(ROOT, "traffic_sim.py"),
            "run", scenario,
            "--seed", str(seed),
            "--controller", ctrl_arg,
            "--out-dir", out_dir,
        ]
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            print(f"  [FAIL] {r.stderr[-800:]}")
            return {}

    # Load meta
    meta_path = os.path.join(out_dir, "run_meta.json")
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path) as f:
        meta = json.load(f)

    row = {"scenario": scenario, "seed": seed, "controller": controller_name,
           "controller_label": CONTROLLER_LABELS.get(controller_name, controller_name)}
    row.update(meta.get("kpis", {}))

    # Add system-level KPIs from system_metrics.csv
    sys_csv = os.path.join(out_dir, "system_metrics.csv")
    if os.path.exists(sys_csv):
        df = pd.read_csv(sys_csv)
        row["avg_throughput_veh_h"] = round(df["throughput_veh_h"].mean(), 1)
        row["avg_total_halting"]    = round(df["total_halting"].mean(), 2)
        row["avg_mean_speed_ms"]    = round(df["mean_speed"].mean(), 3)
        row["avg_mean_wait_s"]      = round(df["mean_waiting_time"].mean(), 2)

    return row


# ============================================================ load existing runs
def load_existing_runs() -> list:
    rows = []
    for run_path in sorted(glob.glob(os.path.join(RAW_DIR, "*_seed*_*"))):
        run = os.path.basename(run_path)
        meta_path = os.path.join(run_path, "run_meta.json")
        if not os.path.exists(meta_path):
            continue
        with open(meta_path) as f:
            meta = json.load(f)

        scenario = meta.get("scenario")
        seed = meta.get("seed")
        controller_name = meta.get("controller")
        if not scenario or seed is None or not controller_name:
            continue

        row = {"scenario": scenario, "seed": int(seed), "controller": controller_name,
               "controller_label": CONTROLLER_LABELS.get(controller_name, controller_name)}
        row.update(meta.get("kpis", {}))

        sys_csv = os.path.join(run_path, "system_metrics.csv")
        if os.path.exists(sys_csv):
            df = pd.read_csv(sys_csv)
            row["avg_throughput_veh_h"] = round(df["throughput_veh_h"].mean(), 1)
            row["avg_total_halting"]    = round(df["total_halting"].mean(), 2)
            row["avg_mean_speed_ms"]    = round(df["mean_speed"].mean(), 3)
            row["avg_mean_wait_s"]      = round(df["mean_waiting_time"].mean(), 2)
        rows.append(row)
    return rows


# ============================================================ reporting
def _fmt(val, ref=None, lower_is_better=True):
    """Format a value with optional improvement % over ref."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return "N/A"
    s = f"{val:.2f}"
    if ref is not None and ref != 0 and not pd.isna(ref):
        delta = (val - ref) / abs(ref) * 100
        sign = "+" if delta > 0 else ""
        good = (delta < 0) == lower_is_better
        tag  = "✓" if good else "✗"
        s   += f" ({sign}{delta:.1f}% {tag})"
    return s


def generate_report(df: pd.DataFrame, out_path: str):
    """Write a markdown comparison report."""
    # Ensure controller_label exists
    if "controller_label" not in df.columns:
        df = df.copy()
        df["controller_label"] = df["controller"].map(CONTROLLER_LABELS).fillna(df["controller"])

    lines = ["# Traffic Control Strategy Comparison Report", "",
             f"*Generated: {time.strftime('%Y-%m-%d %H:%M')}*", ""]

    # Columns to aggregate – only include those actually present in the df
    OPTIONAL_AGG = [
        ("label",                        "controller_label",              "first"),
        ("completed_trips",              "completed_trips",               "mean"),
        ("mean_travel_time_s",           "mean_travel_time_s",           "mean"),
        ("mean_waiting_time_s",          "mean_waiting_time_s",          "mean"),
        ("mean_time_loss_s",             "mean_time_loss_s",             "mean"),
        ("avg_throughput_veh_h",         "avg_throughput_veh_h",         "mean"),
        ("avg_total_halting",            "avg_total_halting",            "mean"),
        ("avg_mean_speed_ms",            "avg_mean_speed_ms",            "mean"),
        ("avg_mean_wait_s",              "avg_mean_wait_s",              "mean"),
        ("emergency_mean_travel_time_s", "emergency_mean_travel_time_s", "mean"),
    ]

    for scenario in sorted(df["scenario"].unique()):
        sdf = df[df["scenario"] == scenario].copy()
        lines += [f"## Scenario: `{scenario}`", ""]

        # Build agg spec only for columns that exist
        agg_spec = {
            out_col: (src_col, func)
            for out_col, src_col, func in OPTIONAL_AGG
            if src_col in sdf.columns
        }

        agg = sdf.groupby("controller").agg(**agg_spec).reset_index()

        # Ensure label column exists
        if "label" not in agg.columns:
            agg["label"] = agg["controller"].map(CONTROLLER_LABELS).fillna(agg["controller"])

        # Fixed-time reference
        fixed_row = agg[agg["controller"] == "fixed"]
        ref = fixed_row.iloc[0].to_dict() if len(fixed_row) > 0 else {}

        # Table header
        lines.append("| Controller | Trips | Travel Time (s) | Wait Time (s) | Time Loss (s) | Throughput (veh/h) | Queue (halt) | Speed (m/s) | Emg Travel (s) |")
        lines.append("|---|---|---|---|---|---|---|---|---|")

        for _, r in agg.iterrows():
            r = r.to_dict()
            emg = _fmt(r.get("emergency_mean_travel_time_s"),
                       ref.get("emergency_mean_travel_time_s"), lower_is_better=True)
            lines.append(
                f"| **{r.get('label', r.get('controller', '?'))}** "
                f"| {_fmt(r.get('completed_trips'), ref.get('completed_trips'), lower_is_better=False)} "
                f"| {_fmt(r.get('mean_travel_time_s'), ref.get('mean_travel_time_s'))} "
                f"| {_fmt(r.get('mean_waiting_time_s'), ref.get('mean_waiting_time_s'))} "
                f"| {_fmt(r.get('mean_time_loss_s'), ref.get('mean_time_loss_s'))} "
                f"| {_fmt(r.get('avg_throughput_veh_h'), ref.get('avg_throughput_veh_h'), lower_is_better=False)} "
                f"| {_fmt(r.get('avg_total_halting'), ref.get('avg_total_halting'))} "
                f"| {_fmt(r.get('avg_mean_speed_ms'), ref.get('avg_mean_speed_ms'), lower_is_better=False)} "
                f"| {emg} |"
            )
        lines += ["", "> ✓ = improvement over fixed-time baseline  ✗ = regression", ""]

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[Report] Saved to {out_path}")



# ============================================================ main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--controllers", nargs="+",
                    default=["fixed", "queue_adaptive"],
                    choices=list(CONTROLLER_MAP),
                    help="Controllers to compare")
    ap.add_argument("--scenarios", nargs="+",
                    default=["peak"],
                    help="Scenario names from scenarios.yaml")
    ap.add_argument("--seeds", nargs="+", type=int, default=[1, 2],
                    help="Random seeds")
    ap.add_argument("-j", "--jobs", type=int, default=1,
                    help="Parallel simulation runs")
    ap.add_argument("--no-sim", action="store_true",
                    help="Skip running simulations, only re-analyse data/raw")
    a = ap.parse_args()

    os.makedirs(COMP_DIR, exist_ok=True)

    rows = []
    if a.no_sim:
        print("[compare] Loading existing runs from data/raw ...")
        rows = load_existing_runs()
    else:
        jobs = [(sc, seed, ctrl)
                for sc in a.scenarios
                for seed in a.seeds
                for ctrl in a.controllers]

        if a.jobs > 1:
            from concurrent.futures import ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=a.jobs) as ex:
                results = list(ex.map(lambda j: run_one(*j), jobs))
            rows = [r for r in results if r]
        else:
            for sc, seed, ctrl in jobs:
                r = run_one(sc, seed, ctrl)
                if r:
                    rows.append(r)

    if not rows:
        sys.exit("No data found. Run simulations first.")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(COMP_DIR, "kpi_comparison.csv")
    df.to_csv(csv_path, index=False)
    print(f"[compare] KPI table saved to {csv_path}")

    md_path = os.path.join(COMP_DIR, "comparison_report.md")
    generate_report(df, md_path)

    # Print quick summary to terminal
    print("\n=== Quick Summary (mean across seeds) ===")
    summary_cols = [c for c in ["controller", "scenario", "mean_waiting_time_s",
                                  "mean_time_loss_s", "avg_throughput_veh_h",
                                  "emergency_mean_travel_time_s"] if c in df.columns]
    if summary_cols:
        print(df[summary_cols].groupby(["controller", "scenario"]).mean().round(2).to_string())


if __name__ == "__main__":
    main()
