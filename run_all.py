#!/usr/bin/env python3
"""
run_all.py  --  Master Runner for the Smart Traffic Simulation & IoT Project
===========================================================================
Executes the entire project pipeline seamlessly in one terminal using the
working SUMO/conda environment.

Pipeline Stages:
  1. Generate Multi-Scenario Dataset (traffic_sim.py all)
  2. Train ML Congestion Models (train_classifier.py)
  3. Benchmark & Compare Signal Controllers (compare_strategies.py)
  4. Live IoT MQTT Streaming Demo (mqtt_demo.py)

Usage:
  python run_all.py          # Runs all 4 stages end-to-end
  python run_all.py --train  # Train ML models only
  python run_all.py --compare# Run controller comparison only
  python run_all.py --mqtt   # Run live IoT MQTT demo only
"""

import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

def get_python_exe():
    """Find Python binary with working SUMO & amqtt."""
    candidates = [
        "/opt/miniconda3/bin/python",
        sys.executable,
        "/opt/homebrew/bin/python3",
        "/usr/local/bin/python3",
    ]
    for c in candidates:
        if os.path.exists(c):
            try:
                res = subprocess.run(
                    [c, "-c", "import sumolib, amqtt; sumolib.checkBinary('duarouter')"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if res.returncode == 0:
                    return c
            except Exception:
                pass
    return "/opt/miniconda3/bin/python" if os.path.exists("/opt/miniconda3/bin/python") else sys.executable

PYTHON = get_python_exe()

def print_banner(title):
    print("\n" + "=" * 65)
    print(f"  🚦 {title}")
    print("=" * 65 + "\n")

def run_step(step_name, cmd):
    print_banner(step_name)
    print(f"[CMD] {' '.join(cmd)}\n")
    ret = subprocess.run(cmd, cwd=ROOT)
    if ret.returncode != 0:
        print(f"\n⚠️  Warning: Stage '{step_name}' returned exit code {ret.returncode}")
    else:
        print(f"\n✅ Completed: {step_name}\n")
    return ret.returncode

def main():
    parser = argparse.ArgumentParser(description="Master runner for the Traffic & IoT Project")
    parser.add_argument("--data", action="store_true", help="Generate multi-scenario ML dataset only")
    parser.add_argument("--train", action="store_true", help="Train ML models only")
    parser.add_argument("--compare", action="store_true", help="Run strategy comparison only")
    parser.add_argument("--mqtt", action="store_true", help="Run live IoT MQTT streaming demo only")
    parser.add_argument("--gui", action="store_true", help="Launch interactive visual simulation with SUMO-GUI")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3], help="Seeds for data generation")
    args = parser.parse_args()

    if args.gui:
        cmd = [PYTHON, os.path.join(ROOT, "visualize.py")]
        subprocess.run(cmd)
        return

    run_all = not (args.data or args.train or args.compare or args.mqtt)

    print("=" * 65)
    print("  🚀 Smart Traffic IoT & Adaptive Signal Control System")
    print(f"  🐍 Using Python: {PYTHON}")
    print("=" * 65)

    if run_all or args.data:
        seeds_str = [str(s) for s in args.seeds]
        run_step(
            "Stage 1: Generate Multi-Scenario ML Dataset",
            [PYTHON, os.path.join(ROOT, "traffic_sim.py"), "all", "--seeds"] + seeds_str + ["-j", "4"]
        )

    if run_all or args.train:
        run_step(
            "Stage 2: Train ML Congestion Classifiers (RandomForest & XGBoost)",
            [PYTHON, os.path.join(ROOT, "train_classifier.py")]
        )
        run_step(
            "Stage 2b: Test ML Prediction API",
            [PYTHON, os.path.join(ROOT, "predict_api.py")]
        )

    if run_all or args.compare:
        seeds_str = [str(s) for s in args.seeds]
        run_step(
            "Stage 3: Benchmark Signal Controllers (Fixed vs Queue vs ML vs RL)",
            [
                PYTHON, os.path.join(ROOT, "compare_strategies.py"),
                "--controllers", "fixed", "queue_adaptive", "ml_adaptive", "rl",
                "--scenarios", "peak", "congestion", "emergency",
                "--seeds"
            ] + seeds_str + ["-j", "2"]
        )

    if run_all or args.mqtt:
        run_step(
            "Stage 4: Live IoT MQTT Simulation Demo",
            [PYTHON, os.path.join(ROOT, "mqtt_demo.py")]
        )

    print("\n" + "=" * 65)
    print("  🎉 All requested project workflows finished successfully!")
    print("=" * 65)

if __name__ == "__main__":
    main()
