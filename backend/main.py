"""FastAPI bridge for the SUMO traffic simulation and saved evaluation data."""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
LIVE_FILE = ROOT / "data" / "live" / "dashboard_state.json"
STOP_FILE = ROOT / "data" / "live" / "stop.signal"
COMPARISON_FILE = ROOT / "data" / "comparison" / "kpi_comparison.csv"
TRAFFIC_FILE = ROOT / "data" / "ml" / "traffic_long.csv"
MODEL_FILES = (
    ROOT / "data" / "ml" / "congestion_classifier.pkl",
    ROOT / "data" / "ml" / "feature_cols.pkl",
)
CONTROLLERS = {
    "fixed": "fixed",
    "queue_adaptive": "adaptive_controller:QueueAdaptiveController",
    "ml_adaptive": "adaptive_controller:MLAdaptiveController",
    "emergency": "adaptive_controller:EmergencyController",
    "rl": "adaptive_controller:RLController",
}
KPI_FIELDS = (
    "completed_trips",
    "mean_travel_time_s",
    "mean_waiting_time_s",
    "mean_time_loss_s",
    "avg_throughput_veh_h",
    "avg_total_halting",
    "emergency_mean_travel_time_s",
)

app = FastAPI(title="Smart Traffic Dashboard API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_simulation: subprocess.Popen | None = None
_simulation_lock = threading.Lock()


class SimulationRequest(BaseModel):
    scenario: str = "peak"
    controller: str = "queue_adaptive"
    seed: int = Field(default=1, ge=1)
    duration_s: int | None = Field(default=None, ge=60, le=86400)


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise HTTPException(status_code=404, detail=f"Data file not found: {path.relative_to(ROOT)}")
    try:
        with path.open(newline="", encoding="utf-8-sig") as source:
            return list(csv.DictReader(source))
    except (OSError, csv.Error) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read {path.relative_to(ROOT)}: {exc}") from exc


def _number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if pd.notna(number) else None


def _scenario_names() -> list[str]:
    path = ROOT / "scenarios.yaml"
    try:
        with path.open(encoding="utf-8") as source:
            config = yaml.safe_load(source)
        return sorted(config["scenarios"])
    except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not load scenarios.yaml: {exc}") from exc


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "smart-traffic-dashboard"}


@app.get("/api/config")
def config() -> dict[str, Any]:
    return {
        "scenarios": _scenario_names(),
        "controllers": list(CONTROLLERS),
        "live_file": str(LIVE_FILE.relative_to(ROOT)),
    }


@app.get("/api/topology")
def topology() -> dict[str, Any]:
    path = ROOT / "data" / "network_info.json"
    try:
        with path.open(encoding="utf-8") as source:
            network = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not load network topology: {exc}") from exc
    if not isinstance(network, dict) or not {"nodes", "edges"}.issubset(network):
        raise HTTPException(status_code=500, detail="Network topology has an unexpected format.")
    return network


@app.get("/api/live")
def live_state() -> dict[str, Any]:
    if not LIVE_FILE.is_file():
        raise HTTPException(
            status_code=404,
            detail="No live simulation state yet. Start a simulation from the dashboard.",
        )
    try:
        with LIVE_FILE.open(encoding="utf-8") as source:
            state = json.load(source)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=500, detail=f"Could not read live simulation state: {exc}") from exc
    if not isinstance(state, dict) or not {"system", "intersections", "emergency"}.issubset(state):
        raise HTTPException(status_code=500, detail="Live simulation state has an unexpected format.")
    return state


@app.get("/api/analytics")
def analytics(scenario: str | None = Query(default=None)) -> dict[str, Any]:
    rows = _read_csv(COMPARISON_FILE)
    if scenario:
        rows = [row for row in rows if row.get("scenario") == scenario]
    scenarios = sorted({row["scenario"] for row in _read_csv(COMPARISON_FILE) if row.get("scenario")})
    if not rows:
        raise HTTPException(status_code=404, detail=f"No comparison results for scenario '{scenario}'.")

    grouped: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault((row.get("scenario", ""), row.get("controller", "")), []).append(row)

    summaries = []
    for (scenario_name, controller), group in sorted(grouped.items()):
        summary: dict[str, Any] = {
            "scenario": scenario_name,
            "controller": controller,
            "controller_label": group[0].get("controller_label", controller),
            "runs": len(group),
        }
        for field in KPI_FIELDS:
            values = [value for row in group if (value := _number(row.get(field))) is not None]
            summary[field] = round(sum(values) / len(values), 2) if values else None
        summaries.append(summary)
    return {"scenarios": scenarios, "summaries": summaries}


@app.get("/api/predictions")
def predictions(scenario: str | None = Query(default=None)) -> dict[str, Any]:
    if not all(path.is_file() for path in MODEL_FILES):
        raise HTTPException(status_code=503, detail="Trained congestion model files are missing.")
    if not TRAFFIC_FILE.is_file():
        raise HTTPException(status_code=404, detail="ML traffic dataset not found. Generate the project dataset first.")

    try:
        from predict_api import predict_congestion

        if _simulation is not None and _simulation.poll() is None and LIVE_FILE.is_file():
            with LIVE_FILE.open(encoding="utf-8") as source:
                live = json.load(source)
            live_scenario = live.get("scenario")
            if scenario and live_scenario and scenario != live_scenario:
                raise HTTPException(
                    status_code=409,
                    detail=f"Live simulation is '{live_scenario}', not '{scenario}'.",
                )
            histories = live.get("prediction_history", {})
            results = []
            for edge_id, samples in histories.items():
                if len(samples) < 3:
                    continue
                recent = samples[-3:]
                latest = recent[-1]
                row = {
                    "speed_lag1": float(recent[-1]["speed_ms"]),
                    "speed_lag2": float(recent[-2]["speed_ms"]),
                    "speed_lag3": float(recent[-3]["speed_ms"]),
                    "queue_lag1": float(recent[-1]["queue_vehicles"]),
                    "queue_lag2": float(recent[-2]["queue_vehicles"]),
                    "queue_lag3": float(recent[-3]["queue_vehicles"]),
                    "occ_lag1": float(recent[-1]["occupancy"]),
                    "occ_lag2": float(recent[-2]["occupancy"]),
                    "occ_lag3": float(recent[-3]["occupancy"]),
                    "hour": int(float(latest["time"]) // 3600) % 24,
                    "free_flow_speed": float(latest["free_flow_speed"]),
                }
                results.append({
                    "edge_id": str(edge_id),
                    "tls_id": str(latest["tls_id"]),
                    "sample_time_s": float(latest["time"]),
                    "speed_ms": row["speed_lag1"],
                    "queue_vehicles": row["queue_lag1"],
                    "queue_length_m": float(live.get("approaches", {}).get(edge_id, {}).get("queue_length_m", 0)),
                    **predict_congestion(row),
                })
            return {
                "scenario": live_scenario,
                "run": f"live-{live.get('controller', 'simulation')}",
                "horizon_minutes": 5,
                "source": "live simulation (5-minute rolling samples)",
                "ready": len(results) > 0,
                "samples_needed": max(0, 3 - min(
                    (len(samples) for samples in histories.values()), default=0
                )),
                "predictions": results,
            }

        scenario = scenario or "peak"
        frame = pd.read_csv(TRAFFIC_FILE)
        frame = frame[frame["scenario"] == scenario]
        if frame.empty:
            raise HTTPException(status_code=404, detail=f"No saved traffic samples for scenario '{scenario}'.")
        # Use the longest saved run for this scenario; forecast labels are based on
        # five-minute bins, matching the model's training interval.
        run_lengths = frame.groupby("run")["time"].max().sort_values(ascending=False)
        selected_run = str(run_lengths.index[0])
        run_data = frame[frame["run"] == selected_run].sort_values(["edge_id", "time"])
        results = []
        for edge_id, edge_rows in run_data.groupby("edge_id"):
            recent = edge_rows.tail(3)
            if len(recent) < 3:
                continue
            latest = recent.iloc[-1]
            row = {
                "speed_lag1": float(recent.iloc[-1]["speed_ms"]),
                "speed_lag2": float(recent.iloc[-2]["speed_ms"]),
                "speed_lag3": float(recent.iloc[-3]["speed_ms"]),
                "queue_lag1": float(recent.iloc[-1]["queue_vehicles"]),
                "queue_lag2": float(recent.iloc[-2]["queue_vehicles"]),
                "queue_lag3": float(recent.iloc[-3]["queue_vehicles"]),
                "occ_lag1": float(recent.iloc[-1]["occupancy"]),
                "occ_lag2": float(recent.iloc[-2]["occupancy"]),
                "occ_lag3": float(recent.iloc[-3]["occupancy"]),
                "hour": int(float(latest["time"]) // 3600) % 24,
                "free_flow_speed": float(latest["free_flow_speed"]),
            }
            prediction = predict_congestion(row)
            results.append({
                "edge_id": str(edge_id),
                "tls_id": str(latest["tls_id"]),
                "sample_time_s": float(latest["time"]),
                "speed_ms": row["speed_lag1"],
                "queue_vehicles": row["queue_lag1"],
                **prediction,
            })
        return {
            "scenario": scenario,
            "run": selected_run,
            "horizon_minutes": 5,
            "source": "saved simulation dataset (simulation is not running)",
            "ready": True,
            "predictions": results,
        }
    except HTTPException:
        raise
    except (ImportError, OSError, ValueError, KeyError, TypeError) as exc:
        raise HTTPException(status_code=500, detail=f"Prediction could not be generated: {exc}") from exc


@app.get("/api/simulation/status")
def simulation_status() -> dict[str, Any]:
    with _simulation_lock:
        running = _simulation is not None and _simulation.poll() is None
        return {
            "running": running,
            "pid": _simulation.pid if running and _simulation else None,
            "return_code": _simulation.poll() if _simulation else None,
        }


@app.post("/api/simulation/start", status_code=202)
def start_simulation(request: SimulationRequest) -> dict[str, Any]:
    if request.scenario not in _scenario_names():
        raise HTTPException(status_code=422, detail=f"Unknown scenario '{request.scenario}'.")
    if request.controller not in CONTROLLERS:
        raise HTTPException(status_code=422, detail=f"Unknown controller '{request.controller}'.")

    global _simulation
    with _simulation_lock:
        if _simulation is not None and _simulation.poll() is None:
            raise HTTPException(status_code=409, detail="A simulation is already running.")
        LIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        try:
            LIVE_FILE.unlink(missing_ok=True)
            STOP_FILE.unlink(missing_ok=True)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not clear previous live state: {exc}") from exc
        log_path = LIVE_FILE.parent / "simulation.log"
        command = [
            sys.executable,
            str(ROOT / "traffic_sim.py"),
            "run",
            request.scenario,
            "--seed",
            str(request.seed),
            "--controller",
            CONTROLLERS[request.controller],
            "--live-json",
            str(LIVE_FILE),
            "--stop-file",
            str(STOP_FILE),
        ]
        if request.duration_s is not None:
            command.extend(["--duration", str(request.duration_s)])
        creation_flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
        try:
            with log_path.open("a", encoding="utf-8") as log:
                _simulation = subprocess.Popen(
                    command,
                    cwd=ROOT,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    creationflags=creation_flags,
                    start_new_session=os.name != "nt",
                )
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not start SUMO simulation: {exc}") from exc
        try:
            log_reference = str(log_path.relative_to(ROOT))
        except ValueError:
            log_reference = str(log_path)
        return {"started": True, "pid": _simulation.pid, "scenario": request.scenario,
                "controller": request.controller, "log": log_reference}


@app.post("/api/simulation/stop")
def stop_simulation() -> dict[str, Any]:
    global _simulation
    with _simulation_lock:
        if _simulation is None or _simulation.poll() is not None:
            raise HTTPException(status_code=409, detail="There is no running simulation to stop.")
        process = _simulation
        try:
            STOP_FILE.write_text("stop\n", encoding="utf-8")
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.terminate()
            process.wait(timeout=5)
        except OSError as exc:
            raise HTTPException(status_code=500, detail=f"Could not stop the simulation: {exc}") from exc
        return {"stopped": True, "return_code": process.returncode}
