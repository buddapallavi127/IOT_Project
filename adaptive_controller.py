#!/usr/bin/env python3
"""
adaptive_controller.py  –  Person 3: Adaptive Traffic Signal Control & IoT Communication
=========================================================================================

Provides four plug-in controllers that work with traffic_sim.py:

  FixedTimeController      – baseline: keeps SUMO's default static timings (already in traffic_sim.py)
  QueueAdaptiveController  – extends green phase for the direction with the longest queue
  MLAdaptiveController     – uses predict_congestion() to anticipate congestion & pre-extend green
  EmergencyController      – wraps any base controller; forces green corridor for ambulances
  RLController             – simple Q-learning controller (tabular, trains online)

MQTT IoT Layer
--------------
When mqtt_host is supplied every controller publishes:
  traffic/p3/intersection/<tls_id>/state   → phase, queues, timing
  traffic/p3/intersection/<tls_id>/cmd     ← (subscribe) receive remote override commands
  traffic/p3/emergency                     → active ambulance alert
  traffic/p3/performance                   → periodic KPI snapshot

Usage with traffic_sim.py
--------------------------
  python traffic_sim.py run emergency --controller adaptive_controller:EmergencyController
  python traffic_sim.py run peak      --controller adaptive_controller:MLAdaptiveController
  python traffic_sim.py run peak      --controller adaptive_controller:QueueAdaptiveController
  python traffic_sim.py run peak      --controller adaptive_controller:RLController
"""

from __future__ import annotations
import json, os, sys, time, threading, collections, math, random
from typing import Optional, Dict, Any

# --------------------------------------------------------------------------- helpers
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# Safely try to import predict_congestion (Person 1's API)
try:
    from predict_api import predict_congestion as _predict_congestion
    _ML_AVAILABLE = True
except Exception:          # model file may not exist during development
    _ML_AVAILABLE = False
    def _predict_congestion(data: dict) -> dict:          # type: ignore[misc]
        return {"congested_5min": 0, "confidence": 0.0}


# --------------------------------------------------------------------------- MQTT IoT layer
class MQTTBroker:
    """
    Thin wrapper around paho-mqtt for inter-intersection IoT communication.
    Gracefully degrades (no-op) when paho is not installed or host is None.
    """

    PREFIX = "traffic/p3"

    def __init__(self, host: Optional[str] = None, port: int = 1883):
        self.client = None
        self._cmds: Dict[str, Any] = {}   # tls_id -> latest remote command
        self._lock = threading.Lock()

        if host is None:
            return
        try:
            import paho.mqtt.client as mqtt
            try:
                c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="p3_controller")
            except AttributeError:
                c = mqtt.Client(client_id="p3_controller")

            def on_connect(client, userdata, flags, rc, *args):
                client.subscribe(f"{self.PREFIX}/intersection/+/cmd")
                client.subscribe(f"{self.PREFIX}/override")

            def on_message(client, userdata, msg):
                try:
                    payload = json.loads(msg.payload.decode())
                    # Extract tls_id from topic:  .../intersection/<tls_id>/cmd
                    parts = msg.topic.split("/")
                    if len(parts) >= 4 and parts[-1] == "cmd":
                        tls_id = parts[-2]
                        with self._lock:
                            self._cmds[tls_id] = payload
                except Exception:
                    pass

            c.on_connect = on_connect
            c.on_message = on_message
            c.connect(host, port, keepalive=60)
            c.loop_start()
            self.client = c
            print(f"[MQTT] Connected to {host}:{port}  prefix={self.PREFIX}")
        except Exception as exc:
            print(f"[MQTT] Warning - could not connect ({exc}). Running without MQTT.")

    # ------------------------------------------------------------------ publish helpers
    def publish_intersection(self, tls_id: str, payload: dict):
        self._pub(f"{self.PREFIX}/intersection/{tls_id}/state", payload)

    def publish_emergency(self, vehicles: list):
        self._pub(f"{self.PREFIX}/emergency", {"vehicles": vehicles, "ts": time.time()})

    def publish_performance(self, kpis: dict):
        self._pub(f"{self.PREFIX}/performance", kpis)

    def _pub(self, topic: str, payload: dict):
        if self.client:
            try:
                self.client.publish(topic, json.dumps(payload), qos=0, retain=False)
            except Exception:
                pass

    # ------------------------------------------------------------------ subscribe helper
    def get_cmd(self, tls_id: str) -> Optional[dict]:
        """Return (and clear) the latest remote command for a given intersection."""
        with self._lock:
            return self._cmds.pop(tls_id, None)

    def close(self):
        if self.client:
            self.client.loop_stop()
            self.client.disconnect()


# --------------------------------------------------------------------------- utility helpers
def _safe_mean(lst: list) -> float:
    return round(sum(lst) / len(lst), 3) if lst else 0.0


def _get_queue(traci, edges: list) -> Dict[str, int]:
    """Return {edge_id: halting_vehicles} for a list of edges."""
    return {e: traci.edge.getLastStepHaltingNumber(e) for e in edges}


def _get_mean_speed(traci, edges: list) -> float:
    speeds, counts = [], []
    for e in edges:
        n = traci.edge.getLastStepVehicleNumber(e)
        s = traci.edge.getLastStepMeanSpeed(e)
        if n > 0:
            speeds.append(s * n)
            counts.append(n)
    return sum(speeds) / sum(counts) if counts else 0.0


def _get_phase_queues(traci, tls_id: str, topology: dict) -> Dict[int, int]:
    """
    For each logical phase index compute total queued vehicles on the
    incoming edges most associated with that phase's green links.
    Returns {phase_index: queue_count}.
    """
    logic = traci.trafficlight.getAllProgramLogics(tls_id)
    if not logic:
        return {}
    phases = logic[0].phases
    links  = traci.trafficlight.getControlledLinks(tls_id)
    incoming = topology["tls"][tls_id]["incoming_edges"]

    result: Dict[int, int] = {}
    for ph_idx, ph in enumerate(phases):
        state = ph.state
        green_edges: set = set()
        for i, link_group in enumerate(links):
            if i < len(state) and state[i].lower() == "g":
                for link in link_group:
                    if link:
                        edge = link[0].rsplit("_", 1)[0]
                        if edge in incoming:
                            green_edges.add(edge)
        q = sum(traci.edge.getLastStepHaltingNumber(e) for e in green_edges)
        result[ph_idx] = q
    return result


# ===========================================================================
# SHARED BASE
# ===========================================================================
class BaseController:
    """
    Matches the plug-in interface expected by traffic_sim.py.
    All Person-3 controllers inherit from here.
    """
    name = "base"

    # ----- mqtt config (set before calling setup if needed) --------
    mqtt_host: Optional[str] = None
    mqtt_port: int = 1883

    def setup(self, traci, topology: dict):
        self.traci = traci
        self.topology = topology
        self.tls_ids: list = list(topology["tls"])
        self.mqtt = MQTTBroker(self.mqtt_host, self.mqtt_port)
        self._perf_acc: Dict[str, list] = {"wait": [], "queue": [], "speed": []}
        self._t_last_pub = 0.0
        self._pub_every = 30   # seconds between performance MQTT publishes

    def step(self, t: float, traci):
        pass

    # ------------------------------------------------------------------ MQTT helpers
    def _maybe_publish_perf(self, t: float):
        if t - self._t_last_pub >= self._pub_every:
            self.mqtt.publish_performance({
                "controller": self.name,
                "time": t,
                "avg_queue": _safe_mean(self._perf_acc["queue"]),
                "avg_wait":  _safe_mean(self._perf_acc["wait"]),
                "avg_speed": _safe_mean(self._perf_acc["speed"]),
            })
            self._perf_acc = {"wait": [], "queue": [], "speed": []}
            self._t_last_pub = t

    def _publish_tls(self, t: float, tls_id: str, extra: Optional[dict] = None):
        try:
            phase  = self.traci.trafficlight.getPhase(tls_id)
            state  = self.traci.trafficlight.getRedYellowGreenState(tls_id)
            remain = round(self.traci.trafficlight.getNextSwitch(tls_id) - t, 1)
            queues = {e: self.traci.edge.getLastStepHaltingNumber(e)
                      for e in self.topology["tls"][tls_id]["incoming_edges"]}
            payload: dict = {"tls": tls_id, "time": t, "phase": phase, "state": state,
                             "next_switch_in": remain, "queues": queues}
            if extra:
                payload.update(extra)
            self.mqtt.publish_intersection(tls_id, payload)
        except Exception:
            pass

    def teardown(self):
        self.mqtt.close()


# ===========================================================================
# 1. FIXED-TIME CONTROLLER  (baseline - no-op, keeps SUMO defaults)
# ===========================================================================
class FixedTimeController(BaseController):
    """
    Baseline: does not touch the SUMO signal programs.
    Publishes state over MQTT every `publish_every` seconds.
    """
    name = "fixed"
    publish_every: int = 10

    def setup(self, traci, topology: dict):
        super().setup(traci, topology)
        self._t_last = -999.0

    def step(self, t: float, traci):
        if t - self._t_last >= self.publish_every:
            for tls in self.tls_ids:
                self._publish_tls(t, tls)
            self._t_last = t
        self._maybe_publish_perf(t)


# ===========================================================================
# 2. QUEUE-ADAPTIVE CONTROLLER
# ===========================================================================
class QueueAdaptiveController(BaseController):
    """
    Real-time queue-aware green extension.

    Algorithm
    ---------
    - Poll every `check_every` seconds.
    - Read halting-vehicle counts on every approach edge.
    - Map each approach edge to the TLS phase that gives it green (built once at setup).
    - If the CURRENT green phase has the highest queue → extend its duration by
      `extend_s` seconds (up to `max_green`) using setPhaseDuration() only.
    - If a DIFFERENT phase has a meaningfully larger queue → switch to it directly
      with setPhase().  No program-logic mutation; SUMO transitions safely.
    - Enforce `min_green` before any switch is allowed.
    """
    name = "queue_adaptive"

    min_green:   int = 15    # minimum green before a switch is considered (s)
    max_green:   int = 60    # maximum green duration per phase (s)
    extend_s:    int = 5     # green extension step (s)
    check_every: int = 5     # decision interval (s)
    switch_threshold: int = 3  # queue difference to justify a switch

    def setup(self, traci, topology: dict):
        super().setup(traci, topology)
        self._phase_start: Dict[str, float] = {t: 0.0 for t in self.tls_ids}
        self._prev_phase:  Dict[str, int]   = {}
        self._t_last_check: float = -999.0
        # Build edge -> set of green phases mapping once
        self._edge_to_green_phases: Dict[str, Dict[str, set]] = {}  # tls -> edge -> {phase_idx}
        self._n_phases: Dict[str, int] = {}
        self._build_phase_map(traci, topology)

    def _build_phase_map(self, traci, topology: dict):
        """Pre-compute which phases give green to which approach edges."""
        for tls in self.tls_ids:
            self._edge_to_green_phases[tls] = {}
            try:
                logic = traci.trafficlight.getAllProgramLogics(tls)
                if not logic:
                    continue
                phases = logic[0].phases
                self._n_phases[tls] = len(phases)
                links = traci.trafficlight.getControlledLinks(tls)
                incoming = set(topology["tls"][tls]["incoming_edges"])

                for ph_idx, ph in enumerate(phases):
                    state = ph.state
                    for i, link_group in enumerate(links):
                        if i < len(state) and state[i].lower() == "g":
                            for link in link_group:
                                if link:
                                    edge = link[0].rsplit("_", 1)[0]
                                    if edge in incoming:
                                        self._edge_to_green_phases[tls].setdefault(edge, set()).add(ph_idx)
            except Exception:
                pass

    def _phase_queue(self, traci, tls: str, ph_idx: int) -> int:
        """Total halting vehicles on edges that are green in phase ph_idx."""
        total = 0
        for edge, phases in self._edge_to_green_phases.get(tls, {}).items():
            if ph_idx in phases:
                total += traci.edge.getLastStepHaltingNumber(edge)
        return total

    def step(self, t: float, traci):
        if t - self._t_last_check < self.check_every:
            return
        self._t_last_check = t

        for tls in self.tls_ids:
            # ---- MQTT override ----
            cmd = self.mqtt.get_cmd(tls)
            if cmd and "phase" in cmd:
                try:
                    traci.trafficlight.setPhase(tls, int(cmd["phase"]))
                    self._phase_start[tls] = t
                    self._prev_phase[tls] = int(cmd["phase"])
                except Exception:
                    pass
                continue

            try:
                current_phase = traci.trafficlight.getPhase(tls)
                n_phases = self._n_phases.get(tls, 0)
                if n_phases == 0:
                    continue

                # Track phase start time
                if self._prev_phase.get(tls, -1) != current_phase:
                    self._phase_start[tls] = t
                    self._prev_phase[tls] = current_phase

                time_in_phase = t - self._phase_start.get(tls, t)

                if time_in_phase < self.min_green:
                    continue

                # Score every phase by queue
                phase_queues = {ph: self._phase_queue(traci, tls, ph) for ph in range(n_phases)}
                curr_q = phase_queues.get(current_phase, 0)
                best_phase = max(phase_queues, key=lambda ph: phase_queues[ph])
                best_q = phase_queues[best_phase]

                incoming = self.topology["tls"][tls]["incoming_edges"]
                total_q  = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming)
                self._perf_acc["queue"].append(total_q)
                self._perf_acc["speed"].append(_get_mean_speed(traci, incoming))

                if best_phase == current_phase:
                    # Extend current green if it still has the worst queue
                    if curr_q > 2 and time_in_phase < self.max_green:
                        remaining = round(traci.trafficlight.getNextSwitch(tls) - t, 1)
                        if remaining < self.extend_s:
                            # Only extend if near the end of the current phase
                            traci.trafficlight.setPhaseDuration(tls, self.extend_s)
                else:
                    # Switch only if difference is meaningful
                    if best_q - curr_q >= self.switch_threshold:
                        traci.trafficlight.setPhase(tls, best_phase)
                        self._phase_start[tls] = t
                        self._prev_phase[tls] = best_phase

                self._publish_tls(t, tls, extra={
                    "phase_queues": phase_queues,
                    "best_phase": best_phase,
                    "controller": self.name,
                })
            except Exception:
                pass

        self._maybe_publish_perf(t)


# ===========================================================================
# 3. ML-ADAPTIVE CONTROLLER  (uses Person 1's predict_congestion)
# ===========================================================================
class MLAdaptiveController(BaseController):
    """
    ML-enhanced signal controller.

    Uses predict_congestion() every `ml_interval` seconds to look 5 minutes ahead.
    If congestion is predicted on an approach, that approach's green phase is
    pre-extended before the congestion materialises.

    Uses the same safe TraCI calls as QueueAdaptiveController:
    setPhaseDuration() for extension, setPhase() for switching.
    No setProgramLogic() in the hot path.
    """
    name = "ml_adaptive"

    ml_interval:      int   = 30
    extend_high:      int   = 10    # extension (s) when confidence >= high_threshold
    extend_medium:    int   = 5
    high_threshold:   float = 0.75
    medium_threshold: float = 0.50
    history_len:      int   = 3
    check_every:      int   = 5
    min_green:        int   = 10
    switch_threshold: int   = 3

    def setup(self, traci, topology: dict):
        super().setup(traci, topology)
        self._ml_history: Dict[str, collections.deque] = {}
        self._t_last_ml   = -999.0
        self._t_last_check = -999.0
        self._phase_start: Dict[str, float] = {t: 0.0 for t in self.tls_ids}
        self._prev_phase:  Dict[str, int]   = {}

        self._free_flow: Dict[str, float] = {
            eid: v["speed_limit"]
            for eid, v in topology["edges"].items()
            if v.get("approach")
        }
        for eid in self._free_flow:
            self._ml_history[eid] = collections.deque(maxlen=self.history_len)

        # Reuse the phase map from QueueAdaptiveController logic
        self._edge_to_green_phases: Dict[str, Dict[str, set]] = {}
        self._n_phases: Dict[str, int] = {}
        self._build_phase_map(traci, topology)

    def _build_phase_map(self, traci, topology: dict):
        for tls in self.tls_ids:
            self._edge_to_green_phases[tls] = {}
            try:
                logic = traci.trafficlight.getAllProgramLogics(tls)
                if not logic:
                    continue
                phases = logic[0].phases
                self._n_phases[tls] = len(phases)
                links = traci.trafficlight.getControlledLinks(tls)
                incoming = set(topology["tls"][tls]["incoming_edges"])
                for ph_idx, ph in enumerate(phases):
                    state = ph.state
                    for i, link_group in enumerate(links):
                        if i < len(state) and state[i].lower() == "g":
                            for link in link_group:
                                if link:
                                    edge = link[0].rsplit("_", 1)[0]
                                    if edge in incoming:
                                        self._edge_to_green_phases[tls].setdefault(edge, set()).add(ph_idx)
            except Exception:
                pass

    def _phase_score(self, traci, tls: str, ph_idx: int, ml_boost: Dict[str, float]) -> float:
        """Queue count + ML confidence boost for edges green in ph_idx."""
        score = 0.0
        for edge, phases in self._edge_to_green_phases.get(tls, {}).items():
            if ph_idx in phases:
                score += traci.edge.getLastStepHaltingNumber(edge)
                score += ml_boost.get(edge, 0.0)
        return score

    def step(self, t: float, traci):
        # ---- collect per-edge observations ----
        for eid, ff in self._free_flow.items():
            try:
                self._ml_history[eid].append({
                    "speed": traci.edge.getLastStepMeanSpeed(eid),
                    "queue": traci.edge.getLastStepHaltingNumber(eid),
                    "occ":   traci.edge.getLastStepOccupancy(eid),
                    "ff":    ff,
                })
            except Exception:
                pass

        # ---- run ML inference periodically ----
        edge_ml: Dict[str, dict] = {}   # edge -> {congested_5min, confidence}
        if t - self._t_last_ml >= self.ml_interval and _ML_AVAILABLE:
            self._t_last_ml = t
            hour = int((t % 86400) / 3600)
            for eid, hist in self._ml_history.items():
                if len(hist) < self.history_len:
                    continue
                h = list(hist)
                try:
                    result = _predict_congestion({
                        "speed_lag1": h[-1]["speed"], "queue_lag1": h[-1]["queue"], "occ_lag1": h[-1]["occ"],
                        "speed_lag2": h[-2]["speed"], "queue_lag2": h[-2]["queue"], "occ_lag2": h[-2]["occ"],
                        "speed_lag3": h[-3]["speed"], "queue_lag3": h[-3]["queue"], "occ_lag3": h[-3]["occ"],
                        "hour": hour, "free_flow_speed": h[-1]["ff"],
                    })
                    edge_ml[eid] = result
                except Exception:
                    pass

        # ---- ML boost map: edge -> confidence bonus ----
        ml_boost: Dict[str, float] = {
            eid: res.get("confidence", 0.0) * 10
            for eid, res in edge_ml.items()
            if res.get("congested_5min") == 1
        }

        if t - self._t_last_check < self.check_every:
            return
        self._t_last_check = t

        for tls in self.tls_ids:
            cmd = self.mqtt.get_cmd(tls)
            if cmd and "phase" in cmd:
                try:
                    traci.trafficlight.setPhase(tls, int(cmd["phase"]))
                    self._phase_start[tls] = t
                    self._prev_phase[tls] = int(cmd["phase"])
                except Exception:
                    pass
                continue

            try:
                current_phase = traci.trafficlight.getPhase(tls)
                n_phases = self._n_phases.get(tls, 0)
                if n_phases == 0:
                    continue

                if self._prev_phase.get(tls, -1) != current_phase:
                    self._phase_start[tls] = t
                    self._prev_phase[tls] = current_phase

                time_in_phase = t - self._phase_start.get(tls, t)
                if time_in_phase < self.min_green:
                    continue

                phase_scores = {ph: self._phase_score(traci, tls, ph, ml_boost) for ph in range(n_phases)}
                curr_score = phase_scores.get(current_phase, 0)
                best_phase = max(phase_scores, key=lambda ph: phase_scores[ph])
                best_score = phase_scores[best_phase]

                incoming = self.topology["tls"][tls]["incoming_edges"]
                total_q  = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming)
                self._perf_acc["queue"].append(total_q)
                self._perf_acc["speed"].append(_get_mean_speed(traci, incoming))

                if best_phase == current_phase:
                    # Extend if ML predicts congestion on current green direction
                    tls_ml_edges = {
                        eid: edge_ml[eid]
                        for eid, phases in self._edge_to_green_phases.get(tls, {}).items()
                        if current_phase in phases and eid in edge_ml
                        and edge_ml[eid].get("congested_5min") == 1
                    }
                    if tls_ml_edges and time_in_phase < 55:
                        max_conf = max(v.get("confidence", 0) for v in tls_ml_edges.values())
                        ext = self.extend_high if max_conf >= self.high_threshold else (
                              self.extend_medium if max_conf >= self.medium_threshold else 0)
                        if ext > 0:
                            remaining = round(traci.trafficlight.getNextSwitch(tls) - t, 1)
                            if remaining < ext:
                                traci.trafficlight.setPhaseDuration(tls, ext)
                else:
                    if best_score - curr_score >= self.switch_threshold:
                        traci.trafficlight.setPhase(tls, best_phase)
                        self._phase_start[tls] = t
                        self._prev_phase[tls] = best_phase

                self._publish_tls(t, tls, extra={
                    "controller": self.name,
                    "ml_congested_edges": list(ml_boost.keys()),
                    "ml_available": _ML_AVAILABLE,
                })
            except Exception:
                pass

        self._maybe_publish_perf(t)



# ===========================================================================
# 4. EMERGENCY VEHICLE PRIORITIZATION CONTROLLER
# ===========================================================================
class EmergencyController(BaseController):
    """
    Wraps QueueAdaptiveController with emergency-vehicle corridor logic.

    When an ambulance is detected within `approach_distance` metres of a
    signalised intersection:
      1. The current phase is safely yellowed out.
      2. A green corridor phase is forced on the ambulance's approach.
      3. Conflicting directions are held on red.
      4. After the ambulance passes, normal control is restored.
    """
    name = "emergency"

    approach_distance: float = 200.0   # metres - trigger range
    yellow_duration:   int   = 3       # yellow before forcing green (s)
    hold_red_after:    int   = 5       # extra red after ambulance clears (s)

    def setup(self, traci, topology: dict):
        super().setup(traci, topology)
        self._base = QueueAdaptiveController()
        self._base.mqtt_host = self.mqtt_host
        self._base.mqtt_port = self.mqtt_port
        self._base.setup(traci, topology)

        self._emg_lock: Dict[str, float] = {}
        self._emg_vehicles: set = set()
        self._saved_logics: Dict[str, Any] = {}
        self._in_yellow: Dict[str, float] = {}

    def step(self, t: float, traci):
        # ---- track active emergency vehicles ----
        for vid in traci.simulation.getDepartedIDList():
            try:
                if traci.vehicle.getTypeID(vid) == "emergency":
                    self._emg_vehicles.add(vid)
            except Exception:
                pass
        arrived = set(traci.simulation.getArrivedIDList())
        self._emg_vehicles -= arrived

        active_emg: list = []
        for vid in list(self._emg_vehicles):
            try:
                nxt = traci.vehicle.getNextTLS(vid)
                if nxt:
                    tls_id, link_idx, dist, phase_state = nxt[0]
                    active_emg.append({
                        "vid": vid, "tls_id": tls_id, "dist": dist,
                        "edge": traci.vehicle.getRoadID(vid),
                        "speed": round(traci.vehicle.getSpeed(vid), 2),
                        "x": round(traci.vehicle.getPosition(vid)[0], 1),
                        "y": round(traci.vehicle.getPosition(vid)[1], 1),
                    })
            except Exception:
                pass

        if active_emg:
            self.mqtt.publish_emergency(active_emg)

        locked_tls: set = set()
        for emg in active_emg:
            tls_id = emg["tls_id"]
            if emg["dist"] <= self.approach_distance:
                locked_tls.add(tls_id)
                self._handle_emergency_approach(t, traci, tls_id, emg["edge"])

        # ---- release intersections once ambulance is gone ----
        for tls_id in list(self._emg_lock.keys()):
            if tls_id not in locked_tls and t >= self._emg_lock[tls_id]:
                self._restore_normal(traci, tls_id)
                del self._emg_lock[tls_id]

        # ---- run base controller for all intersections ----
        self._base.step(t, traci)

    def _handle_emergency_approach(self, t: float, traci, tls_id: str, emg_edge: str):
        """Force green for the ambulance's approach lane."""
        self._emg_lock[tls_id] = t + self.hold_red_after

        try:
            logic_list = traci.trafficlight.getAllProgramLogics(tls_id)
            if not logic_list:
                return
            logic = logic_list[0]
            phases = logic.phases
            links  = traci.trafficlight.getControlledLinks(tls_id)
            current_phase = traci.trafficlight.getPhase(tls_id)

            # Find the phase that best serves the emergency edge
            best_ph, best_score = 0, -1
            for ph_idx, ph in enumerate(phases):
                score = 0
                state = ph.state
                for i, link_group in enumerate(links):
                    if i < len(state) and state[i].lower() == "g":
                        for link in link_group:
                            if link:
                                lp = link[0].rsplit("_", 1)[0]
                                if lp == emg_edge:
                                    score += 10
                if score > best_score:
                    best_score, best_ph = score, ph_idx

            if best_ph == current_phase:
                # Already green for emergency - just extend it
                try:
                    ph = phases[current_phase]
                    ph_copy = type(ph)(duration=max(ph.duration, 30),
                                      state=ph.state, minDur=ph.minDur,
                                      maxDur=ph.maxDur, name=ph.name)
                    updated = list(phases)
                    updated[current_phase] = ph_copy
                    logic.phases = updated
                    traci.trafficlight.setProgramLogic(tls_id, logic)
                except Exception:
                    pass
            else:
                # Safe yellow then switch
                if self._in_yellow.get(tls_id, 0) <= t:
                    yellow_state = phases[current_phase].state.replace("G", "y").replace("g", "y")
                    try:
                        ph_y = type(phases[0])(duration=self.yellow_duration,
                                               state=yellow_state,
                                               minDur=self.yellow_duration,
                                               maxDur=self.yellow_duration,
                                               name="emg_yellow")
                        new_phases = list(phases) + [ph_y]
                        logic.phases = new_phases
                        traci.trafficlight.setProgramLogic(tls_id, logic)
                        traci.trafficlight.setPhase(tls_id, len(phases))
                        self._in_yellow[tls_id] = t + self.yellow_duration
                        self._saved_logics[tls_id] = (best_ph, list(phases))
                    except Exception:
                        traci.trafficlight.setPhase(tls_id, best_ph)

            self._publish_tls(t, tls_id, extra={
                "emergency_override": True,
                "emergency_edge": emg_edge,
                "target_phase": best_ph,
            })
        except Exception:
            pass

    def _restore_normal(self, traci, tls_id: str):
        """Restore the saved signal program after the ambulance has passed."""
        try:
            if tls_id in self._saved_logics:
                restore_phase, orig_phases = self._saved_logics.pop(tls_id)
                logic_list = traci.trafficlight.getAllProgramLogics(tls_id)
                if logic_list:
                    logic_list[0].phases = orig_phases
                    traci.trafficlight.setProgramLogic(tls_id, logic_list[0])
                    traci.trafficlight.setPhase(tls_id, restore_phase)
        except Exception:
            pass


# ===========================================================================
# 5. REINFORCEMENT LEARNING CONTROLLER (tabular Q-learning, online)
# ===========================================================================
class RLController(BaseController):
    """
    Tabular Q-learning traffic signal controller.

    State  : discretised (total_queue_bin, phase) per intersection
    Actions: 0 = keep current phase, 1 = advance to next phase
    Reward : negative of total halting vehicles (minimise queue)

    Saves/loads Q-table to/from data/ml/rl_qtable.json so learning
    persists across runs.
    """
    name = "rl"

    alpha:       float = 0.1     # learning rate
    gamma:       float = 0.95    # discount factor
    epsilon:     float = 0.15    # exploration rate (epsilon-greedy)
    check_every: int   = 10      # decision interval (s)
    min_green:   int   = 10      # minimum green before action allowed (s)
    max_green:   int   = 60      # maximum green per phase (s)
    q_bins: list = [0, 3, 7, 15, 30, 999]

    QTABLE_PATH = os.path.join(ROOT, "data", "ml", "rl_qtable.json")

    def setup(self, traci, topology: dict):
        super().setup(traci, topology)
        self._q: Dict[str, Dict[str, list]] = {}
        self._last_state:  Dict[str, str] = {}
        self._last_action: Dict[str, int] = {}
        self._phase_start: Dict[str, float] = {t: 0.0 for t in self.tls_ids}
        self._prev_phase:  Dict[str, int]   = {}
        self._t_last:      float = -999.0
        self._load_qtable()

    def _discretise(self, total_queue: int, phase: int) -> str:
        q_bin = sum(1 for b in self.q_bins if total_queue > b)
        return f"{q_bin}_{phase}"

    def _get_q(self, tls: str, state: str) -> list:
        return self._q.setdefault(tls, {}).setdefault(state, [0.0, 0.0])

    def step(self, t: float, traci):
        if t - self._t_last < self.check_every:
            return
        self._t_last = t

        for tls in self.tls_ids:
            try:
                incoming = self.topology["tls"][tls]["incoming_edges"]
                total_q  = sum(traci.edge.getLastStepHaltingNumber(e) for e in incoming)
                current_phase = traci.trafficlight.getPhase(tls)

                if self._prev_phase.get(tls, -1) != current_phase:
                    self._phase_start[tls] = t
                    self._prev_phase[tls] = current_phase

                time_in_phase = t - self._phase_start.get(tls, t)

                # ---- Q-learning update for previous step ----
                reward = -float(total_q)
                state = self._discretise(total_q, current_phase)
                if tls in self._last_state:
                    prev_s = self._last_state[tls]
                    prev_a = self._last_action[tls]
                    q_vals = self._get_q(tls, prev_s)
                    next_max = max(self._get_q(tls, state))
                    q_vals[prev_a] += self.alpha * (reward + self.gamma * next_max - q_vals[prev_a])

                # ---- choose action (epsilon-greedy) ----
                action = 0
                if time_in_phase >= self.min_green:
                    if random.random() < self.epsilon:
                        action = random.randint(0, 1)
                    else:
                        q_vals = self._get_q(tls, state)
                        action = int(q_vals[1] > q_vals[0])

                    if action == 1 and time_in_phase < self.max_green:
                        logic = traci.trafficlight.getAllProgramLogics(tls)
                        if logic:
                            n_phases = len(logic[0].phases)
                            next_ph  = (current_phase + 1) % n_phases
                            traci.trafficlight.setPhase(tls, next_ph)
                            self._phase_start[tls] = t

                self._last_state[tls]  = state
                self._last_action[tls] = action

                self._perf_acc["queue"].append(total_q)
                self._perf_acc["speed"].append(_get_mean_speed(traci, incoming))
                self._publish_tls(t, tls, extra={
                    "controller": self.name,
                    "rl_state": state,
                    "rl_action": action,
                    "epsilon": self.epsilon,
                })
            except Exception:
                pass

        self._maybe_publish_perf(t)

    def teardown(self):
        self._save_qtable()
        super().teardown()

    def _load_qtable(self):
        try:
            if os.path.exists(self.QTABLE_PATH):
                with open(self.QTABLE_PATH) as f:
                    self._q = json.load(f)
                print(f"[RL] Loaded Q-table from {self.QTABLE_PATH}")
        except Exception as exc:
            print(f"[RL] Could not load Q-table: {exc}")

    def _save_qtable(self):
        try:
            os.makedirs(os.path.dirname(self.QTABLE_PATH), exist_ok=True)
            with open(self.QTABLE_PATH, "w") as f:
                json.dump(self._q, f, indent=2)
            print(f"[RL] Q-table saved to {self.QTABLE_PATH}")
        except Exception as exc:
            print(f"[RL] Could not save Q-table: {exc}")


# ===========================================================================
# Quick self-test (no SUMO needed)
# ===========================================================================
if __name__ == "__main__":
    print("=== adaptive_controller.py -- component checks ===")

    sample = {
        "speed_lag1": 5.0, "queue_lag1": 8, "occ_lag1": 0.6,
        "speed_lag2": 5.5, "queue_lag2": 7, "occ_lag2": 0.55,
        "speed_lag3": 6.0, "queue_lag3": 6, "occ_lag3": 0.50,
        "hour": 8, "free_flow_speed": 13.9,
    }
    result = _predict_congestion(sample)
    print(f"predict_congestion({sample['speed_lag1']} m/s ...) -> {result}")

    broker = MQTTBroker(host=None)
    broker.publish_emergency([{"vid": "emg_0", "dist": 50}])
    broker.close()
    print("MQTT broker (no-op): OK")

    ctrl = RLController()
    assert ctrl._discretise(0, 0) != ctrl._discretise(5, 0)
    print("RL discretise: OK")

    print("\nAll checks passed. Use with traffic_sim.py:")
    print("  python traffic_sim.py run emergency --controller adaptive_controller:EmergencyController")
    print("  python traffic_sim.py run peak      --controller adaptive_controller:MLAdaptiveController")
    print("  python traffic_sim.py run peak      --controller adaptive_controller:QueueAdaptiveController")
    print("  python traffic_sim.py run peak      --controller adaptive_controller:RLController")
