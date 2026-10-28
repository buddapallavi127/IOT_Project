#!/usr/bin/env python3
"""
visualize.py  --  Interactive Visual Simulation Launcher with SUMO GUI & Live MQTT
==================================================================================
Launches any traffic scenario with any signal controller in a visual GUI window,
simultaneously broadcasting and displaying live IoT MQTT telemetry.

Usage:
  python visualize.py                               # Interactive menu mode
  python visualize.py --scenario peak --controller queue_adaptive
  python visualize.py --scenario emergency --controller emergency
  python visualize.py --scenario congestion --controller ml_adaptive
  python visualize.py --scenario normal --controller rl
"""

import argparse
import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import warnings

# Suppress broker background logs
warnings.filterwarnings("ignore")
logging.getLogger("amqtt").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

import paho.mqtt.client as mqtt

ROOT = os.path.dirname(os.path.abspath(__file__))
BROKER_PORT = 1883
MQTT_HOST   = "localhost"
PREFIX      = "traffic/sim"

# Auto-detect Python binary
def get_python_exe():
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
                    [c, "-c", "import sumolib; sumolib.checkBinary('sumo-gui')"],
                    stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                if res.returncode == 0:
                    return c
            except Exception:
                pass
    return "/opt/miniconda3/bin/python" if os.path.exists("/opt/miniconda3/bin/python") else sys.executable

PYTHON = get_python_exe()

SCENARIOS = {
    "1": ("peak", "Rush-hour ramp-up and heavy traffic wave"),
    "2": ("congestion", "Bottleneck incident on edge B1B0 with high volume"),
    "3": ("emergency", "Ambulances requesting green-wave corridor priority"),
    "4": ("normal", "Free-flowing steady traffic baseline"),
    "5": ("daily_cycle", "Compressed 24-hour full day cycle"),
}

CONTROLLERS = {
    "1": ("queue_adaptive", "adaptive_controller:QueueAdaptiveController", "Queue-Adaptive (Dynamic green extensions)"),
    "2": ("ml_adaptive", "adaptive_controller:MLAdaptiveController", "ML-Adaptive (Congestion-prediction lookahead)"),
    "3": ("rl", "adaptive_controller:RLController", "Reinforcement Learning (Q-learning agent)"),
    "4": ("emergency", "adaptive_controller:EmergencyController", "Emergency Priority (Ambulance preemptive green)"),
    "5": ("fixed", "fixed", "Fixed-Time Baseline (90s pre-timed cycles)"),
}


# ─────────────────────────────────────────────── broker thread
def run_broker():
    async def _run():
        from amqtt.broker import Broker
        config = {
            "listeners": {"default": {"type": "tcp", "bind": f"0.0.0.0:{BROKER_PORT}"}},
            "sys_interval": 0,
            "auth": {"allow-anonymous": True},
            "topic-check": {"enabled": False},
        }
        broker = Broker(config)
        await broker.start()
        await asyncio.Event().wait()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run())


# ─────────────────────────────────────────────── live subscriber
def make_subscriber():
    def on_connect(client, userdata, flags, rc, *args):
        client.subscribe(f"{PREFIX}/#")

    def on_message(client, userdata, msg):
        try:
            payload = json.loads(msg.payload.decode())
            topic = msg.topic

            if "/intersection/" in topic:
                tls = topic.split("/")[-1]
                phase = payload.get("phase", "?")
                state = payload.get("state", "")
                q = payload.get("total_queue", sum(payload.get("approach_queues", {}).values()))
                spd = payload.get("mean_speed", 0.0)
                spd_str = f"{spd:>4.1f}" if isinstance(spd, (int, float)) else f"{spd:>4}"
                next_sw = payload.get("next_switch", "?")
                print(f"  🚦 [{tls:3s}] phase={phase} state={state:<8} queue={q:>2} veh  speed={spd_str} m/s  switch_in={next_sw:>4}s")

            elif "/emergency" in topic:
                vehs = payload if isinstance(payload, list) else [payload]
                for v in vehs:
                    vid = v.get("id", v.get("vid", "?"))
                    edge = v.get("edge", "?")
                    spd = v.get("speed", "?")
                    print(f"  🚑 AMBULANCE PRIORITY: id={vid} on edge={edge} speed={spd} m/s")

            elif "/system" in topic:
                run = payload.get("running", 0)
                arr = payload.get("arrived", 0)
                halt = payload.get("total_halting", 0)
                spd = payload.get("mean_speed", 0.0)
                spd_str = f"{spd:>4.1f}" if isinstance(spd, (int, float)) else f"{spd:>4}"
                t = payload.get("sim_time", payload.get("time", "?"))
                print(f"\n📊 [SYSTEM t={t}s] Active Vehicles: {run:>3} | Arrived: {arr:>3} | Halting: {halt:>3} | Avg Speed: {spd_str} m/s")

        except Exception:
            pass

    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="gui_sub")
    except AttributeError:
        c = mqtt.Client(client_id="gui_sub")

    c.on_connect = on_connect
    c.on_message = on_message
    return c


def interactive_selection():
    print("=" * 68)
    print("  🚦 Visual Traffic Simulation Launcher (SUMO-GUI + Live IoT)")
    print("=" * 68)
    print("\nSelect a Scenario to Simulate:")
    for k, (name, desc) in SCENARIOS.items():
        print(f"  [{k}] {name.upper():<12} - {desc}")
    sc_choice = input("\nEnter choice [1-5] (default: 1): ").strip() or "1"
    scenario = SCENARIOS.get(sc_choice, SCENARIOS["1"])[0]

    print("\nSelect a Signal Controller:")
    for k, (key, _, desc) in CONTROLLERS.items():
        print(f"  [{k}] {desc}")
    ctrl_choice = input("\nEnter choice [1-5] (default: 1): ").strip() or "1"
    ctrl_key, ctrl_arg, _ = CONTROLLERS.get(ctrl_choice, CONTROLLERS["1"])

    return scenario, ctrl_arg, ctrl_key


def main():
    parser = argparse.ArgumentParser(description="Launch SUMO visual simulation with MQTT telemetry")
    parser.add_argument("--scenario", choices=["peak", "congestion", "emergency", "normal", "daily_cycle"],
                        help="Scenario name")
    parser.add_argument("--controller", choices=["queue_adaptive", "ml_adaptive", "rl", "emergency", "fixed"],
                        help="Signal controller")
    parser.add_argument("--no-mqtt", action="store_true", help="Disable MQTT broker & telemetry stream")
    parser.add_argument("--duration", type=int, default=600, help="Duration in seconds (default: 600)")
    args = parser.parse_args()

    # Determine scenario and controller
    if args.scenario and args.controller:
        scenario = args.scenario
        ctrl_map = {v[0]: v[1] for v in CONTROLLERS.values()}
        ctrl_arg = ctrl_map[args.controller]
        ctrl_key = args.controller
    else:
        # Interactive mode
        scenario, ctrl_arg, ctrl_key = interactive_selection()

    print("\n" + "=" * 68)
    print(f"  🚀 Launching Simulation:")
    print(f"     Scenario   : {scenario.upper()}")
    print(f"     Controller : {ctrl_key.upper()} ({ctrl_arg})")
    print(f"     Interface  : SUMO GUI (2D Visualization)")
    print(f"     Duration   : {args.duration}s")
    print(f"     MQTT IoT   : {'Enabled on port 1883' if not args.no_mqtt else 'Disabled'}")
    print("=" * 68 + "\n")

    sub = None
    if not args.no_mqtt:
        # Start broker
        b_thread = threading.Thread(target=run_broker, daemon=True)
        b_thread.start()
        time.sleep(1.5)

        # Start subscriber
        sub = make_subscriber()
        try:
            sub.connect(MQTT_HOST, BROKER_PORT, keepalive=60)
            sub.loop_start()
            print("[IoT] MQTT broker ready & streaming telemetry in background.\n")
        except Exception as e:
            print(f"[WARN] Could not connect subscriber: {e}")

    # Launch GUI simulation command
    cmd = [
        PYTHON, os.path.join(ROOT, "traffic_sim.py"),
        "run", scenario,
        "--gui",
        "--duration", str(args.duration),
        "--controller", ctrl_arg,
    ]
    if not args.no_mqtt:
        cmd += ["--mqtt-host", MQTT_HOST, "--mqtt-port", str(BROKER_PORT), "--live-every", "5"]

    print(f"[INFO] Opening SUMO-GUI window on your screen...")
    print(f"       Tip: In the SUMO window, set 'Delay (ms)' to 50 for smooth viewing.\n")
    
    subprocess.run(cmd)

    if sub:
        time.sleep(1)
        sub.loop_stop()
        sub.disconnect()

    print("\n" + "=" * 68)
    print("  ✅ Simulation run completed!")
    print("=" * 68)


if __name__ == "__main__":
    main()
