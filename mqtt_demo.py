#!/usr/bin/env python3
"""
mqtt_demo.py  --  Live MQTT IoT demo (no mosquitto binary needed)
=================================================================
1. Starts amqtt MQTT broker in a background thread
2. Subscribes via paho-mqtt and prints verified IoT payloads live
3. Runs traffic simulation with --mqtt-host localhost on 'peak' traffic
   → demonstrates real-time queues, speeds, signal phases, and system telemetry

Run:
    python mqtt_demo.py
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
import warnings

# Suppress harmless warnings and broker internal logging
warnings.filterwarnings("ignore")
logging.getLogger("amqtt").setLevel(logging.ERROR)
logging.getLogger("asyncio").setLevel(logging.ERROR)

import paho.mqtt.client as mqtt

BROKER_PORT = 1883
MQTT_HOST   = "localhost"
PREFIX      = "traffic/sim"   # matches traffic_sim.py's Publisher prefix
ROOT        = os.path.dirname(os.path.abspath(__file__))
PYTHON      = "/opt/miniconda3/bin/python" if os.path.exists("/opt/miniconda3/bin/python") else sys.executable


# ─────────────────────────────────────────────── broker thread
def run_broker_thread():
    """Run amqtt broker in its own event loop (background thread)."""
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
        print(f"[BROKER] MQTT broker ready on port {BROKER_PORT}")
        await asyncio.Event().wait()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(_run())


# ─────────────────────────────────────────────── subscriber
received_count = [0]
printed_samples = set()

def make_subscriber():
    """Connect paho subscriber, print messages, return client."""
    def on_connect(client, userdata, flags, rc, *args):
        client.subscribe(f"{PREFIX}/#")
        print(f"[SUB]    Listening on {PREFIX}/# (streaming live events)...\n")

    def on_message(client, userdata, msg):
        received_count[0] += 1
        try:
            payload = json.loads(msg.payload.decode())
            topic   = msg.topic

            # Print one raw sample payload for each topic category for validation
            category = topic.split("/")[2] if len(topic.split("/")) > 2 else topic
            if category not in printed_samples:
                printed_samples.add(category)
                print(f"\n[RAW MQTT SAMPLE] Topic: {topic}")
                print(f"                  Payload: {json.dumps(payload)}\n")

            # 1. Intersection Telemetry: traffic/sim/intersection/<tls_id>
            if "/intersection/" in topic:
                tls   = topic.split("/")[-1]
                phase = payload.get("phase", "?")
                state = payload.get("state", "")
                # Read total_queue or sum approach_queues
                q = payload.get("total_queue")
                if q is None:
                    q = sum(payload.get("approach_queues", {}).values())
                spd = payload.get("mean_speed", "?")
                spd_str = f"{spd:>4.1f}" if isinstance(spd, (int, float)) else f"{spd:>4}"
                t = payload.get("sim_time", payload.get("time", "?"))
                next_sw = payload.get("next_switch", "?")
                print(f"  🚦 [{tls:3s}] phase={phase} state={state:<8} "
                      f"queue={q:>2} veh  "
                      f"speed={spd_str} m/s  "
                      f"next_switch={next_sw:>4}s  t={t}s")

            # 2. Emergency Vehicle Telemetry: traffic/sim/emergency
            elif "/emergency" in topic:
                vehs = payload if isinstance(payload, list) else [payload]
                for v in vehs:
                    vid = v.get("id", v.get("vid", "?"))
                    edge = v.get("edge", "?")
                    spd = v.get("speed", "?")
                    x, y = v.get("x", "?"), v.get("y", "?")
                    print(f"  🚑 EMRG  id={vid}  edge={edge}  "
                          f"speed={spd} m/s  pos=({x},{y})")

            # 3. Network System Telemetry: traffic/sim/system
            elif "/system" in topic:
                run  = payload.get("running", "?")
                arr  = payload.get("arrived", 0)
                halt = payload.get("total_halting", 0)
                spd  = payload.get("mean_speed", "?")
                spd_str = f"{spd:>4.1f}" if isinstance(spd, (int, float)) else f"{spd:>4}"
                t    = payload.get("sim_time", payload.get("time", "?"))
                print(f"  📊 SYS   running={run:>3}  arrived={arr:>3}  "
                      f"halting={halt:>3}  mean_speed={spd_str} m/s  t={t}s")

        except Exception as e:
            print(f"[ERR] subscriber parsing error: {e}")

    try:
        c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="demo_sub")
    except AttributeError:
        c = mqtt.Client(client_id="demo_sub")

    c.on_connect = on_connect
    c.on_message = on_message
    return c


# ─────────────────────────────────────────────── main
def main():
    print("=" * 65)
    print("  🚦 IoT MQTT Live Demo — Adaptive Traffic Signal Control")
    print("=" * 65)

    # 1. Start broker in background thread
    t = threading.Thread(target=run_broker_thread, daemon=True)
    t.start()
    time.sleep(2)

    # 2. Connect subscriber
    sub = make_subscriber()
    connected = False
    for _ in range(10):
        try:
            sub.connect(MQTT_HOST, BROKER_PORT, keepalive=60)
            connected = True
            break
        except Exception:
            time.sleep(0.5)

    if not connected:
        print("[ERROR] Could not connect to broker!")
        sys.exit(1)

    sub.loop_start()
    time.sleep(2)

    print("[TEST]   Subscriber ready. Starting 'peak' simulation with QueueAdaptiveController...\n")

    # 3. Run simulation with peak demand so queues naturally build and clear
    cmd = [
        PYTHON, os.path.join(ROOT, "traffic_sim.py"),
        "run", "peak",
        "--duration", "400",
        "--controller", "adaptive_controller:QueueAdaptiveController",
        "--mqtt-host", MQTT_HOST,
        "--mqtt-port", str(BROKER_PORT),
        "--live-every", "10",
    ]
    subprocess.run(cmd)

    time.sleep(2)
    sub.loop_stop()
    sub.disconnect()

    print(f"\n{'=' * 65}")
    print(f"  ✅ Done!  Total verified IoT messages received: {received_count[0]}")
    print(f"{'=' * 65}")


if __name__ == "__main__":
    main()
