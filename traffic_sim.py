#!/usr/bin/env python3
"""Smart traffic simulation (SUMO + TraCI) in a single file.

    python traffic_sim.py run peak                 # one scenario (seed 1) -> data/raw/peak_seed1_fixed/
    python traffic_sim.py run emergency --gui      # watch it in sumo-gui
    python traffic_sim.py all                      # every scenario x seeds 1-3, then the ML dataset
    python traffic_sim.py all --seeds 1 2 3 4 5 -j 4   # more seeds, 4 runs in parallel
    python traffic_sim.py ml                       # rebuild ML files from data/raw
    python traffic_sim.py network --size 3         # rebuild the road network (only if you want to resize it)

Plug in your own signal controller (subclass BaseController below):
    python traffic_sim.py run peak --controller my_module:MyController

Scenario settings live in scenarios.yaml. Column descriptions are in README.md.
"""
import argparse, csv, glob, importlib, json, os, random, shutil, subprocess, sys, tempfile, time
import xml.etree.ElementTree as ET
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
NET_FILE = os.path.join(ROOT, "grid.net.xml")
CONFIG_FILE = os.path.join(ROOT, "scenarios.yaml")
TOPO_FILE = os.path.join(ROOT, "data", "network_info.json")
RAW_DIR = os.path.join(ROOT, "data", "raw")
ML_DIR = os.path.join(ROOT, "data", "ml")
sys.path.insert(0, ROOT)          # so --controller my_module:Class works for files next to this script
sys.path.insert(0, os.getcwd())


# =============================================================================== config
def load_config():
    with open(CONFIG_FILE) as f:
        return yaml.safe_load(f)


def get_scenario(name):
    cfg = load_config()
    if name not in cfg["scenarios"]:
        raise SystemExit(f"Unknown scenario '{name}'. Options: {list(cfg['scenarios'])}")
    sc = dict(cfg["scenarios"][name])
    sc["name"] = name
    for k, v in cfg["defaults"].items():
        sc.setdefault(k, v)
    if "hourly_rates" in sc:
        seg = sc["segment_seconds"]
        sc["profile"] = [[i * seg, (i + 1) * seg, r] for i, r in enumerate(sc["hourly_rates"])]
    return sc


# =============================================================================== controller plug-in
class BaseController:
    """Subclass this to plug in adaptive / emergency / RL signal control.

    setup(traci, topology) is called once after the simulation starts;
    step(t, traci) is called after EVERY simulation step (t = sim time in s).
    `name` becomes the run-folder suffix and the `controller` column in the KPI table.
    """
    name = "base"

    def setup(self, traci, topology):
        self.traci, self.topology, self.tls_ids = traci, topology, list(topology["tls"])

    def step(self, t, traci):
        pass


class FixedTimeController(BaseController):
    """Baseline: keep the fixed-time programs generated in grid.net.xml."""
    name = "fixed"


def load_controller(spec):
    if spec in ("fixed", "", None):
        return FixedTimeController()
    mod, cls = spec.split(":")
    return getattr(importlib.import_module(mod), cls)()


# =============================================================================== network
E2_LENGTH, E1_OFFSET = 100.0, 8.0     # queue-zone length / loop offset before the stop line (m)


def read_net():
    import sumolib
    return sumolib.net.readNet(NET_FILE)


def sumo_tool(name):
    """Full path of a SUMO program (duarouter, netgenerate, ...); works even when SUMO is not on PATH (pip install)."""
    import sumolib
    return sumolib.checkBinary(name)


def approach_lanes(net):
    """Yield (tls_id, edge_id, lane_id, lane_length) for each lane entering a signalised junction."""
    for node in net.getNodes():
        if node.getType() != "traffic_light":
            continue
        for edge in node.getIncoming():
            for lane in edge.getLanes():
                yield node.getID(), edge.getID(), lane.getID(), lane.getLength()


def write_detector_file(net, out_xml, e1_out, e2_out, period=60):
    """E1 induction loop + E2 lane-area detector on every approach lane."""
    lines = ["<additional>"]
    for _, _, lane_id, L in approach_lanes(net):
        length = min(E2_LENGTH, L - 2)
        lines.append(f'  <laneAreaDetector id="e2_{lane_id}" lane="{lane_id}" pos="{L - 1 - length:.2f}" '
                     f'length="{length:.2f}" period="{period}" file="{e2_out}" friendlyPos="true"/>')
        lines.append(f'  <inductionLoop id="e1_{lane_id}" lane="{lane_id}" pos="{L - E1_OFFSET:.2f}" '
                     f'period="{period}" file="{e1_out}"/>')
    lines.append("</additional>")
    with open(out_xml, "w") as f:
        f.write("\n".join(lines) + "\n")


def export_topology(net, out_json=TOPO_FILE):
    """Static network description (adjacency, signals, entries/exits) shared with the rest of the team."""
    boundary = {n.getID() for n in net.getNodes() if n.getType() != "traffic_light"}
    info = {"nodes": {}, "edges": {}, "tls": {}, "entry_edges": [], "exit_edges": []}
    for n in net.getNodes():
        x, y = n.getCoord()
        info["nodes"][n.getID()] = {"x": x, "y": y, "tls": n.getType() == "traffic_light"}
    for e in net.getEdges():
        fn, tn = e.getFromNode().getID(), e.getToNode().getID()
        info["edges"][e.getID()] = {"from": fn, "to": tn, "length": e.getLength(), "lanes": e.getLaneNumber(),
                                    "speed_limit": e.getSpeed(), "approach": tn not in boundary}
        if fn in boundary:
            info["entry_edges"].append(e.getID())
        if tn in boundary:
            info["exit_edges"].append(e.getID())
    for t in net.getTrafficLights():
        node = net.getNode(t.getID())
        info["tls"][t.getID()] = {"incoming_edges": [e.getID() for e in node.getIncoming()],
                                  "outgoing_edges": [e.getID() for e in node.getOutgoing()],
                                  "num_signal_links": len(t.getConnections())}
    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(info, f, indent=2)
    return info


def build_network(size=3, block=250, lanes=2, speed=13.89, cycle=90):
    """3x3 (by default) signalised grid; only inner junctions get traffic lights."""
    base = os.path.join(tempfile.mkdtemp(), "base.net.xml")
    subprocess.run([sumo_tool("netgenerate"), "--grid", f"--grid.x-number={size}", f"--grid.y-number={size}",
                    f"--grid.x-length={block}", f"--grid.y-length={block}", f"--grid.attach-length={block}",
                    f"--default.lanenumber={lanes}", f"--default.speed={speed}",
                    "--default-junction-type=priority", "-o", base], check=True)
    inner = ",".join(f"{c}{r}" for c in "ABCDEFGH"[:size] for r in range(size))
    subprocess.run([sumo_tool("netconvert"), "-s", base, "--tls.set", inner, "--tls.default-type", "static",
                    "--tls.cycle.time", str(cycle), "-o", NET_FILE], check=True)
    shutil.rmtree(os.path.dirname(base), ignore_errors=True)
    export_topology(read_net())
    print("Network and topology written.")


def ensure_network():
    """Build the network on first use if grid.net.xml is missing; return the topology dict."""
    if not os.path.exists(NET_FILE):
        print("grid.net.xml not found, building it ...")
        build_network()
    if not os.path.exists(TOPO_FILE):
        export_topology(read_net())
    with open(TOPO_FILE) as f:
        return json.load(f)


# =============================================================================== demand
VTYPES = """  <vType id="car" vClass="passenger" length="4.5" accel="2.6" decel="4.5" maxSpeed="16.7" sigma="0.5" color="0.2,0.6,1"/>
  <vType id="truck" vClass="truck" length="10" accel="1.3" decel="4.0" maxSpeed="13.9" sigma="0.5" color="0.5,0.5,0.5"/>
  <vType id="bus" vClass="bus" length="12" accel="1.2" decel="4.0" maxSpeed="13.9" sigma="0.5" color="1,0.8,0"/>
  <vType id="emergency" vClass="emergency" length="6.5" accel="3.5" decel="6.0" maxSpeed="25" speedFactor="1.2"
         sigma="0.2" guiShape="emergency" color="1,0,0">
    <param key="has.bluelight.device" value="true"/>
  </vType>
"""


def make_trips(sc, net, seed):
    """Seeded Poisson arrivals at every entry edge, plus scheduled ambulances."""
    rng = random.Random(seed)
    entries, exits, n2in, n2out = [], [], {}, {}
    for e in net.getEdges():
        if e.getFromNode().getType() != "traffic_light":
            entries.append(e.getID()); n2in[e.getFromNode().getID()] = e.getID()
        if e.getToNode().getType() != "traffic_light":
            exits.append(e.getID()); n2out[e.getToNode().getID()] = e.getID()
    types, weights = list(sc["vehicle_mix"]), list(sc["vehicle_mix"].values())
    trips = []
    for entry in entries:
        origin = net.getEdge(entry).getFromNode().getID()
        valid_exits = [x for x in exits if net.getEdge(x).getToNode().getID() != origin]
        for start, end, rate in sc["profile"]:
            t = start
            if rate <= 0:
                continue
            while True:
                t += rng.expovariate(rate / 3600.0)
                if t >= end:
                    break
                trips.append((t, entry, rng.choice(valid_exits), rng.choices(types, weights)[0]))
    for em in sc.get("emergency", []):
        trips.append((float(em["depart"]), n2in[em["from_boundary"]], n2out[em["to_boundary"]], "emergency"))
    trips.sort()
    return trips


def generate_routes(sc, seed, workdir):
    """Write trips, run duarouter, return path of the .rou.xml (kept in the temporary workdir)."""
    trips = make_trips(sc, read_net(), seed)
    trips_file, route_file = os.path.join(workdir, "trips.xml"), os.path.join(workdir, "routes.rou.xml")
    with open(trips_file, "w") as f:
        f.write("<routes>\n" + VTYPES)
        ec = 0
        for i, (t, fr, to, vt) in enumerate(trips):
            vid = f"emg_{ec}" if vt == "emergency" else f"{vt}_{i}"
            ec += vt == "emergency"
            f.write(f'  <trip id="{vid}" type="{vt}" depart="{t:.2f}" from="{fr}" to="{to}" '
                    f'departLane="best" departSpeed="max"/>\n')
        f.write("</routes>\n")
    subprocess.run([sumo_tool("duarouter"), "-n", NET_FILE, "--route-files", trips_file, "-o", route_file,
                    "--ignore-errors", "--no-warnings", "--no-step-log"], check=True)
    return route_file, len(trips)


# =============================================================================== live publishing (optional)
class Publisher:
    """Optional live state outputs: JSON file (for the backend) and MQTT (for dashboard / signal control)."""
    def __init__(self, live_json=None, mqtt_host=None, mqtt_port=1883, prefix="traffic/sim"):
        self.live_json, self.prefix, self.client = live_json, prefix, None
        if live_json:
            os.makedirs(os.path.dirname(os.path.abspath(live_json)), exist_ok=True)
        if mqtt_host:
            import paho.mqtt.client as mqtt
            try:
                self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
            except AttributeError:      # paho-mqtt 1.x
                self.client = mqtt.Client()
            self.client.connect(mqtt_host, mqtt_port)
            self.client.loop_start()

    def publish(self, state):
        if self.live_json:
            tmp = self.live_json + ".tmp"
            with open(tmp, "w") as f:
                json.dump(state, f)
            os.replace(tmp, self.live_json)
        if self.client:
            self.client.publish(f"{self.prefix}/system", json.dumps(state["system"]))
            for tls, v in state["intersections"].items():
                self.client.publish(f"{self.prefix}/intersection/{tls}", json.dumps(v))
            if state["emergency"]:
                self.client.publish(f"{self.prefix}/emergency", json.dumps(state["emergency"]))

    def close(self):
        if self.client:
            self.client.loop_stop(); self.client.disconnect()


# =============================================================================== simulation run
def run_simulation(a):
    import sumolib, traci
    import traci.constants as tc

    topo = ensure_network()
    sc = get_scenario(a.scenario)
    dur = a.duration or sc["duration"]
    interval = a.interval or sc["interval"]
    step_len = float(sc["step_length"])
    controller = load_controller(a.controller)
    out = a.out_dir or os.path.join(RAW_DIR, f"{a.scenario}_seed{a.seed}_{controller.name}")
    os.makedirs(out, exist_ok=True)
    work = tempfile.mkdtemp(prefix="trafficsim_")        # routes + SUMO native outputs; deleted at the end

    try:
        net = read_net()
        route, n_veh = generate_routes(sc, a.seed, work)
        det_xml = os.path.join(work, "detectors.add.xml")
        write_detector_file(net, det_xml, os.path.join(work, "e1.xml"), os.path.join(work, "e2.xml"), period=interval)
        cmd = [sumolib.checkBinary("sumo-gui" if a.gui else "sumo"), "-n", NET_FILE, "-r", route, "-a", det_xml,
               "--step-length", str(step_len), "--seed", str(a.seed), "--end", str(dur),
               "--tripinfo-output", os.path.join(work, "tripinfo.xml"),
               "--summary-output", os.path.join(work, "summary.xml"),
               "--statistic-output", os.path.join(work, "statistics.xml"),
               "--no-step-log", "--no-warnings", "--waiting-time-memory", "300"]
        if a.gui:
            cmd += ["--start", "--quit-on-end", "--delay", "50"]
        traci.start(cmd)

        edges = list(topo["edges"])
        approach = [e for e in edges if topo["edges"][e]["approach"]]
        edge_tls = {e: topo["edges"][e]["to"] for e in approach}
        lane_det = defaultdict(list)          # approach edge -> [(e1_id, e2_id)]
        for _, edge, lane, _ in approach_lanes(net):
            lane_det[edge].append((f"e1_{lane}", f"e2_{lane}"))
        e1_ids = [d for v in lane_det.values() for d, _ in v]
        for e in edges:
            traci.edge.subscribe(e, [tc.LAST_STEP_MEAN_SPEED, tc.LAST_STEP_VEHICLE_NUMBER, tc.LAST_STEP_OCCUPANCY,
                                     tc.LAST_STEP_VEHICLE_HALTING_NUMBER, tc.VAR_WAITING_TIME])
        for d in e1_ids:
            traci.inductionloop.subscribe(d, [tc.LAST_STEP_VEHICLE_NUMBER])

        controller.setup(traci, topo)
        pub = Publisher(a.live_json, a.mqtt_host, a.mqtt_port) if (a.live_json or a.mqtt_host) else None

        def fresh_edge():
            return dict(spd=0.0, spd_n=0, cnt=0.0, occ=0.0, halt=0.0, wait=0.0, flow=0, n=0, q_veh=0.0, q_m=0.0, q_n=0)
        acc = {e: fresh_edge() for e in edges}
        tls_changes, last_phase = defaultdict(int), {}
        sysacc = dict(dep=0, arr=0, tele=0, spd_w=0.0, spd_c=0.0, run=0.0, n=0)
        emg_active = set()
        has_emergency = bool(sc.get("emergency"))
        files = {k: open(os.path.join(out, f"{k}.csv"), "w", newline="")
                 for k in ("edge_metrics", "intersection_metrics", "system_metrics")}
        ew, iw, sw = (csv.writer(f, lineterminator="\n") for f in files.values())
        ew.writerow(["time", "edge_id", "tls_id", "is_approach", "free_flow_speed", "mean_speed", "vehicle_count",
                     "occupancy", "halting_vehicles", "queue_vehicles", "queue_length_m", "waiting_time", "flow_veh_h"])
        iw.writerow(["time", "tls_id", "phase", "phase_state", "phase_changes", "total_queue_vehicles",
                     "total_halting", "total_waiting_time", "mean_speed"])
        sw.writerow(["time", "running", "departed", "arrived", "throughput_veh_h", "mean_speed",
                     "total_halting", "mean_waiting_time", "teleported"])
        mw = None
        if has_emergency:                      # ambulance trace only written for scenarios that have ambulances
            files["emergency_log"] = open(os.path.join(out, "emergency_log.csv"), "w", newline="")
            mw = csv.writer(files["emergency_log"], lineterminator="\n")
            mw.writerow(["time", "vehicle_id", "edge", "lane_pos", "speed", "x", "y", "waiting_time",
                         "next_tls", "dist_to_tls", "tls_state"])

        t0 = time.time()
        inc = sc.get("incident")
        for step in range(1, int(dur / step_len) + 1):
            traci.simulationStep()
            t = traci.simulation.getTime()
            controller.step(t, traci)
            if inc:
                lanes = [f"{inc['edge']}_{i}" for i in range(topo["edges"][inc["edge"]]["lanes"])]
                if t == inc["start"]:
                    for ln in lanes:
                        traci.lane.setMaxSpeed(ln, 1.0)            # incident: approach throttled to 1 m/s
                elif t == inc["start"] + inc["duration"]:
                    for ln in lanes:
                        traci.lane.setMaxSpeed(ln, topo["edges"][inc["edge"]]["speed_limit"])   # reopen

            subs = traci.edge.getAllSubscriptionResults()
            e1 = traci.inductionloop.getAllSubscriptionResults()
            sample_q = (step % max(1, int(a.sample_every / step_len)) == 0)
            for e in edges:
                r, s = subs[e], acc[e]
                n_v = r[tc.LAST_STEP_VEHICLE_NUMBER]
                if n_v > 0:
                    s["spd"] += r[tc.LAST_STEP_MEAN_SPEED]; s["spd_n"] += 1
                s["cnt"] += n_v; s["occ"] += r[tc.LAST_STEP_OCCUPANCY]
                s["halt"] += r[tc.LAST_STEP_VEHICLE_HALTING_NUMBER]; s["wait"] += r[tc.VAR_WAITING_TIME]; s["n"] += 1
                sysacc["spd_w"] += r[tc.LAST_STEP_MEAN_SPEED] * n_v; sysacc["spd_c"] += n_v
                if e in lane_det:
                    s["flow"] += sum(e1[d][tc.LAST_STEP_VEHICLE_NUMBER] for d, _ in lane_det[e])
                    if sample_q:
                        s["q_veh"] += sum(traci.lanearea.getJamLengthVehicle(d2) for _, d2 in lane_det[e])
                        s["q_m"] += max(traci.lanearea.getJamLengthMeters(d2) for _, d2 in lane_det[e])
                        s["q_n"] += 1
            sysacc["dep"] += traci.simulation.getDepartedNumber()
            sysacc["arr"] += traci.simulation.getArrivedNumber()
            sysacc["tele"] += traci.simulation.getStartingTeleportNumber()
            sysacc["run"] += traci.vehicle.getIDCount(); sysacc["n"] += 1
            for tls in topo["tls"]:
                ph = traci.trafficlight.getPhase(tls)
                if tls in last_phase and last_phase[tls] != ph:
                    tls_changes[tls] += 1
                last_phase[tls] = ph

            if has_emergency:                  # log ambulances every step while they are in the network
                for vid in traci.simulation.getDepartedIDList():
                    if traci.vehicle.getTypeID(vid) == "emergency":
                        emg_active.add(vid)
                arrived = traci.simulation.getArrivedIDList()
                for vid in list(emg_active):
                    if vid in arrived:
                        emg_active.discard(vid); continue
                    nxt = traci.vehicle.getNextTLS(vid)
                    tl, dist, st = (nxt[0][0], nxt[0][2], nxt[0][3]) if nxt else ("", "", "")
                    x, y = traci.vehicle.getPosition(vid)
                    mw.writerow([t, vid, traci.vehicle.getRoadID(vid), round(traci.vehicle.getLanePosition(vid), 1),
                                 round(traci.vehicle.getSpeed(vid), 2), round(x, 1), round(y, 1),
                                 round(traci.vehicle.getWaitingTime(vid), 1), tl,
                                 dist if dist == "" else round(dist, 1), st])

            if step * step_len % interval == 0:
                flush(traci, t, interval, topo, acc, tls_changes, sysacc, ew, iw, sw)
                for e in edges:
                    acc[e] = fresh_edge()
                tls_changes.clear()
                sysacc.update(dep=0, arr=0, tele=0, spd_w=0.0, spd_c=0.0, run=0.0, n=0)
            elif pub and step * step_len % a.live_every == 0:
                pub.publish(live_state(traci, t, topo, sysacc, emg_active))
            if traci.simulation.getMinExpectedNumber() == 0:
                break
        wall = time.time() - t0
        traci.close()
        for f in files.values():
            f.close()
        if pub:
            pub.close()

        kpis = summarise(os.path.join(work, "tripinfo.xml"))
        meta = dict(scenario=a.scenario, seed=a.seed, controller=controller.name, duration=dur, interval=interval,
                    vehicles=n_veh, wall_seconds=round(wall, 1), kpis=kpis)
        with open(os.path.join(out, "run_meta.json"), "w") as f:
            json.dump(meta, f, indent=2)
        if a.native:                           # keep SUMO's own output files too
            for name in ("tripinfo.xml", "summary.xml", "statistics.xml", "e1.xml", "e2.xml"):
                if os.path.exists(os.path.join(work, name)):
                    shutil.copy(os.path.join(work, name), out)
        print(f"{a.scenario} seed={a.seed} controller={controller.name}: {json.dumps(kpis)}  -> {out}")
    finally:
        shutil.rmtree(work, ignore_errors=True)


def live_state(traci, t, topo, sysacc, emg_active):
    inter = {}
    for tls in topo["tls"]:
        inter[tls] = {"phase": traci.trafficlight.getPhase(tls),
                      "state": traci.trafficlight.getRedYellowGreenState(tls),
                      "next_switch": round(traci.trafficlight.getNextSwitch(tls) - t, 1),
                      "approach_queues": {e: traci.edge.getLastStepHaltingNumber(e)
                                          for e in topo["tls"][tls]["incoming_edges"]}}
    emg = [{"id": v, "edge": traci.vehicle.getRoadID(v), "speed": round(traci.vehicle.getSpeed(v), 1),
            "x": traci.vehicle.getPosition(v)[0], "y": traci.vehicle.getPosition(v)[1]} for v in emg_active]
    return {"time": t, "system": {"time": t, "running": traci.vehicle.getIDCount()},
            "intersections": inter, "emergency": emg}


def flush(traci, t, interval, topo, acc, tls_changes, sysacc, ew, iw, sw):
    nan = float("nan")
    tls_tot = defaultdict(lambda: dict(q=0.0, halt=0.0, wait=0.0, spd_w=0.0, cnt=0.0))
    for e, s in acc.items():
        n = max(s["n"], 1)
        mean_speed = s["spd"] / s["spd_n"] if s["spd_n"] else nan
        info = topo["edges"][e]
        q_n = max(s["q_n"], 1)
        is_app = info["approach"]
        q_veh = s["q_veh"] / q_n if is_app else nan
        ew.writerow([t, e, info["to"] if is_app else "", int(is_app), round(info["speed_limit"], 2),
                     round(mean_speed, 3), round(s["cnt"] / n, 3), round(s["occ"] / n, 3), round(s["halt"] / n, 3),
                     round(q_veh, 3), round(s["q_m"] / q_n, 2) if is_app else nan,
                     round(s["wait"] / n, 3), round(s["flow"] * 3600.0 / interval, 1) if is_app else nan])
        if is_app:
            d = tls_tot[info["to"]]
            d["q"] += q_veh; d["halt"] += s["halt"] / n; d["wait"] += s["wait"] / n
            d["spd_w"] += (mean_speed if mean_speed == mean_speed else 0) * s["cnt"]; d["cnt"] += s["cnt"]
    for tls in topo["tls"]:
        d = tls_tot[tls]
        iw.writerow([t, tls, traci.trafficlight.getPhase(tls), traci.trafficlight.getRedYellowGreenState(tls),
                     tls_changes[tls], round(d["q"], 2), round(d["halt"], 2), round(d["wait"], 2),
                     round(d["spd_w"] / d["cnt"], 3) if d["cnt"] else nan])
    tot_halt = sum(s["halt"] / max(s["n"], 1) for s in acc.values())
    tot_wait = sum(s["wait"] / max(s["n"], 1) for s in acc.values())
    n = max(sysacc["n"], 1)
    running = sysacc["run"] / n
    sw.writerow([t, round(running, 1), sysacc["dep"], sysacc["arr"], round(sysacc["arr"] * 3600.0 / interval, 1),
                 round(sysacc["spd_w"] / sysacc["spd_c"], 3) if sysacc["spd_c"] else nan, round(tot_halt, 1),
                 round(tot_wait / running, 2) if running else 0, sysacc["tele"]])


def summarise(tripinfo_xml):
    """Headline KPIs from SUMO's native tripinfo output."""
    k = {}
    try:
        trips = ET.parse(tripinfo_xml).getroot().findall("tripinfo")
        if trips:
            mean = lambda key: round(sum(float(x.get(key)) for x in trips) / len(trips), 2)
            k.update(completed_trips=len(trips), mean_travel_time_s=mean("duration"),
                     mean_waiting_time_s=mean("waitingTime"), mean_time_loss_s=mean("timeLoss"))
            em = [float(x.get("duration")) for x in trips if x.get("vType") == "emergency"]
            if em:
                k.update(emergency_mean_travel_time_s=round(sum(em) / len(em), 2), emergency_trips=len(em))
    except Exception as exc:
        k["error"] = str(exc)
    return k


# =============================================================================== ML dataset
def build_ml_dataset(bin_s=300, congestion_ratio=0.4, matrices=False):
    """Raw runs -> data/ml/{traffic_long,kpi_summary,sensors,adjacency}.csv (+ METR-LA-style matrices if asked)."""
    import numpy as np, pandas as pd
    topo = ensure_network()
    os.makedirs(ML_DIR, exist_ok=True)
    if matrices:
        os.makedirs(os.path.join(ML_DIR, "matrices"), exist_ok=True)

    longs, kpis = [], []
    for run_path in sorted(glob.glob(os.path.join(RAW_DIR, "*_seed*_*"))):
        run = os.path.basename(run_path)
        f = os.path.join(run_path, "edge_metrics.csv")
        if not os.path.exists(f):
            continue
        scenario, seed, controller = run.rsplit("_", 2)
        df = pd.read_csv(f)
        df = df[df.is_approach == 1].copy()
        df["bin"] = (np.ceil(df["time"] / bin_s) * bin_s).astype(int)
        df["w"] = df["vehicle_count"]
        g = df.groupby(["bin", "edge_id"])
        agg = g.agg(tls_id=("tls_id", "first"), free_flow_speed=("free_flow_speed", "first"),
                    flow_veh_h=("flow_veh_h", "mean"), occupancy=("occupancy", "mean"),
                    queue_vehicles=("queue_vehicles", "mean"), halting=("halting_vehicles", "mean"),
                    waiting_time=("waiting_time", "mean"), vehicle_count=("vehicle_count", "mean")).reset_index()
        spd = g.apply(lambda x: np.average(x.mean_speed.fillna(x.free_flow_speed), weights=x.w + 1e-6),
                      include_groups=False).rename("speed_ms").reset_index()
        agg = agg.merge(spd, on=["bin", "edge_id"])
        agg["congested"] = ((agg.speed_ms < congestion_ratio * agg.free_flow_speed) | (agg.queue_vehicles > 8)).astype(int)
        agg.insert(0, "run", run); agg.insert(1, "scenario", scenario)
        agg.insert(2, "seed", int(seed.replace("seed", ""))); agg.insert(3, "controller", controller)
        agg = agg.rename(columns={"bin": "time"})
        longs.append(agg)
        if matrices:
            for col, name in [("speed_ms", "speed"), ("flow_veh_h", "flow"), ("occupancy", "occupancy"),
                              ("queue_vehicles", "queue")]:
                agg.pivot(index="time", columns="edge_id", values=col).to_csv(
                    os.path.join(ML_DIR, "matrices", f"{run}_{name}.csv"))
        meta = os.path.join(run_path, "run_meta.json")
        if os.path.exists(meta):
            with open(meta) as fh:
                m = json.load(fh)
            sysm = pd.read_csv(os.path.join(run_path, "system_metrics.csv"))
            kpis.append(dict(run=run, scenario=scenario, seed=m["seed"], controller=controller,
                             avg_throughput_veh_h=round(sysm.throughput_veh_h.mean(), 1),
                             avg_queue_total=round(sysm.total_halting.mean(), 2), **m.get("kpis", {})))
    if not longs:
        raise SystemExit("No raw runs found in data/raw. Run `python traffic_sim.py all` first.")
    pd.concat(longs).to_csv(os.path.join(ML_DIR, "traffic_long.csv"), index=False)
    pd.DataFrame(kpis).to_csv(os.path.join(ML_DIR, "kpi_summary.csv"), index=False)

    approach = [e for e, v in topo["edges"].items() if v["approach"]]
    pd.DataFrame([{"edge_id": e, "tls_id": topo["edges"][e]["to"], "from_node": topo["edges"][e]["from"],
                   "length_m": topo["edges"][e]["length"], "lanes": topo["edges"][e]["lanes"],
                   "speed_limit": topo["edges"][e]["speed_limit"]} for e in approach]
                 ).to_csv(os.path.join(ML_DIR, "sensors.csv"), index=False)
    adj = pd.DataFrame(0, index=approach, columns=approach)
    for e1 in approach:
        for e2 in approach:
            if e1 != e2 and topo["edges"][e1]["to"] == topo["edges"][e2]["from"]:
                adj.loc[e1, e2] = 1       # e1 feeds e2 through a junction
    adj.to_csv(os.path.join(ML_DIR, "adjacency.csv"))
    print(f"Wrote ML dataset for {len(longs)} runs to {ML_DIR}")


# =============================================================================== batch
def run_all(a):
    cfg = load_config()["scenarios"]
    scenarios = a.scenarios or list(cfg)
    ensure_network()
    jobs = [(sc, seed) for sc in scenarios for seed in a.seeds]

    def one(job):
        sc, seed = job
        r = subprocess.run([sys.executable, os.path.abspath(__file__), "run", sc, "--seed", str(seed),
                            "--controller", a.controller], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"{sc} seed={seed} failed:\n{(r.stderr or r.stdout)[-1500:]}")
        print(r.stdout.strip().splitlines()[-1], flush=True)

    print(f"{len(jobs)} runs, {a.jobs} at a time ...", flush=True)
    with ThreadPoolExecutor(max_workers=max(1, a.jobs)) as ex:
        list(ex.map(one, jobs))
    if not a.no_ml:
        build_ml_dataset(matrices=a.matrices)


# =============================================================================== CLI
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="run one scenario and write CSV outputs")
    r.add_argument("scenario")
    r.add_argument("--seed", type=int, default=1)
    r.add_argument("--controller", default="fixed", help="'fixed' or module:Class")
    r.add_argument("--duration", type=int, help="override scenario duration (s)")
    r.add_argument("--interval", type=int, help="aggregation interval (s)")
    r.add_argument("--sample-every", type=int, default=5, help="seconds between queue samples")
    r.add_argument("--gui", action="store_true", help="open sumo-gui")
    r.add_argument("--native", action="store_true", help="also keep SUMO's tripinfo/summary/statistics/E1/E2 files")
    r.add_argument("--live-json", help="write live state to this JSON file")
    r.add_argument("--live-every", type=int, default=5)
    r.add_argument("--mqtt-host"); r.add_argument("--mqtt-port", type=int, default=1883)
    r.add_argument("--out-dir")

    b = sub.add_parser("all", help="run scenarios x seeds, then build the ML dataset")
    b.add_argument("--scenarios", nargs="+")
    b.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    b.add_argument("--controller", default="fixed")
    b.add_argument("-j", "--jobs", type=int, default=1, help="runs in parallel")
    b.add_argument("--no-ml", action="store_true")
    b.add_argument("--matrices", action="store_true", help="also write METR-LA-style wide matrices")

    m = sub.add_parser("ml", help="build ML files from data/raw")
    m.add_argument("--bin", type=int, default=300, help="time bin in seconds")
    m.add_argument("--congestion-ratio", type=float, default=0.4)
    m.add_argument("--matrices", action="store_true")

    n = sub.add_parser("network", help="rebuild the road network")
    n.add_argument("--size", type=int, default=3); n.add_argument("--block", type=float, default=250)
    n.add_argument("--lanes", type=int, default=2); n.add_argument("--speed", type=float, default=13.89)
    n.add_argument("--cycle", type=int, default=90)

    a = ap.parse_args()
    if a.cmd == "run":
        run_simulation(a)
    elif a.cmd == "all":
        run_all(a)
    elif a.cmd == "ml":
        build_ml_dataset(a.bin, a.congestion_ratio, a.matrices)
    elif a.cmd == "network":
        build_network(a.size, a.block, a.lanes, a.speed, a.cycle)


if __name__ == "__main__":
    main()