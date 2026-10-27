# Smart Traffic Simulation (SUMO + TraCI)
AI457 IoT and Edge Computing, SVNIT Surat. Traffic simulation and data engineering module.

3x3 grid of signalised intersections, seeded Poisson demand, E1/E2 detectors on every approach lane,
five scenarios, CSV / live-JSON / MQTT output, and ML-ready datasets.

## Files
```
traffic_sim.py      the whole pipeline (network, demand, simulation, ML dataset, controller plug-in, CLI)
scenarios.yaml      scenario settings (demand rates, duration, incident, ambulances)
grid.net.xml        the road network (rebuilt automatically if missing)
requirements.txt
data/network_info.json   nodes, edges, signals, entry/exit edges
data/ml/                 ready-made ML files from seed 1 (regenerate with the commands below)
```

## Setup (once)
```bash
pip install -r requirements.txt     # also installs SUMO via the eclipse-sumo wheel
```
(`sumo-gui` needs the full SUMO install from sumo.dlr.de; set `SUMO_HOME`.)

## Run
```bash
python traffic_sim.py run peak                    # one scenario, seed 1  -> data/raw/peak_seed1_fixed/
python traffic_sim.py run emergency --gui         # watch it in sumo-gui
python traffic_sim.py all                         # all scenarios x seeds 1-3, then the ML dataset
python traffic_sim.py all --seeds 1 2 3 4 5 -j 4  # more seeds, 4 runs in parallel
python traffic_sim.py ml --bin 60                 # rebuild ML files (1-min bins)
python traffic_sim.py network --size 3            # only if you want to resize the network
```
Useful flags for `run`: `--seed N`, `--duration S`, `--native` (also keep SUMO's tripinfo/summary/statistics/E1/E2 files),
`--live-json PATH`, `--mqtt-host HOST`. Routes and SUMO's temporary files are created and deleted automatically.
Add `--matrices` to `all` / `ml` to also write METR-LA-style wide tables to `data/ml/matrices/`.

## Scenarios (`scenarios.yaml`)
Demand is veh/h per entry edge (12 entries). Fixed-time KPIs are for seed 1.

| Scenario | Demand | Notes | Travel / wait |
|---|---|---|---|
| normal | 250 steady | free flow | 122 s / 30 s |
| peak | 250 -> 650 -> 250 | rush-hour ramp | 170 s / 62 s |
| congestion | 350 -> 800 -> 350 | incident on `B1B0`, 1200-2100 s | 310 s / 162 s |
| emergency | 350 steady | 4 ambulances with blue-light devices | ambulance 134 s |
| daily_cycle | 30 -> 650 by hour | compressed 24 h (1 h = 600 s), 4 h run | 154 s |

## Plug in a signal controller
```python
# my_ctrl.py (next to traffic_sim.py)
from traffic_sim import BaseController
class Adaptive(BaseController):
    name = "adaptive"                      # becomes the run-folder suffix and `controller` column
    def step(self, t, traci):              # called after every simulation step
        ...                                # traci.trafficlight.setPhase(...), traci.vehicle.getNextTLS(...)
```
`python traffic_sim.py run peak --controller my_ctrl:Adaptive`. Same seed = same demand, so fixed vs adaptive is a fair comparison.

Live MQTT topics: `traffic/sim/system`, `traffic/sim/intersection/<tls_id>`, `traffic/sim/emergency`.

## Outputs
Per run in `data/raw/<scenario>_seed<N>_<controller>/`: `edge_metrics.csv`, `intersection_metrics.csv`,
`system_metrics.csv`, `run_meta.json`, and `emergency_log.csv` (emergency scenario only).
Time is simulation seconds, stamped at the **end** of each interval (default 60 s). `nan` = not applicable / no vehicles.

**edge_metrics.csv**: `time, edge_id, tls_id, is_approach` (1 = ends at a signal; the 36 sensors), `free_flow_speed` (m/s),
`mean_speed`, `vehicle_count`, `occupancy` (%), `halting_vehicles`, `queue_vehicles` and `queue_length_m` (E2 jam, approach only),
`waiting_time`, `flow_veh_h` (E1 loops, approach only).

**intersection_metrics.csv**: `phase`, `phase_state`, `phase_changes`, `total_queue_vehicles`, `total_halting`, `total_waiting_time`, `mean_speed`.

**system_metrics.csv**: `running`, `departed`, `arrived`, `throughput_veh_h`, `mean_speed`, `total_halting`, `mean_waiting_time`, `teleported`.

**emergency_log.csv**: per second per ambulance: `edge, lane_pos, speed, x, y, waiting_time, next_tls, dist_to_tls, tls_state`.

**data/ml/** (5-min bins by default):
- `traffic_long.csv`: `run, scenario, seed, controller, time, edge_id, tls_id, free_flow_speed, flow_veh_h, occupancy, queue_vehicles, halting, waiting_time, vehicle_count, speed_ms, congested`.
  `congested = 1` if `speed_ms < 0.4 x free_flow_speed` or `queue_vehicles > 8` (`--congestion-ratio`).
- `kpi_summary.csv`: one row per run (travel/waiting time, time loss, throughput, queue, ambulance travel time).
- `sensors.csv`, `adjacency.csv`: sensor metadata and graph.
- `matrices/<run>_{speed,flow,occupancy,queue}.csv` (with `--matrices`).

For prediction, split by seed or time rather than randomly. `daily_cycle` gives the longest series; use several seeds or `--bin 60` for more samples.

## Limitations
Synthetic demand with random origin-destination pairs (not calibrated to real data); the incident is a 1 m/s speed throttle,
not a physical blockage; fixed-time programs (90 s cycle) come from SUMO's default generator.
