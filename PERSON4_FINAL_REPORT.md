# Person 4: Backend, Dashboard, Integration, and Evaluation

## Deliverables

- **Backend:** FastAPI service in `backend/main.py`. It exposes health/configuration,
  live-state, KPI analytics, congestion-prediction, and simulation start/status/stop routes.
- **Dashboard:** React/Vite application in `dashboard/`. It displays network activity,
  intersection signal phases, per-approach queues, measured queue length in metres,
  emergency alerts, congestion forecasts, and strategy comparisons. Its city-style map
  renders the SUMO road topology, live TraCI vehicle positions, and signal states.
- **Integration:** The API launches the SUMO/TraCI simulation and reads its atomic live
  JSON output. The simulator publishes per-approach speed, queue, occupancy, and detector
  queue length. It also retains rolling five-minute samples for live XGBoost inference.
- **Evaluation:** Saved multi-run comparison data is available in
  `data/comparison/kpi_comparison.csv`. The dashboard compares the selected controller
  with fixed-time control and visualizes waiting time, travel time, throughput, and queue
  counts. Live E2 detector queue length is shown in metres.

## Demonstration and validation

The following checks were run on the Windows project environment:

- `npm run build` completed successfully for the React dashboard.
- FastAPI health, analytics, and prediction endpoints were exercised over HTTP.
- The API started a SUMO/TraCI run, served live approach telemetry, returned a live
  forecast warm-up response, and stopped the simulator cleanly.
- A 900-second simulated normal-traffic run was completed for seed 1 using both fixed-time
  and queue-adaptive control. The saved run metrics were:

| Metric | Fixed-time | Queue-adaptive | Change |
|---|---:|---:|---:|
| Completed trips | 689 | 705 | +2.3% |
| Mean travel time | 119.57 s | 103.77 s | -13.2% |
| Mean waiting time | 29.51 s | 13.81 s | -53.2% |
| Mean throughput | 2,756 veh/h | 2,820 veh/h | +2.3% |
| Mean total halting | 24.33 veh | 11.09 veh | -54.4% |
| Mean measured approach queue length | 3.25 m | 1.63 m | -49.8% |

These are a **single-seed, 900-second demonstration comparison**, not a statistically
conclusive benchmark. The run artifacts are under the ignored `data/live/` directory.
The bundled model metadata reports PR-AUC 0.808 and F1 0.710 on its held-out seed.

## Run the application

From the project root, install the Python requirements, then start the backend:

```powershell
python -m pip install -r requirements.txt
python -m uvicorn backend.main:app --reload
```

In a second terminal:

```powershell
Set-Location dashboard
npm install
npm run dev
```

Open `http://127.0.0.1:5173`, select a scenario and controller, and click **Run simulation**.
The dashboard then refreshes live data and supports stopping the run. The API docs are at
`http://127.0.0.1:8000/docs`.

Live congestion predictions use the most recent three completed five-minute traffic
windows and predict congestion five minutes ahead. Allow enough simulated time for those
windows to accumulate. When no simulation is running, the dashboard clearly labels
forecasts generated from the saved ML dataset instead.

## Evaluation caveats

The checked-in `data/comparison/kpi_comparison.csv` contains questionable queue-adaptive
rows: for example, one peak run records 282 completed trips while reporting 2,430 veh/h,
and one emergency run records 229 completed trips. Treat the existing multi-seed
comparison report as provisional until those runs are regenerated and inspected. The
900-second demonstration above is separate from those historical aggregate results.

The project’s prediction model is a congestion classifier, not a travel-speed regression
model. Its current prediction horizon is five minutes. Broader 5–15-minute speed
forecasting would require additional model work outside this dashboard integration.
