import { useCallback, useEffect, useMemo, useState } from "react";

const API = "/api";
const REFRESH_MS = 2500;
const controllerNames = {
  fixed: "Fixed time",
  queue_adaptive: "Queue adaptive",
  ml_adaptive: "ML adaptive",
  emergency: "Emergency priority",
  rl: "Reinforcement learning",
};

async function request(path, options) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(body.detail || `Request failed (${response.status})`);
  return body;
}

function Metric({ label, value, suffix = "", accent = "", icon }) {
  return (
    <article className={`metric-card ${accent}`}>
      <span className="metric-label"><i>{icon}</i>{label}</span>
      <strong>{value ?? "—"}<small>{value == null ? "" : suffix}</small></strong>
    </article>
  );
}

function getMapPoint(node, nodes) {
  const coordinates = Object.values(nodes).map((value) => value);
  const minX = Math.min(...coordinates.map((value) => value.x));
  const maxX = Math.max(...coordinates.map((value) => value.x));
  const minY = Math.min(...coordinates.map((value) => value.y));
  const maxY = Math.max(...coordinates.map((value) => value.y));
  const x = 82 + ((node.x - minX) / (maxX - minX || 1)) * 836;
  const y = 918 - ((node.y - minY) / (maxY - minY || 1)) * 836;
  return { x, y };
}

function NetworkMap({ topology, live, running }) {
  const nodes = topology?.nodes || {};
  const edges = topology?.edges || {};
  const intersections = live?.intersections || {};
  const approaches = live?.approaches || {};
  const vehicles = live?.vehicles || [];
  const extents = useMemo(() => {
    const points = Object.values(nodes).map((node) => getMapPoint(node, nodes));
    if (!points.length) return { x: [], y: [] };
    const xs = [...new Set(points.map((point) => point.x.toFixed(1)))].map(Number).sort((a, b) => a - b);
    const ys = [...new Set(points.map((point) => point.y.toFixed(1)))].map(Number).sort((a, b) => a - b);
    return { x: xs, y: ys };
  }, [nodes]);

  const roads = useMemo(() => {
    const grouped = new Map();
    Object.entries(edges).forEach(([id, edge]) => {
      if (!nodes[edge.from] || !nodes[edge.to]) return;
      const key = [edge.from, edge.to].sort().join(":");
      const queue = approaches[id];
      const previous = grouped.get(key);
      if (!previous || (queue?.queue_vehicles || 0) > (previous.queue?.queue_vehicles || 0)) {
        grouped.set(key, { id, edge, queue });
      }
    });
    return [...grouped.values()].map((road) => ({
      ...road,
      from: getMapPoint(nodes[road.edge.from], nodes),
      to: getMapPoint(nodes[road.edge.to], nodes),
    }));
  }, [edges, nodes, approaches]);

  const blocks = useMemo(() => {
    const result = [];
    for (let row = 0; row < extents.y.length - 1; row += 1) {
      for (let col = 0; col < extents.x.length - 1; col += 1) {
        const left = extents.x[col] + 25;
        const right = extents.x[col + 1] - 25;
        const top = extents.y[row] + 25;
        const bottom = extents.y[row + 1] - 25;
        result.push({ x: left, y: top, width: right - left, height: bottom - top, key: `${row}-${col}` });
      }
    }
    return result;
  }, [extents]);

  const busiest = Object.entries(approaches)
    .map(([id, value]) => ({ id, ...value }))
    .sort((a, b) => Number(b.queue_vehicles || 0) - Number(a.queue_vehicles || 0))
    .slice(0, 5);

  return (
    <section className="map-panel panel">
      <div className="map-toolbar">
        <div className="map-title">
          <span className="map-kicker"><i className={`live-dot ${running ? "on" : ""}`} /> {running ? "LIVE NETWORK" : "NETWORK MAP"}</span>
          <h2>City traffic overview</h2>
          <p>{running ? "SUMO grid · 9 signalized intersections · live sensor overlay" : "SUMO grid · 9 signalized intersections · last TraCI snapshot"}</p>
        </div>
        <div className="map-toolbar-right">
          <span className="map-time">{live ? `SIM TIME  ${formatSimTime(live.time)}` : "AWAITING TELEMETRY"}</span>
          <span className="map-live-tag">
            {running
              ? `${live?.vehicle_count ?? vehicles.length} LIVE VEHICLES`
              : live
                ? `${live.vehicle_count ?? vehicles.length} VEHICLES · LAST SNAPSHOT`
                : "AWAITING TELEMETRY"}
          </span>
        </div>
      </div>
      {Object.keys(nodes).length ? (
        <div className="map-viewport">
          <svg className="city-map" viewBox="0 0 1000 1000" role="img" aria-label="Live schematic map of the SUMO road network">
            <defs>
              <pattern id="map-grid" width="36" height="36" patternUnits="userSpaceOnUse">
                <path d="M 36 0 L 0 0 0 36" fill="none" stroke="#1b2b31" strokeWidth="1" />
              </pattern>
              <filter id="vehicle-glow" x="-100%" y="-100%" width="300%" height="300%">
                <feGaussianBlur stdDeviation="3" result="blur" />
                <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
              </filter>
            </defs>
            <rect width="1000" height="1000" fill="#101b20" />
            <rect width="1000" height="1000" fill="url(#map-grid)" />
            <g className="city-blocks">
              {blocks.map((block, index) => (
                <g key={block.key}>
                  <rect x={block.x} y={block.y} width={block.width} height={block.height} rx="9" />
                  <rect className="building" x={block.x + 12} y={block.y + 13} width={block.width * 0.39} height={block.height * 0.35} rx="4" />
                  <rect className="building alt" x={block.x + block.width * 0.55} y={block.y + 13} width={block.width * 0.34} height={block.height * 0.28} rx="4" />
                  <rect className="building alt" x={block.x + 12} y={block.y + block.height * 0.53} width={block.width * 0.30} height={block.height * 0.33} rx="4" />
                  <rect className="building" x={block.x + block.width * 0.45} y={block.y + block.height * 0.48} width={block.width * 0.43} height={block.height * 0.39} rx="4" />
                  {(index % 3 === 0) && <text className="district-label" x={block.x + block.width / 2} y={block.y + block.height / 2}>{["NORTH QUARTER", "MARKET DISTRICT", "RIVER PARK", "CENTRAL"][index % 4]}</text>}
                </g>
              ))}
            </g>
            <g className="road-layer">
              {roads.map((road) => {
                const path = `M ${road.from.x} ${road.from.y} L ${road.to.x} ${road.to.y}`;
                const queue = Number(road.queue?.queue_vehicles || 0);
                const trafficClass = !road.queue ? "road-flow quiet" : queue >= 10 ? "road-flow heavy" : queue >= 4 ? "road-flow busy" : "road-flow clear";
                return (
                  <g key={road.id}>
                    <path className="road-casing" d={path} />
                    <path className="road-surface" d={path} />
                    <path className="road-center" d={path} />
                    {road.queue && <path className={trafficClass} d={path} />}
                  </g>
                );
              })}
            </g>
            <g className="junction-layer">
              {Object.entries(nodes).filter(([, node]) => node.tls).map(([id, node]) => {
                const point = getMapPoint(node, nodes);
                const state = String(intersections[id]?.state || "");
                const hasGreen = state.toLowerCase().includes("g");
                const hasAmber = state.toLowerCase().includes("y");
                const queue = Number(intersections[id]?.total_queue || 0);
                const status = queue >= 20 ? "congested" : queue >= 8 ? "busy" : "clear";
                return (
                  <g className={`junction ${status}`} key={id} transform={`translate(${point.x} ${point.y})`}>
                    <circle className="junction-halo" r="31" />
                    <rect className="junction-pad" x="-24" y="-24" width="48" height="48" rx="8" />
                    <rect className="signal-housing" x="-8" y="-13" width="16" height="29" rx="6" />
                    <circle className={`map-light red ${!hasGreen && !hasAmber && state ? "active" : ""}`} cx="0" cy="-7" r="3" />
                    <circle className={`map-light amber ${!hasGreen && hasAmber ? "active" : ""}`} cx="0" cy="1" r="3" />
                    <circle className={`map-light green ${hasGreen ? "active" : ""}`} cx="0" cy="9" r="3" />
                    <text className="junction-label" x="0" y="39">{id}</text>
                    <text className="junction-queue" x="0" y="51">{queue} QUEUED</text>
                  </g>
                );
              })}
            </g>
            <g className="vehicle-layer">
              {vehicles.map((vehicle) => {
                const point = getMapPoint(vehicle, nodes);
                return (
                  <g
                    className={`map-vehicle ${vehicle.emergency ? "emergency" : ""} ${!running ? "stale" : ""}`}
                    key={vehicle.id}
                    transform={`translate(${point.x} ${point.y}) rotate(${-Number(vehicle.angle || 0)})`}
                    filter={vehicle.emergency ? "url(#vehicle-glow)" : undefined}
                  >
                    {vehicle.emergency && <circle className="emergency-pulse" r="10" />}
                    <rect x="-2.7" y="-5.5" width="5.4" height="11" rx="2" />
                    <circle className="vehicle-light" cx="0" cy="-3.3" r="1" />
                  </g>
                );
              })}
            </g>
            {!running && !vehicles.length && (
              <g className="map-idle">
                <rect x="337" y="460" width="326" height="78" rx="10" />
                <text x="500" y="492">NETWORK READY</text>
                <text className="idle-subtitle" x="500" y="515">Start a simulation to stream live vehicles</text>
              </g>
            )}
          </svg>
          <div className="map-compass"><span>N</span><i>↑</i></div>
          <div className="map-scale"><span /> 250 m</div>
        </div>
      ) : (
        <div className="empty-state"><strong>Network map unavailable</strong><p>Check that the SUMO topology file exists.</p></div>
      )}
      <div className="map-footer">
        <div className="map-legend">
          <span><i className="legend-line clear" />Free flow</span>
          <span><i className="legend-line busy" />Building queue</span>
          <span><i className="legend-line heavy" />Congested</span>
          <span><i className="legend-car" />Vehicle</span>
          <span><i className="legend-car emergency" />Emergency</span>
        </div>
        <span className="map-disclaimer">Vehicle positions from TraCI · schematic street geometry</span>
      </div>
      {busiest.length > 0 && (
        <div className="approach-strip">
          <span className="approach-title">TOP APPROACH QUEUES</span>
          {busiest.map((approach) => (
            <div className="approach-chip" key={approach.id}>
              <strong>{approach.id}</strong>
              <span>{Number(approach.queue_vehicles || 0)} veh</span>
              <b>{Number(approach.queue_length_m || 0).toFixed(0)} m</b>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}

function formatSimTime(value) {
  const seconds = Math.max(0, Math.floor(Number(value) || 0));
  const hours = String(Math.floor(seconds / 3600)).padStart(2, "0");
  const minutes = String(Math.floor((seconds % 3600) / 60)).padStart(2, "0");
  const remainder = String(seconds % 60).padStart(2, "0");
  return `${hours}:${minutes}:${remainder}`;
}

function App() {
  const [config, setConfig] = useState({ scenarios: [], controllers: [] });
  const [topology, setTopology] = useState(null);
  const [scenario, setScenario] = useState("peak");
  const [controller, setController] = useState("queue_adaptive");
  const [live, setLive] = useState(null);
  const [analytics, setAnalytics] = useState({ summaries: [], scenarios: [] });
  const [forecast, setForecast] = useState(null);
  const [sim, setSim] = useState({ running: false });
  const [error, setError] = useState("");
  const [busy, setBusy] = useState(false);

  const refresh = useCallback(async () => {
    try {
      const [state, status] = await Promise.all([
        request("/live").catch((err) => {
          if (!String(err.message).includes("No live simulation state")) throw err;
          return null;
        }),
        request("/simulation/status"),
      ]);
      setLive(state);
      setSim(status);
      setError("");
    } catch (err) {
      setError(err.message);
    }
  }, []);

  useEffect(() => {
    request("/config").then((value) => {
      setConfig(value);
      if (value.scenarios.includes(scenario)) return;
      setScenario(value.scenarios[0] || "peak");
    }).catch((err) => setError(err.message));
    refresh();
    const timer = window.setInterval(refresh, REFRESH_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  useEffect(() => {
    let active = true;
    let retryTimer;
    const loadTopology = () => {
      request("/topology")
        .then((value) => { if (active) setTopology(value); })
        .catch((err) => {
          if (!active) return;
          setError(err.message);
          retryTimer = window.setTimeout(loadTopology, REFRESH_MS);
        });
    };
    loadTopology();
    return () => {
      active = false;
      window.clearTimeout(retryTimer);
    };
  }, []);

  useEffect(() => {
    let active = true;
    request(`/analytics?scenario=${encodeURIComponent(scenario)}`)
      .then((value) => { if (active) setAnalytics(value); })
      .catch((err) => { if (active) setError(err.message); });
    return () => { active = false; };
  }, [scenario]);

  useEffect(() => {
    let active = true;
    request(`/predictions?scenario=${encodeURIComponent(scenario)}`)
      .then((value) => { if (active) setForecast(value); })
      .catch((err) => { if (active) setForecast({ error: err.message, predictions: [] }); });
    return () => { active = false; };
  }, [scenario, live?.time, sim.running]);

  const selectedSummary = useMemo(() => {
    const rows = analytics.summaries || [];
    const adaptive = rows.find((row) => row.controller === controller);
    const fixed = rows.find((row) => row.controller === "fixed");
    return { current: adaptive, baseline: fixed };
  }, [analytics, controller]);

  const start = async () => {
    setBusy(true);
    try {
      await request("/simulation/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scenario, controller, seed: 1 }),
      });
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const stop = async () => {
    setBusy(true);
    try {
      await request("/simulation/stop", { method: "POST" });
      await refresh();
    } catch (err) {
      setError(err.message);
    } finally {
      setBusy(false);
    }
  };

  const system = live?.system || {};
  const intersections = Object.entries(live?.intersections || {});
  const alerts = live?.emergency || [];
  const approaches = live?.approaches || {};
  const totalQueueLength = Object.values(approaches).reduce((sum, approach) => sum + Number(approach.queue_length_m || 0), 0);
  const current = selectedSummary.current;
  const baseline = selectedSummary.baseline;

  return (
    <main className="shell">
      <header className="topbar">
        <a className="brand" href="#" aria-label="FlowState home">
          <span className="brand-mark"><i /></span>
          <span>METRO<span className="brand-dot">FLOW</span><small>INTELLIGENT TRAFFIC CONTROL</small></span>
        </a>
        <nav className="top-nav">
          <a className="nav-active" href="#overview">NETWORK OVERVIEW</a>
          <a href="#performance">PERFORMANCE</a>
          <a href="#incidents">INCIDENTS</a>
        </nav>
        <div className="top-status">
          <span className="connection-indicator"><i className={`live-dot ${sim.running ? "on" : ""}`} />{sim.running ? "SYSTEM LIVE" : "SYSTEM READY"}</span>
          <span className="clock">SUMO / TRACI</span>
        </div>
      </header>

      <section className="intro">
        <div>
          <p className="eyebrow">OPERATIONS CENTER <span>/</span> URBAN MOBILITY</p>
          <h1>Network <em>overview</em></h1>
          <p className="subtitle">A live view of traffic flow, intersection health, and control performance.</p>
        </div>
        <div className="run-controls">
          <label>TRAFFIC SCENARIO
            <select value={scenario} onChange={(event) => setScenario(event.target.value)}>
              {(config.scenarios.length ? config.scenarios : ["peak", "normal"]).map((name) =>
                <option key={name} value={name}>{name.replaceAll("_", " ")}</option>)}
            </select>
          </label>
          <label>CONTROL STRATEGY
            <select value={controller} onChange={(event) => setController(event.target.value)}>
              {(config.controllers.length ? config.controllers : Object.keys(controllerNames)).map((name) =>
                <option key={name} value={name}>{controllerNames[name] || name}</option>)}
            </select>
          </label>
          {sim.running
            ? <button className="button button-stop" disabled={busy} onClick={stop}><i className="stop-square" /> Stop run</button>
            : <button className="button" disabled={busy} onClick={start}>{busy ? "Starting…" : <><i className="play-triangle" /> Launch simulation</>}</button>}
        </div>
      </section>

      {error && <div className="notice" role="status">{error}</div>}

      <section className="metrics-grid" aria-label="Live network metrics">
        <Metric label="VEHICLES IN NETWORK" value={system.running ?? 0} accent="metric-primary" icon="⌁" />
        <Metric label="TOTAL HALTING" value={system.total_halting ?? 0} suffix=" veh" icon="Ⅱ" />
        <Metric label="DETECTED QUEUE" value={live ? totalQueueLength.toFixed(1) : "—"} suffix=" m" icon="↔" />
        <Metric label="AVERAGE SPEED" value={system.mean_speed ?? "—"} suffix=" m/s" icon="↗" />
        <Metric label="TRIPS COMPLETED" value={system.arrived ?? 0} suffix="" icon="✓" />
      </section>

      <section className="map-layout" id="overview">
        <NetworkMap topology={topology} live={live} running={sim.running} />
        <aside className="ops-rail">
          <article className="panel rail-panel">
            <div className="panel-heading">
              <div><p className="eyebrow">SIGNAL CONTROL</p><h2>Intersection status</h2></div>
              <span className="count-pill">{intersections.length || Object.values(topology?.nodes || {}).filter((node) => node.tls).length} NODES</span>
            </div>
            <div className="junction-list">
              {intersections.length ? intersections.map(([id, item]) => (
                <div className="junction-row" key={id}>
                  <span className="junction-mark"><i className={String(item.state).toLowerCase().includes("g") ? "green" : "red"} /></span>
                  <div><strong>{id}</strong><small>PHASE {item.phase} · {String(item.state || "").length} SIGNALS</small></div>
                  <b>{item.total_queue ?? 0}<small> QUEUED</small></b>
                </div>
              )) : <div className="rail-empty">Start a simulation to see live signal states.</div>}
            </div>
          </article>
          <article className="panel incident-panel" id="incidents">
            <div className="panel-heading">
              <div><p className="eyebrow">PRIORITY MONITOR</p><h2>Emergency response</h2></div>
              <span className={`count-pill ${alerts.length ? "alert" : ""}`}>{alerts.length} ACTIVE</span>
            </div>
            {alerts.length ? <div className="alert-list">{alerts.map((vehicle) =>
              <div className="alert-item" key={vehicle.id}>
                <span className="alert-icon">!</span>
                <div><strong>{vehicle.id}</strong><small>{vehicle.edge} · {vehicle.speed} m/s</small></div>
                <span className="alert-live">LIVE</span>
              </div>)}</div> :
              <div className="quiet-state"><span>✓</span><strong>All clear</strong><small>Emergency vehicle alerts will appear here.</small></div>}
            <div className="mini-stat"><span>ACTIVE STRATEGY</span><strong>{controllerNames[controller] || controller}</strong></div>
          </article>
        </aside>
      </section>

      <section className="lower-grid" id="performance">
        <article className="panel performance-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">EVALUATION / SAVED RUNS</p><h2>Strategy performance</h2></div>
            <span className="data-tag">MEAN OF AVAILABLE SEEDS</span>
          </div>
          {current ? <>
            <div className="comparison-cards">
              <Compare label="Mean waiting time" value={current.mean_waiting_time_s} baseline={baseline?.mean_waiting_time_s} unit="s" lower />
              <Compare label="Mean travel time" value={current.mean_travel_time_s} baseline={baseline?.mean_travel_time_s} unit="s" lower />
              <Compare label="Throughput" value={current.avg_throughput_veh_h} baseline={baseline?.avg_throughput_veh_h} unit="veh/h" />
              <Compare label="Average queue" value={current.avg_total_halting} baseline={baseline?.avg_total_halting} unit="veh" lower />
            </div>
            <div className="metric-charts">
              <MetricChart title="WAITING TIME" unit="s" rows={analytics.summaries} field="mean_waiting_time_s" lower />
              <MetricChart title="TRAVEL TIME" unit="s" rows={analytics.summaries} field="mean_travel_time_s" lower />
              <MetricChart title="THROUGHPUT" unit="veh/h" rows={analytics.summaries} field="avg_throughput_veh_h" />
              <MetricChart title="QUEUE" unit="veh" rows={analytics.summaries} field="avg_total_halting" lower />
            </div>
          </> : <div className="empty-state compact"><strong>No comparison results found</strong><p>Generate strategy results with compare_strategies.py.</p></div>}
        </article>

        <article className="panel prediction-panel">
          <div className="panel-heading">
            <div><p className="eyebrow">MACHINE LEARNING / +5 MIN</p><h2>Congestion outlook</h2></div>
            <span className="data-tag">XGBOOST</span>
          </div>
          <p className="prediction-note">{forecast?.source || "Loading forecast source…"}. Forecast horizon: five minutes.</p>
          {forecast?.error ? <div className="empty-state compact"><strong>Forecast unavailable</strong><p>{forecast.error}</p></div> :
            <div className="prediction-list">
              {(forecast?.predictions || []).slice(0, 7).map((item) => (
                <div className="prediction-row" key={item.edge_id}>
                  <strong>{item.edge_id}<small>→ {item.tls_id}</small></strong>
                  <span>{item.queue_vehicles.toFixed(1)} queued</span>
                  <span className={`forecast-badge ${item.congested_5min ? "congested" : ""}`}>{item.status}</span>
                  <small>{Math.round(item.probability * 100)}%</small>
                </div>
              ))}
              {!forecast?.predictions?.length && !forecast?.error &&
                <div className="empty-state compact"><strong>{sim.running ? "Live forecast warming up" : "Waiting for forecast data"}</strong><p>{sim.running ? "The model needs three five-minute traffic samples before it can forecast." : "Provide a trained model and dataset samples."}</p></div>}
            </div>}
          {forecast?.run && <p className="source-note">Sample run: {forecast.run} · 5-minute prediction horizon</p>}
        </article>
      </section>
      <footer><span>FLOWSTATE <i>•</i> SMART TRAFFIC OPERATIONS</span><span>Live data refreshes every 2.5 seconds</span></footer>
    </main>
  );
}

function Compare({ label, value, baseline, unit, lower = false }) {
  const delta = value == null || baseline == null || baseline === 0 ? null : ((value - baseline) / Math.abs(baseline)) * 100;
  const better = delta == null ? null : (delta < 0) === lower;
  const atBaseline = delta != null && Math.abs(delta) < 0.05;
  return (
    <div className="compare-card">
      <span>{label}</span>
      <strong>{value == null ? "—" : `${value} ${unit}`}</strong>
      <small>{delta == null ? "No fixed-time baseline" : atBaseline ? "At fixed-time baseline" : `${delta > 0 ? "+" : ""}${delta.toFixed(1)}% vs fixed-time`}
        {better != null && !atBaseline && <i className={better ? "better" : "worse"}>{better ? "  IMPROVED" : "  REGRESSED"}</i>}
      </small>
    </div>
  );
}

function MetricChart({ title, unit, rows, field, lower = false }) {
  const values = rows.map((row) => Number(row[field])).filter(Number.isFinite);
  const max = Math.max(1, ...values);
  return (
    <div className="metric-chart">
      <div className="chart-title">{title}<small>{unit}</small></div>
      {rows.map((row) => {
        const value = Number(row[field]);
        const width = Number.isFinite(value) ? Math.max(value > 0 ? 2 : 0, (value / max) * 100) : 0;
        const isBaseline = row.controller === "fixed";
        return (
          <div className="chart-row" key={`${title}-${row.controller}`} title={`${row.controller_label}: ${Number.isFinite(value) ? value : "N/A"} ${unit}`}>
            <span>{controllerNames[row.controller] || row.controller}</span>
            <span className="chart-track"><span className={isBaseline ? "baseline-bar" : lower ? "lower-bar" : "higher-bar"} style={{ width: `${width}%` }} /></span>
            <b>{Number.isFinite(value) ? value.toFixed(1) : "—"}</b>
          </div>
        );
      })}
    </div>
  );
}

export default App;
