/**
 * Traffic Visualizer – Multi-Panel Dashboard
 *
 * Manages multiple independent CityFlow simulation panels that play back
 * synchronized. Each panel is a self-contained PIXI.js Application.
 */

"use strict";

// ── Constants ──────────────────────────────────────────────────────────────

const API_BASE = "http://localhost:8080";

const POLICY_META = {
  no_intervention: { label: "No Intervention", color: 0x94a3b8, cssVar: "--color-no-intervention" },
  fixed:           { label: "Fixed Cycle",     color: 0xf59e0b, cssVar: "--color-fixed" },
  random:          { label: "Random Phase",    color: 0x10b981, cssVar: "--color-random" },
  learned:         { label: "DQN (Learned)",   color: 0x5b6cf9, cssVar: "--color-learned" },
  dqn_heuristic:   { label: "DQN + Heuristic", color: 0x06b6d4, cssVar: "--color-dqn-heuristic", comingSoon: true },
  llm_dqn:         { label: "LLM + DQN",       color: 0xec4899, cssVar: "--color-llm-dqn",       comingSoon: true },
};

const ALL_POLICY_KEYS = Object.keys(POLICY_META);

const BACKGROUND_COLOR  = 0xf5f0e8;
const LANE_COLOR        = 0x8a7f72;
const LANE_BORDER_COLOR = 0x4a4035;
const LANE_INNER_COLOR  = 0x4a4035;
const LANE_BORDER_WIDTH = 1;
const LANE_DASH  = 10;
const LANE_GAP   = 12;
const ROTATE     = 90;
const TRAFFIC_LIGHT_WIDTH = 3;
const MAX_TRAFFIC_LIGHT_NUM = 2000;
const NUM_CAR_POOL = 5000;
const MAX_REPLAY_STEPS = 60; // 1 min of sim; server trims the file before sending
const CAR_LENGTH   = 5;
const CAR_WIDTH    = 2;
const LIGHT_RED    = 0xdb635e;
const LIGHT_GREEN  = 0x85ee00;
const CAR_COLORS   = [0xf2bfd7, 0xb7ebe4, 0xdbebb7, 0xf5ddb5, 0xd4b5f5];

// ── State ──────────────────────────────────────────────────────────────────

let activeCityId       = null;
let activeScenario     = null;
let sharedRoadnetData  = null; // parsed roadnetLogFile.json (static)
let activeReplayPayloads = []; // cached after last run — used when toggling view modes
let activePanels       = [];   // SimPanel[]
let viewLayout         = 1;
let paused          = false;
let globalStep      = 0;
let totalSteps      = 0;
let frameElapsed    = 0;
let replaySpeed     = 0.5;
let animFrameId     = null;

// ── DOM refs ───────────────────────────────────────────────────────────────

const panelsContainer  = document.getElementById("panels-container");
const welcomeOverlay   = document.getElementById("welcome-overlay");
const playbackBar      = document.getElementById("playback-bar");
const pauseBtn         = document.getElementById("pause-btn");
const scrubber         = document.getElementById("scrubber");
const stepDisplay      = document.getElementById("step-display");
const speedLabel       = document.getElementById("speed-label");
const speedDown        = document.getElementById("speed-down");
const speedUp          = document.getElementById("speed-up");
// metrics-bar element removed — metrics now live inside each panel overlay
const progressSection  = document.getElementById("progress-section");
const runBtn           = document.getElementById("run-btn");
const citySelect       = document.getElementById("city-select");
const scenarioSelect   = document.getElementById("scenario-select");

// ── Initialise page ────────────────────────────────────────────────────────

(async function init() {
  setupDropZone();
  setupPolicyCheckboxes();
  setupViewModeButtons();
  setupPlaybackControls();
  setupSidebarResize();
  await loadCityList();
})();

// ── City / scenario pickers ────────────────────────────────────────────────

async function loadCityList() {
  try {
    const resp = await fetch(`${API_BASE}/cities`);
    const data = await resp.json();
    populateCitySelect(data.cities || []);
  } catch (_) {
    // Server not running yet – leave selects empty.
  }
}

function populateCitySelect(cities) {
  citySelect.innerHTML = '<option value="">-- select city --</option>';
  for (const c of cities) {
    const opt = document.createElement("option");
    opt.value = c;
    opt.textContent = c;
    citySelect.appendChild(opt);
  }
}

citySelect.addEventListener("change", async () => {
  const city = citySelect.value;
  scenarioSelect.innerHTML = '<option value="">-- loading --</option>';
  scenarioSelect.disabled = true;
  if (!city) return;
  try {
    const resp = await fetch(`${API_BASE}/cities/${city}/scenarios`);
    const data = await resp.json();
    populateScenarioSelect(data.scenarios || []);
  } catch (_) {
    scenarioSelect.innerHTML = '<option value="">-- error --</option>';
  }
});

function populateScenarioSelect(scenarios) {
  scenarioSelect.innerHTML = '<option value="">-- select scenario --</option>';
  for (const s of scenarios) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    scenarioSelect.appendChild(opt);
  }
  scenarioSelect.disabled = false;
}

// ── Roadnet upload drop zone ───────────────────────────────────────────────

function setupDropZone() {
  const zone  = document.getElementById("roadnet-drop-zone");
  const input = document.getElementById("roadnet-file-input");

  zone.addEventListener("click", () => input.click());

  zone.addEventListener("dragover", (e) => {
    e.preventDefault();
    zone.classList.add("drag-over");
  });
  zone.addEventListener("dragleave", () => zone.classList.remove("drag-over"));
  zone.addEventListener("drop", (e) => {
    e.preventDefault();
    zone.classList.remove("drag-over");
    if (e.dataTransfer.files[0]) handleRoadnetFile(e.dataTransfer.files[0]);
  });

  input.addEventListener("change", () => {
    if (input.files[0]) handleRoadnetFile(input.files[0]);
  });
}

async function handleRoadnetFile(file) {
  const zone = document.getElementById("roadnet-drop-zone");
  zone.querySelector(".drop-filename").textContent = file.name;

  const formData = new FormData();
  formData.append("file", file);

  try {
    const resp = await fetch(`${API_BASE}/upload-roadnet`, { method: "POST", body: formData });
    const data = await resp.json();

    if (data.matched && data.city_id) {
      // Auto-select the matched city.
      citySelect.value = data.city_id;
      populateScenarioSelect(data.scenarios);
      showToast(`Matched city: ${data.city_id}`);
    } else if (data.all_cities && data.all_cities.length > 0) {
      populateCitySelect(data.all_cities);
      showToast("No exact match – please select city manually.", "warn");
    }
  } catch (err) {
    showToast("Upload failed: " + err.message, "error");
  }
}

// ── Policy checkboxes ──────────────────────────────────────────────────────

function setupPolicyCheckboxes() {
  const list = document.getElementById("policy-list");

  for (const key of ALL_POLICY_KEYS) {
    const meta = POLICY_META[key];
    const item = document.createElement("label");
    item.className = "policy-item" + (meta.comingSoon ? " disabled" : "");

    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = key;
    cb.id = `cb-${key}`;
    cb.checked = !meta.comingSoon;
    cb.disabled = !!meta.comingSoon;

    const dot = document.createElement("span");
    dot.className = "policy-badge";
    dot.style.background = `#${meta.color.toString(16).padStart(6, "0")}`;

    const label = document.createElement("span");
    label.className = "policy-label";
    label.textContent = meta.label;

    item.appendChild(cb);
    item.appendChild(dot);
    item.appendChild(label);

    if (meta.comingSoon) {
      const tag = document.createElement("span");
      tag.className = "policy-tag";
      tag.textContent = "soon";
      item.appendChild(tag);
    }

    list.appendChild(item);
  }
}

function getSelectedPolicies() {
  return ALL_POLICY_KEYS.filter((key) => {
    const cb = document.getElementById(`cb-${key}`);
    return cb && cb.checked && !POLICY_META[key].comingSoon;
  });
}

// ── View mode buttons ──────────────────────────────────────────────────────

function setupViewModeButtons() {
  document.querySelectorAll(".view-mode-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".view-mode-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      viewLayout = parseInt(btn.dataset.panels, 10);
      if (activePanels.length > 0) renderPanelGrid();
    });
  });
}

// ── Run button ─────────────────────────────────────────────────────────────

runBtn.addEventListener("click", async () => {
  const city     = citySelect.value;
  const scenario = scenarioSelect.value;
  const policies = getSelectedPolicies();

  if (!city || !scenario) {
    showToast("Select a city and scenario first.", "warn");
    return;
  }
  if (policies.length === 0) {
    showToast("Select at least one policy.", "warn");
    return;
  }

  activeCityId   = city;
  activeScenario = scenario;

  runBtn.disabled = true;
  initProgress(policies);

  // Pre-flight: check which policies already have cached results so their
  // progress bars can immediately show "✓ cached" instead of animating.
  let cachedPolicies = new Set();
  try {
    const cacheResp = await fetch(`${API_BASE}/metrics/${city}/${scenario}`);
    if (cacheResp.ok) {
      const cacheData = await cacheResp.json();
      for (const key of policies) {
        if (cacheData.metrics && cacheData.metrics[key]) cachedPolicies.add(key);
      }
    }
  } catch (_) { /* server may not have any results yet — treat all as uncached */ }

  policies.forEach((p) => {
    if (cachedPolicies.has(p)) markPolicyCached(p);
    else markPolicyRunning(p);
  });

  const allResults = await Promise.all(
    policies.map(async (policy) => {
      const t0 = Date.now();
      try {
        const resp = await fetch(`${API_BASE}/run-simulations`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            city_id: city,
            scenario_name: scenario,
            policies: [policy],
            force: document.getElementById("force-rerun-checkbox")?.checked ?? false,
          }),
        });
        if (!resp.ok) {
          const errData = await resp.json();
          throw new Error(errData.detail || resp.statusText);
        }
        const data = await resp.json();
        const result = data.results[0];
        markPolicyDone(policy, Date.now() - t0, result.replay_available ? "done" : "error");
        return result;
      } catch (err) {
        markPolicyDone(policy, Date.now() - t0, "error");
        return {
          policy_name: policy,
          metrics: { error: err.message },
          replay_available: false,
          roadnet_log_available: false,
        };
      }
    })
  );

  const firstSuccess = allResults.find((r) => r.roadnet_log_available);
  if (!firstSuccess) {
    showToast("No successful policy runs.", "error");
    runBtn.disabled = false;
    return;
  }

  try {
    await fetchAndStartVisualization(city, scenario, allResults);
  } catch (err) {
    showToast("Error: " + err.message, "error");
  } finally {
    runBtn.disabled = false;
  }
});

// ── Fetch data & start visualization ──────────────────────────────────────

async function fetchAndStartVisualization(city, scenario, policyResults) {
  stopAnimation();
  destroyAllPanels();

  // Fetch the shared roadnet log (one call is enough – same static network).
  const firstPolicy = policyResults.find((r) => r.roadnet_log_available);
  const roadnetResp = await fetch(
    `${API_BASE}/roadnet-log/${city}/${scenario}/${firstPolicy.policy_name}`
  );
  if (!roadnetResp.ok) throw new Error("Failed to load roadnet log.");
  sharedRoadnetData = await roadnetResp.json();
  const rn = sharedRoadnetData.static || sharedRoadnetData;
  console.log("[viz] roadnet loaded — nodes:", (rn.node || rn.nodes || []).length, "edges:", (rn.edge || rn.edges || []).length);

  // Fetch replay data per policy — sequentially to avoid holding all large
  // replay strings in memory simultaneously (each file can be 100s of MB).
  const replayPayloads = [];
  for (const result of policyResults.filter((r) => r.replay_available)) {
    const replayResp = await fetch(
      `${API_BASE}/replay/${city}/${scenario}/${result.policy_name}?max_steps=${MAX_REPLAY_STEPS}`
    );
    if (!replayResp.ok) {
      console.warn("[viz] replay fetch failed for", result.policy_name, replayResp.status);
      continue;
    }
    const text = await replayResp.text();
    // Split and immediately discard the raw string (lets GC reclaim it).
    const lines = text.split("\n").filter((l) => l.trim().length > 0).slice(0, MAX_REPLAY_STEPS);
    console.log(`[viz] replay loaded — policy: ${result.policy_name}, steps: ${lines.length}`);
    replayPayloads.push({ policy_name: result.policy_name, metrics: result.metrics, lines });
  }
  console.log("[viz] replay payloads ready:", replayPayloads.map((p) => `${p.policy_name}(${p.lines.length}steps)`));

  totalSteps  = Math.min(...replayPayloads.map((p) => p.lines.length));
  globalStep  = 0;
  frameElapsed = 0;
  paused = false;

  scrubber.max   = Math.max(0, totalSteps - 1);
  scrubber.value = 0;

  // Cache for view-mode toggling, then build panels limited by viewLayout.
  activeReplayPayloads = replayPayloads;
  renderPanelGrid();

  welcomeOverlay.classList.add("hidden");
  startAnimation();
}

// ── Panel grid ─────────────────────────────────────────────────────────────

function renderPanelGrid() {
  panelsContainer.className = `layout-${viewLayout}`;
  destroyAllPanels();

  if (!activeReplayPayloads.length) return; // no data yet

  stopAnimation();
  const toShow = activeReplayPayloads.slice(0, viewLayout);
  for (const payload of toShow) {
    const panel = new SimPanel(payload);
    panel.init(sharedRoadnetData);
    panelsContainer.appendChild(panel.element);
    activePanels.push(panel);
  }
  globalStep   = 0;
  frameElapsed = 0;
  scrubber.value = 0;
  startAnimation();
}

function destroyAllPanels() {
  for (const p of activePanels) p.destroy();
  activePanels = [];
  panelsContainer.innerHTML = "";
}

// ── SimPanel class ─────────────────────────────────────────────────────────

class SimPanel {
  constructor({ policy_name, metrics, lines }) {
    this.policyName = policy_name;
    this.metrics    = metrics || {};
    this.lines      = lines;
    this.app        = null;
    this.viewport   = null;
    this.nodes      = {};
    this.edges      = {};
    this.trafficLightsG = {};
    this.carPool    = [];
    this.carContainer = null;
    this.trafficLightContainer = null;
    this.turnSignalContainer = null;
    this.element    = null;
    this.canvasWrapper = null;
    this.renderer   = null;
    this.simulatorContainer = null;
    this._ready     = false;
  }

  get meta() { return POLICY_META[this.policyName] || { label: this.policyName, color: 0x888888 }; }

  init(roadnetData) {
    this._buildElement();
    this._initPixi(roadnetData);
  }

  _buildElement() {
    const colorHex = "#" + this.meta.color.toString(16).padStart(6, "0");

    this.element = document.createElement("div");
    this.element.className = "sim-panel";
    this.element.style.borderTop = `3px solid ${colorHex}`;

    // Header
    const header = document.createElement("div");
    header.className = "panel-header";

    // Label box — dark background so text is readable over the map
    const labelBox = document.createElement("div");
    labelBox.className = "panel-label-box";

    const dot = document.createElement("span");
    dot.className = "panel-policy-dot";
    dot.style.background = colorHex;

    const name = document.createElement("span");
    name.className = "panel-policy-name";
    name.textContent = this.meta.label;
    name.style.color = colorHex;

    labelBox.appendChild(dot);
    labelBox.appendChild(name);
    header.appendChild(labelBox);

    // Zoom buttons (pointer-events: auto overrides the header's none)
    const zoomWrap = document.createElement("div");
    zoomWrap.className = "panel-zoom-controls";
    const zoomIn  = document.createElement("button");
    const zoomOut = document.createElement("button");
    zoomIn.className  = "panel-zoom-btn";
    zoomOut.className = "panel-zoom-btn";
    zoomIn.textContent  = "+";
    zoomOut.textContent = "−";
    zoomIn.addEventListener("click",  () => { if (this.viewport) this.viewport.zoomPercent(0.25, true); });
    zoomOut.addEventListener("click", () => { if (this.viewport) this.viewport.zoomPercent(-0.25, true); });
    zoomWrap.appendChild(zoomOut);
    zoomWrap.appendChild(zoomIn);
    header.appendChild(zoomWrap);

    // Canvas wrapper
    this.canvasWrapper = document.createElement("div");
    this.canvasWrapper.className = "panel-canvas-wrapper";

    // Spinner
    const spinner = document.createElement("div");
    spinner.className = "panel-spinner";
    spinner.innerHTML = '<div class="spinner-ring"></div>';
    this.canvasWrapper.appendChild(spinner);

    // Metrics overlay — bottom-left of the canvas
    const m = this.metrics;
    const metricRows = [
      ["Avg Wait",    m.mean_waiting_vehicles],
      ["Throughput",  m.throughput],
      ["Travel Time", m.average_travel_time],
    ];
    const overlay = document.createElement("div");
    overlay.className = "panel-metrics-overlay";
    for (const [label, val] of metricRows) {
      const row = document.createElement("div");
      row.className = "pmo-row";
      row.innerHTML = `<span class="pmo-label">${label}</span><span class="pmo-value">${val !== undefined ? Number(val).toFixed(1) : "—"}</span>`;
      overlay.appendChild(row);
    }
    this.canvasWrapper.appendChild(overlay);

    this.element.appendChild(header);
    this.element.appendChild(this.canvasWrapper);
  }

  _initPixi(roadnetData) {
    const wrapper = this.canvasWrapper;

    // Defer to next frame so the DOM element has dimensions.
    requestAnimationFrame(() => {
      const w = wrapper.offsetWidth  || 800;
      const h = wrapper.offsetHeight || 600;
      console.log(`[viz] PIXI init — policy: ${this.policyName}, canvas: ${w}x${h}`);

      this.app = new PIXI.Application({
        width: w,
        height: h,
        backgroundColor: BACKGROUND_COLOR,
        antialias: false,
        resolution: window.devicePixelRatio || 1,
        autoDensity: true,
      });

      wrapper.appendChild(this.app.view);

      this.renderer = this.app.renderer;
      this._drawRoadnet(roadnetData);

      // Remove spinner.
      const spinner = wrapper.querySelector(".panel-spinner");
      if (spinner) spinner.classList.add("hidden");
      this._ready = true;
      console.log(`[viz] panel ready — policy: ${this.policyName}`);
    });
  }

  _drawRoadnet(roadnetJson) {
    this.nodes = {};
    this.edges = {};
    this.trafficLightsG = {};

    const vp = new Viewport.Viewport({
      screenWidth:  this.app.renderer.width,
      screenHeight: this.app.renderer.height,
      interaction:  this.app.renderer.plugins.interaction,
    });
    vp.drag().pinch().wheel().decelerate();
    this.app.stage.addChild(vp);
    this.viewport = vp;

    this.simulatorContainer = new PIXI.Container();
    vp.addChild(this.simulatorContainer);

    const roadnet = roadnetJson.static || roadnetJson;
    const nodeList = roadnet.node || roadnet.nodes || [];
    const edgeList = roadnet.edge || roadnet.edges || [];
    console.log(`[viz] _drawRoadnet — policy: ${this.policyName}, nodes: ${nodeList.length}, edges: ${edgeList.length}`);
    for (const node of nodeList) {
      // Spread to avoid mutating shared roadnetData across panels.
      this.nodes[node.id] = { ...node, point: new Point(transCoord(node.point)) };
    }
    for (const edge of edgeList) {
      // Spread + override resolved refs so shared source objects are never touched.
      this.edges[edge.id] = {
        ...edge,
        from:   this.nodes[edge.from],
        to:     this.nodes[edge.to],
        points: (edge.points || []).map((p) => new Point(transCoord(p))),
      };
    }

    this.trafficLightContainer = new PIXI.particles.ParticleContainer(
      MAX_TRAFFIC_LIGHT_NUM, { tint: true }
    );

    const mapGraphics = new PIXI.Graphics();
    this.simulatorContainer.addChild(mapGraphics);

    for (const nodeId in this.nodes) {
      if (!this.nodes[nodeId].virtual) drawNode(this.nodes[nodeId], mapGraphics);
    }
    for (const edgeId in this.edges) {
      this._drawEdge(this.edges[edgeId], mapGraphics);
    }

    const bounds = this.simulatorContainer.getBounds();
    this.simulatorContainer.pivot.set(
      bounds.x + bounds.width  / 2,
      bounds.y + bounds.height / 2
    );
    this.simulatorContainer.position.set(
      this.renderer.width  / 2,
      this.renderer.height / 2
    );
    this.simulatorContainer.addChild(this.trafficLightContainer);

    // Car pool.
    const carG = new PIXI.Graphics();
    carG.lineStyle(0);
    carG.beginFill(0xffffff, 0.8);
    carG.drawRect(0, 0, CAR_LENGTH, CAR_WIDTH);
    const carTexture = this.renderer.generateTexture(carG);

    this.carContainer = new PIXI.particles.ParticleContainer(
      NUM_CAR_POOL, { rotation: true, tint: true }
    );
    this.turnSignalContainer = new PIXI.particles.ParticleContainer(
      NUM_CAR_POOL, { rotation: true }
    );
    this.simulatorContainer.addChild(this.carContainer);
    this.simulatorContainer.addChild(this.turnSignalContainer);

    this.carPool = [];
    for (let i = 0; i < NUM_CAR_POOL; i++) {
      const car = new PIXI.Sprite(carTexture);
      car.anchor.set(1, 0.5);
      this.carPool.push(car);
    }
  }

  _drawEdge(edge, graphics) {
    if (!edge.from || !edge.to || !edge.points || edge.points.length < 2) return;

    const roadWidth = (edge.laneWidths || []).reduce((s, w) => s + w, 0);
    let prevPointBOffset = null;

    const lightG = new PIXI.Graphics();
    lightG.lineStyle(TRAFFIC_LIGHT_WIDTH, 0xffffff);
    lightG.moveTo(0, 0);
    lightG.lineTo(1, 0);
    const lightTexture = this.renderer.generateTexture(lightG);

    for (let i = 1; i < edge.points.length; i++) {
      let pointA, pointAOffset, pointB, pointBOffset;

      if (i === 1) {
        pointA = edge.points[0].moveAlongDirectTo(edge.points[1], edge.from.virtual ? 0 : (edge.from.width || 0));
        pointAOffset = edge.points[0].directTo(edge.points[1]).rotate(ROTATE);
      } else {
        pointA = edge.points[i - 1];
        pointAOffset = prevPointBOffset;
      }
      if (i === edge.points.length - 1) {
        pointB = edge.points[i].moveAlongDirectTo(edge.points[i - 1], edge.to.virtual ? 0 : (edge.to.width || 0));
        pointBOffset = edge.points[i - 1].directTo(edge.points[i]).rotate(ROTATE);
      } else {
        pointB = edge.points[i];
        pointBOffset = edge.points[i - 1].directTo(edge.points[i + 1]).rotate(ROTATE);
      }
      prevPointBOffset = pointBOffset;

      // Traffic lights at lane ends.
      if (i === edge.points.length - 1 && !edge.to.virtual) {
        const lights = [];
        let prevOffset = 0;
        let offset = 0;
        const laneWidths = edge.laneWidths || [];
        for (let lane = 0; lane < (edge.nLane || 0); lane++) {
          offset += laneWidths[lane] || 0;
          const light = new PIXI.Sprite(lightTexture);
          light.anchor.set(0, 0.5);
          light.scale.set(offset - prevOffset, 1);
          const pt = pointB.moveAlong(pointBOffset, prevOffset);
          light.position.set(pt.x, pt.y);
          light.rotation = pointBOffset.getAngleInRadians();
          lights.push(light);
          prevOffset = offset;
          this.trafficLightContainer.addChild(light);
        }
        this.trafficLightsG[edge.id] = lights;
      }

      const pointA1 = pointA.moveAlong(pointAOffset, roadWidth);
      const pointB1 = pointB.moveAlong(pointBOffset, roadWidth);

      graphics.lineStyle(LANE_BORDER_WIDTH, LANE_BORDER_COLOR, 1);
      graphics.moveTo(pointA.x, pointA.y);
      graphics.lineTo(pointB.x, pointB.y);

      graphics.lineStyle(0);
      graphics.beginFill(LANE_COLOR);
      graphics.drawPolygon([
        pointA.x,  pointA.y,
        pointB.x,  pointB.y,
        pointB1.x, pointB1.y,
        pointA1.x, pointA1.y,
      ]);
      graphics.endFill();
    }
  }

  drawStep(step) {
    if (!this._ready || step >= this.lines.length) return;
    if (step === 0 && !this._loggedFirstStep) {
      this._loggedFirstStep = true;
      console.log(`[viz] first drawStep — policy: ${this.policyName}, step 0 of ${this.lines.length}`);
    }

    const line = this.lines[step];
    const semiIdx = line.indexOf(";");
    if (semiIdx < 0) return;
    const carPart = line.substring(0, semiIdx);
    const tlPart  = line.substring(semiIdx + 1);

    // Traffic lights.
    for (const tlEntry of tlPart.split(",")) {
      const parts = tlEntry.trim().split(" ");
      const edgeId = parts[0];
      const statuses = parts.slice(1);
      const lights = this.trafficLightsG[edgeId];
      if (!lights) continue;
      for (let j = 0; j < statuses.length && j < lights.length; j++) {
        const s = statuses[j];
        if (s === "i") { lights[j].alpha = 0; continue; }
        lights[j].alpha = 1;
        lights[j].tint  = s === "g" ? LIGHT_GREEN : LIGHT_RED;
      }
    }

    // Cars.
    this.carContainer.removeChildren();
    const carEntries = carPart.split(",").filter((e) => e.trim().length > 0);
    for (let i = 0; i < carEntries.length && i < this.carPool.length; i++) {
      const parts = carEntries[i].trim().split(" ");
      if (parts.length < 7) continue;
      const [x, y, rot, _id, _lane, length, width] = parts;
      const pos = transCoord([parseFloat(x), parseFloat(y)]);
      const car = this.carPool[i];
      car.position.set(pos[0], pos[1]);
      car.rotation = 2 * Math.PI - parseFloat(rot);
      car.tint     = CAR_COLORS[stringHash(_id || String(i)) % CAR_COLORS.length];
      car.width    = parseFloat(length);
      car.height   = parseFloat(width);
      this.carContainer.addChild(car);
    }
  }

  destroy() {
    this._ready = false;
    if (this.app) {
      this.app.destroy(true, { children: true });
      this.app = null;
    }
  }
}

// ── Animation loop ─────────────────────────────────────────────────────────

function startAnimation() {
  if (animFrameId !== null) cancelAnimationFrame(animFrameId);
  let lastTs = 0;

  function loop(ts) {
    animFrameId = requestAnimationFrame(loop);
    const delta = Math.min((ts - lastTs) / 16.67, 3); // cap at 3 frames
    lastTs = ts;

    if (!paused && activePanels.length > 0) {
      for (const panel of activePanels) panel.drawStep(globalStep);

      frameElapsed += delta;
      const framesPerStep = Math.max(1, Math.round(1 / replaySpeed ** 2));
      if (frameElapsed >= framesPerStep) {
        frameElapsed = 0;
        globalStep = (globalStep + 1) % (totalSteps || 1);
        syncScrubber();
      }
    }
  }

  animFrameId = requestAnimationFrame(loop);
}

function stopAnimation() {
  if (animFrameId !== null) {
    cancelAnimationFrame(animFrameId);
    animFrameId = null;
  }
}

// ── Playback controls ──────────────────────────────────────────────────────

function setupPlaybackControls() {
  pauseBtn.addEventListener("click", () => {
    paused = !paused;
    pauseBtn.textContent = paused ? "Play" : "Pause";
  });

  scrubber.addEventListener("input", () => {
    globalStep   = parseInt(scrubber.value, 10);
    frameElapsed = 0;
    for (const p of activePanels) p.drawStep(globalStep);
    stepDisplay.textContent = `${globalStep + 1} / ${totalSteps}`;
  });

  speedDown.addEventListener("click", () => setSpeed(Math.max(0.1, replaySpeed - 0.1)));
  speedUp.addEventListener("click",   () => setSpeed(Math.min(2.0, replaySpeed + 0.1)));

  document.addEventListener("keydown", (e) => {
    if (e.code === "Space") { e.preventDefault(); pauseBtn.click(); }
    else if (e.code === "ArrowRight") stepForward();
    else if (e.code === "ArrowLeft")  stepBackward();
  });
}

function setSpeed(s) {
  replaySpeed  = +s.toFixed(1);
  speedLabel.textContent = `${replaySpeed.toFixed(1)}x`;
}

function stepForward()  { globalStep = (globalStep + 1) % (totalSteps || 1); syncScrubber(); for (const p of activePanels) p.drawStep(globalStep); }
function stepBackward() { globalStep = ((globalStep - 1) + (totalSteps || 1)) % (totalSteps || 1); syncScrubber(); for (const p of activePanels) p.drawStep(globalStep); }

function syncScrubber() {
  scrubber.value = globalStep;
  stepDisplay.textContent = `${globalStep + 1} / ${totalSteps}`;
}

// ── Progress display ───────────────────────────────────────────────────────

const _progressIntervals = {};

function initProgress(policies) {
  progressSection.style.display = "block";
  progressSection.innerHTML = "";

  const h = document.createElement("h3");
  h.textContent = "Running simulations...";
  progressSection.appendChild(h);

  for (const key of policies) {
    const meta = POLICY_META[key] || { label: key, color: 0x888888 };
    const colorHex = "#" + meta.color.toString(16).padStart(6, "0");

    const row = document.createElement("div");
    row.className = "progress-policy-row";

    // Top line: dot + name + percent label
    const topLine = document.createElement("div");
    topLine.className = "progress-policy-top";

    const dot = document.createElement("span");
    dot.className = "progress-policy-dot";
    dot.style.background = colorHex;

    const name = document.createElement("span");
    name.className = "progress-policy-name";
    name.textContent = meta.label;

    const pct = document.createElement("span");
    pct.className = "progress-policy-pct";
    pct.id = `progress-pct-${key}`;
    pct.textContent = "0%";

    topLine.appendChild(dot);
    topLine.appendChild(name);
    topLine.appendChild(pct);

    // Fill bar
    const track = document.createElement("div");
    track.className = "progress-policy-bar-track";
    const fill = document.createElement("div");
    fill.className = "progress-policy-bar-fill";
    fill.id = `progress-fill-${key}`;
    fill.style.background = colorHex;
    track.appendChild(fill);

    row.appendChild(topLine);
    row.appendChild(track);
    progressSection.appendChild(row);
  }
}

function markPolicyRunning(key) {
  // Linear fill: ~0.8% per 100ms tick → reaches 88% after ~11 s.
  let pct = 0;
  _progressIntervals[key] = setInterval(() => {
    pct = Math.min(88, pct + 0.8);
    _setBar(key, pct);
  }, 100);
}

function markPolicyCached(key) {
  // Immediately show as done — no animation needed for cache hits.
  _setBar(key, 100);
  const pctEl = document.getElementById(`progress-pct-${key}`);
  if (pctEl) {
    pctEl.textContent = "✓ cached";
    pctEl.className = "progress-policy-pct done";
  }
}

function markPolicyDone(key, elapsedMs, status) {
  clearInterval(_progressIntervals[key]);
  delete _progressIntervals[key];
  _setBar(key, 100, status);
  const pctEl = document.getElementById(`progress-pct-${key}`);
  if (pctEl) {
    const secs = (elapsedMs / 1000).toFixed(1);
    pctEl.textContent = status === "done" ? `✓ ${secs}s` : `✗ ${secs}s`;
    pctEl.className = `progress-policy-pct ${status}`;
  }
}

function _setBar(key, pct, status) {
  const fill  = document.getElementById(`progress-fill-${key}`);
  const label = document.getElementById(`progress-pct-${key}`);
  if (fill)  fill.style.width = `${pct.toFixed(1)}%`;
  if (label && !status) label.textContent = `${Math.round(pct)}%`;
  if (fill && status === "error") fill.style.background = "#ef4444";
}

// (showMetricsBar removed — metrics are now rendered inside each SimPanel overlay)

// ── Sidebar resize ─────────────────────────────────────────────────────────

function setupSidebarResize() {
  const sidebar = document.getElementById("sidebar");
  const handle  = document.getElementById("sidebar-resize-handle");
  const MIN_W = 180, MAX_W = 480;

  const saved = localStorage.getItem("sidebarWidth");
  if (saved) sidebar.style.width = parseInt(saved, 10) + "px";

  let dragging = false, startX = 0, startW = 0;

  handle.addEventListener("mousedown", (e) => {
    dragging = true;
    startX = e.clientX;
    startW = sidebar.offsetWidth;
    handle.classList.add("dragging");
    document.body.style.cursor = "col-resize";
    document.body.style.userSelect = "none";
    e.preventDefault();
  });

  document.addEventListener("mousemove", (e) => {
    if (!dragging) return;
    const w = Math.min(MAX_W, Math.max(MIN_W, startW + (e.clientX - startX)));
    sidebar.style.width = w + "px";
  });

  document.addEventListener("mouseup", () => {
    if (!dragging) return;
    dragging = false;
    handle.classList.remove("dragging");
    document.body.style.cursor = "";
    document.body.style.userSelect = "";
    localStorage.setItem("sidebarWidth", sidebar.offsetWidth);
  });
}

// ── Utility helpers ────────────────────────────────────────────────────────

function transCoord(point) {
  if (Array.isArray(point)) return [point[0], -point[1]];
  return [point.x, -point.y];
}

function stringHash(str) {
  let hash = 0, p = 127, pp = 1, m = 1e9 + 9;
  for (let i = 0; i < str.length; i++) {
    hash = (hash + str.charCodeAt(i) * pp) % m;
    pp = (pp * p) % m;
  }
  return hash;
}

function drawNode(node, graphics) {
  graphics.beginFill(LANE_COLOR);
  const outline = node.outline || [];
  for (let i = 0; i < outline.length; i += 2) {
    const px = outline[i], py = -outline[i + 1];
    if (i === 0) graphics.moveTo(px, py);
    else         graphics.lineTo(px, py);
  }
  graphics.endFill();
}

function showToast(msg, type = "info") {
  const colors = { info: "#5b6cf9", warn: "#f59e0b", error: "#ef4444" };
  const toast = document.createElement("div");
  toast.style.cssText = `
    position: fixed; bottom: 20px; right: 20px; z-index: 9999;
    background: ${colors[type] || colors.info}; color: #fff;
    padding: 10px 16px; border-radius: 6px; font-size: 13px;
    box-shadow: 0 4px 16px rgba(0,0,0,0.4);
    animation: fadeInUp 0.2s ease;
  `;
  toast.textContent = msg;
  document.body.appendChild(toast);
  setTimeout(() => toast.remove(), 3500);
}

// Extend PIXI.Graphics with drawLine if not already present.
if (PIXI.Graphics && !PIXI.Graphics.prototype.drawLine) {
  PIXI.Graphics.prototype.drawLine = function(pA, pB) {
    this.moveTo(pA.x, pA.y);
    this.lineTo(pB.x, pB.y);
  };
}
