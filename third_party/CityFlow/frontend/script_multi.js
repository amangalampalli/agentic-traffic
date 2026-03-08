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
  dqn_heuristic:   { label: "DQN + Heuristic", color: 0x06b6d4, cssVar: "--color-dqn-heuristic" },
  llm_dqn:         { label: "DQN + LLM",       color: 0xec4899, cssVar: "--color-llm-dqn" },
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
const DISTRICT_BORDER_COLOR = 0x34495e;
const GATEWAY_NODE_COLOR    = 0xf39c12;
const GATEWAY_EDGE_COLOR    = 0xe67e22;
const ENTRY_EDGE_COLOR      = 0x22c55e;
const EXIT_EDGE_COLOR       = 0xef4444;
const DISTRICT_PALETTE = [
  0x1f77b4, 0xff7f0e, 0x2ca02c, 0xd62728, 0x9467bd, 0x8c564b,
  0xe377c2, 0x7f7f7f, 0xbcbd22, 0x17becf, 0x393b79, 0x637939
];
const TRAFFIC_LIGHT_WIDTH = 3;
const MAX_TRAFFIC_LIGHT_NUM = 2000;
const NUM_CAR_POOL = 5000;
const SIM_WINDOW_SECONDS = 180; // Backend also runs only this much simulated time
const DEFAULT_CITY_ID = "city_0001";
const DEFAULT_SCENARIO = "normal";
const STORAGE_KEYS = {
  uiState: "trafficVisualizer.uiState.v1",
  recentRuns: "trafficVisualizer.recentRuns.v1",
};
const MAX_RECENT_RUNS = 12;
const CAR_LENGTH   = 5;
const CAR_WIDTH    = 2;
const LIGHT_RED    = 0xdb635e;
const LIGHT_GREEN  = 0x85ee00;
const CAR_COLORS   = [0xf2bfd7, 0xb7ebe4, 0xdbebb7, 0xf5ddb5, 0xd4b5f5];

// ── State ──────────────────────────────────────────────────────────────────

let activeCityId       = null;
let activeScenario     = null;
let sharedRoadnetData  = null; // parsed roadnetLogFile.json (static)
let activeDistrictMap  = null;
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
const summaryBar       = document.getElementById("summary-bar");
const pauseBtn         = document.getElementById("pause-btn");
const scrubber         = document.getElementById("scrubber");
const stepDisplay      = document.getElementById("step-display");
const speedLabel       = document.getElementById("speed-label");
const speedDown        = document.getElementById("speed-down");
const speedUp          = document.getElementById("speed-up");
// metrics-bar element removed — metrics now live inside each panel overlay
const progressSection  = document.getElementById("progress-section");
const debugLogEl       = document.getElementById("debug-log");
const recentRunsList   = document.getElementById("recent-runs-list");
const runBtn           = document.getElementById("run-btn");
const citySelect       = document.getElementById("city-select");
const scenarioSelect   = document.getElementById("scenario-select");
const forceRerunCheckbox = document.getElementById("force-rerun-checkbox");

let savedUiState = loadSavedUiState();
let suppressCityAutoDefault = false;

// ── Initialise page ────────────────────────────────────────────────────────

(async function init() {
  debugLog(`init start; sim_window=${SIM_WINDOW_SECONDS}s`);
  setupDropZone();
  setupPolicyCheckboxes();
  setupViewModeButtons();
  setupPlaybackControls();
  setupSidebarResize();
  restoreSavedPolicyState();
  restoreSavedViewLayout();
  renderRecentRuns();
  await loadCityList();
  restoreSavedUiSelections();
  debugLog("init complete");
})();

// ── City / scenario pickers ────────────────────────────────────────────────

async function loadCityList() {
  try {
    const resp = await fetch(`${API_BASE}/cities`);
    const data = await resp.json();
    debugLog(`loaded cities: ${(data.cities || []).join(", ")}`);
    populateCitySelect(data.cities || []);
  } catch (_) {
    // Server not running yet – leave selects empty.
    debugLog("failed to load city list");
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
  if (!suppressCityAutoDefault && cities.includes(DEFAULT_CITY_ID)) {
    citySelect.value = DEFAULT_CITY_ID;
    citySelect.dispatchEvent(new Event("change"));
  }
}

citySelect.addEventListener("change", async () => {
  const city = citySelect.value;
  scenarioSelect.innerHTML = '<option value="">-- loading --</option>';
  scenarioSelect.disabled = true;
   persistUiState();
  if (!city) return;
  try {
    const resp = await fetch(`${API_BASE}/cities/${city}/scenarios`);
    const data = await resp.json();
    debugLog(`loaded scenarios for ${city}: ${(data.scenarios || []).join(", ")}`);
    populateScenarioSelect(data.scenarios || []);
  } catch (_) {
    scenarioSelect.innerHTML = '<option value="">-- error --</option>';
    debugLog(`failed to load scenarios for ${city}`);
  }
});

scenarioSelect.addEventListener("change", persistUiState);
forceRerunCheckbox?.addEventListener("change", persistUiState);

function populateScenarioSelect(scenarios) {
  scenarioSelect.innerHTML = '<option value="">-- select scenario --</option>';
  for (const s of scenarios) {
    const opt = document.createElement("option");
    opt.value = s;
    opt.textContent = s;
    scenarioSelect.appendChild(opt);
  }
  scenarioSelect.disabled = false;
  if (savedUiState && savedUiState.city === citySelect.value && scenarios.includes(savedUiState.scenario)) {
    scenarioSelect.value = savedUiState.scenario;
    savedUiState = { ...savedUiState, scenario: null };
  } else if (scenarios.includes(DEFAULT_SCENARIO)) {
    scenarioSelect.value = DEFAULT_SCENARIO;
  }
  persistUiState();
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

    cb.addEventListener("change", persistUiState);

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
      persistUiState();
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
  persistUiState();
  debugLog(`run requested city=${city} scenario=${scenario} policies=${policies.join(", ")}`);

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
        if (cacheData.metrics && cacheData.metrics[key]?.replay_available) cachedPolicies.add(key);
      }
      debugLog(`cache probe found replay for: ${Array.from(cachedPolicies).join(", ") || "none"}`);
    }
  } catch (_) { /* server may not have any results yet — treat all as uncached */ }

  policies.forEach((p) => {
    if (cachedPolicies.has(p)) markPolicyCached(p);
    else markPolicyRunning(p);
  });

  let allResults;
  try {
    const resp = await fetch(`${API_BASE}/run-simulations`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        city_id: city,
        scenario_name: scenario,
        policies,
        force: document.getElementById("force-rerun-checkbox")?.checked ?? false,
      }),
    });
    if (!resp.ok) {
      const errData = await resp.json();
      throw new Error(errData.detail || resp.statusText);
    }
    const data = await resp.json();
    debugLog(`batch run response received: ${data.results?.length || 0} policies`);
    const resultsByPolicy = new Map((data.results || []).map((result) => [result.policy_name, result]));
    allResults = policies.map((policy) => {
      const result = resultsByPolicy.get(policy) || {
        policy_name: policy,
        metrics: { error: "Missing policy result from server." },
        replay_available: false,
        roadnet_log_available: false,
        elapsed_ms: null,
      };
      const elapsedMs = Number(result.elapsed_ms ?? 0);
      debugLog(
        `policy=${policy} replay=${!!result.replay_available} roadnet=${!!result.roadnet_log_available} ` +
        `elapsed=${(elapsedMs / 1000).toFixed(1)}s`
      );
      markPolicyDone(policy, elapsedMs, result.replay_available ? "done" : "error");
      return result;
    });
  } catch (err) {
    debugLog(`batch run failed: ${err.message}`);
    allResults = policies.map((policy) => {
      markPolicyDone(policy, 0, "error");
      return {
        policy_name: policy,
        metrics: { error: err.message },
        replay_available: false,
        roadnet_log_available: false,
        elapsed_ms: 0,
      };
    });
  }

  const firstSuccess = allResults.find((r) => r.roadnet_log_available);
  if (!firstSuccess) {
    showToast("No successful policy runs.", "error");
    runBtn.disabled = false;
    return;
  }

  try {
    await fetchAndStartVisualization(city, scenario, allResults);
    saveRecentRun({
      city,
      scenario,
      policies,
      durationsMsByPolicy: Object.fromEntries(
        allResults.map((result) => [
          result.policy_name,
          result.elapsed_ms ?? 0,
        ])
      ),
      replayAvailableByPolicy: Object.fromEntries(
        allResults.map((result) => [
          result.policy_name,
          !!result.replay_available,
        ])
      ),
      metricsByPolicy: Object.fromEntries(
        allResults.map((result) => [
          result.policy_name,
          result.metrics || {},
        ])
      ),
      viewLayout,
    });
  } catch (err) {
    debugLog(`visualization failed: ${err.message}`);
    showToast("Error: " + err.message, "error");
  } finally {
    runBtn.disabled = false;
  }
});

// ── Fetch data & start visualization ──────────────────────────────────────

async function fetchAndStartVisualization(city, scenario, policyResults) {
  debugLog(`viz start city=${city} scenario=${scenario}`);
  stopAnimation();
  destroyAllPanels();

  // Fetch the shared roadnet log (one call is enough – same static network).
  const firstPolicy = policyResults.find((r) => r.roadnet_log_available);
  const roadnetResp = await fetch(
    `${API_BASE}/roadnet-log/${city}/${scenario}/${firstPolicy.policy_name}`
  );
  if (!roadnetResp.ok) throw new Error("Failed to load roadnet log.");
  sharedRoadnetData = await roadnetResp.json();
  debugLog(`roadnet loaded from policy=${firstPolicy.policy_name}`);
  activeDistrictMap = null;
  try {
    const districtResp = await fetch(`${API_BASE}/cities/${city}/district-map`);
    if (districtResp.ok) activeDistrictMap = await districtResp.json();
    debugLog(`district map ${activeDistrictMap ? "loaded" : "missing"}`);
  } catch (_) {
    activeDistrictMap = null;
    debugLog("district map fetch failed");
  }
  const rn = sharedRoadnetData.static || sharedRoadnetData;
  console.log("[viz] roadnet loaded — nodes:", (rn.node || rn.nodes || []).length, "edges:", (rn.edge || rn.edges || []).length);

  // Fetch replay data per policy — sequentially to avoid holding all large
  // replay strings in memory simultaneously (each file can be 100s of MB).
  const replayPayloads = [];
  for (const result of policyResults.filter((r) => r.replay_available)) {
    debugLog(`fetching replay for ${result.policy_name}`);
    const replayResp = await fetch(
      `${API_BASE}/replay/${city}/${scenario}/${result.policy_name}`
    );
    if (!replayResp.ok) {
      console.warn("[viz] replay fetch failed for", result.policy_name, replayResp.status);
      continue;
    }
    const text = await replayResp.text();
    // The backend already ran a 30-second scenario, so consume the full replay as-is.
    const lines = text.split("\n").filter((l) => l.trim().length > 0);
    console.log(`[viz] replay loaded — policy: ${result.policy_name}, steps: ${lines.length}`);
    debugLog(`replay loaded for ${result.policy_name}: ${lines.length} steps`);
    replayPayloads.push({ policy_name: result.policy_name, metrics: result.metrics, lines });
  }
  console.log("[viz] replay payloads ready:", replayPayloads.map((p) => `${p.policy_name}(${p.lines.length}steps)`));
  if (!replayPayloads.length) {
    throw new Error("No replay data available for the selected policies.");
  }
  debugLog(`viz payloads ready: ${replayPayloads.map((p) => `${p.policy_name}(${p.lines.length})`).join(", ")}`);

  totalSteps  = Math.min(...replayPayloads.map((p) => p.lines.length));
  globalStep  = 0;
  frameElapsed = 0;
  paused = false;

  scrubber.max   = Math.max(0, totalSteps - 1);
  scrubber.value = 0;

  // Cache for view-mode toggling, then build panels limited by viewLayout.
  activeReplayPayloads = replayPayloads;
  renderSummaryCards(policyResults);
  renderPanelGrid();
  debugLog(`rendered ${Math.min(activeReplayPayloads.length, viewLayout)} panel(s)`);

  welcomeOverlay.classList.add("hidden");
  startAnimation();
}

function loadSavedUiState() {
  try {
    return JSON.parse(localStorage.getItem(STORAGE_KEYS.uiState) || "null");
  } catch (_) {
    return null;
  }
}

function persistUiState() {
  const payload = {
    city: citySelect.value || null,
    scenario: scenarioSelect.value || null,
    policies: getSelectedPolicies(),
    viewLayout,
    forceRerun: !!forceRerunCheckbox?.checked,
  };
  localStorage.setItem(STORAGE_KEYS.uiState, JSON.stringify(payload));
}

function restoreSavedPolicyState() {
  if (!savedUiState || !Array.isArray(savedUiState.policies)) return;
  for (const key of ALL_POLICY_KEYS) {
    const cb = document.getElementById(`cb-${key}`);
    if (!cb || cb.disabled) continue;
    cb.checked = savedUiState.policies.includes(key);
  }
  if (forceRerunCheckbox && typeof savedUiState.forceRerun === "boolean") {
    forceRerunCheckbox.checked = savedUiState.forceRerun;
  }
}

function restoreSavedViewLayout() {
  if (!savedUiState || !savedUiState.viewLayout) return;
  const btn = document.querySelector(`.view-mode-btn[data-panels="${savedUiState.viewLayout}"]`);
  if (btn) btn.click();
}

function restoreSavedUiSelections() {
  if (!savedUiState || !savedUiState.city) return;
  if (!Array.from(citySelect.options).some((opt) => opt.value === savedUiState.city)) return;
  suppressCityAutoDefault = true;
  citySelect.value = savedUiState.city;
  citySelect.dispatchEvent(new Event("change"));
}

function loadRecentRuns() {
  try {
    const parsed = JSON.parse(localStorage.getItem(STORAGE_KEYS.recentRuns) || "[]");
    return Array.isArray(parsed) ? parsed : [];
  } catch (_) {
    return [];
  }
}

function saveRecentRun(run) {
  const recentRuns = loadRecentRuns();
  const normalized = {
    id: `${run.city}__${run.scenario}__${Date.now()}`,
    city: run.city,
    scenario: run.scenario,
    policies: run.policies,
    durationsMsByPolicy: run.durationsMsByPolicy || {},
    replayAvailableByPolicy: run.replayAvailableByPolicy || {},
    metricsByPolicy: run.metricsByPolicy || {},
    viewLayout: run.viewLayout || 1,
    savedAt: new Date().toISOString(),
  };
  const nextRuns = [normalized, ...recentRuns].slice(0, MAX_RECENT_RUNS);
  localStorage.setItem(STORAGE_KEYS.recentRuns, JSON.stringify(nextRuns));
  renderRecentRuns();
}

function renderRecentRuns() {
  if (!recentRunsList) return;
  const recentRuns = loadRecentRuns();
  recentRunsList.innerHTML = "";
  if (!recentRuns.length) {
    recentRunsList.innerHTML = '<div class="recent-run-empty">No saved runs yet.</div>';
    return;
  }
  for (const run of recentRuns) {
    const card = document.createElement("div");
    card.className = "recent-run-card";
    const totalMs = Object.values(run.durationsMsByPolicy || {}).reduce((sum, value) => sum + Number(value || 0), 0);
    const policiesHtml = (run.policies || [])
      .map((policy) => {
        const meta = POLICY_META[policy];
        const label = meta ? meta.label : policy;
        const seconds = Number(run.durationsMsByPolicy?.[policy] || 0) / 1000;
        return `<span class="recent-run-chip">${label} · ${seconds.toFixed(1)}s</span>`;
      })
      .join("");
    card.innerHTML = `
      <div class="recent-run-top">
        <div class="recent-run-title">${run.city} / ${run.scenario}</div>
        <div class="recent-run-time">${formatSavedAt(run.savedAt)}</div>
      </div>
      <div class="recent-run-meta">Total runtime: ${(totalMs / 1000).toFixed(1)}s</div>
      <div class="recent-run-policies">${policiesHtml}</div>
      <div class="recent-run-actions">
        <button class="recent-run-btn" data-action="restore">Restore</button>
        <button class="recent-run-btn" data-action="open-cache">Open Cached</button>
      </div>
    `;
    card.querySelector('[data-action="restore"]').addEventListener("click", () => restoreRecentRun(run, false));
    card.querySelector('[data-action="open-cache"]').addEventListener("click", () => restoreRecentRun(run, true));
    recentRunsList.appendChild(card);
  }
}

async function restoreRecentRun(run, openCached) {
  savedUiState = {
    city: run.city,
    scenario: run.scenario,
    policies: run.policies || [],
    viewLayout: run.viewLayout || 1,
    forceRerun: false,
  };
  restoreSavedPolicyState();
  restoreSavedViewLayout();
  suppressCityAutoDefault = true;
  if (Array.from(citySelect.options).some((opt) => opt.value === run.city)) {
    citySelect.value = run.city;
    await citySelect.dispatchEvent(new Event("change"));
  }
  persistUiState();
  if (!openCached) return;
  try {
    const metricsResp = await fetch(`${API_BASE}/metrics/${run.city}/${run.scenario}`);
    if (!metricsResp.ok) throw new Error("Cached metrics not found.");
    const metricsData = await metricsResp.json();
    const policyResults = (run.policies || [])
      .filter((policy) => metricsData.metrics && metricsData.metrics[policy]?.replay_available)
      .map((policy) => ({
        policy_name: policy,
        metrics: metricsData.metrics[policy],
        replay_available: !!metricsData.metrics[policy]?.replay_available,
        roadnet_log_available: !!metricsData.metrics[policy]?.roadnet_log_available,
        elapsed_ms: Number(run.durationsMsByPolicy?.[policy] || 0),
      }));
    if (!policyResults.length) {
      showToast("No cached run data found for this selection.", "warn");
      return;
    }
    await fetchAndStartVisualization(run.city, run.scenario, policyResults);
  } catch (err) {
    showToast(`Failed to open cached run: ${err.message}`, "error");
  }
}

function formatSavedAt(savedAt) {
  if (!savedAt) return "saved";
  const date = new Date(savedAt);
  if (Number.isNaN(date.getTime())) return "saved";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
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
    panel.init(sharedRoadnetData, activeDistrictMap);
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

function renderSummaryCards(policyResults) {
  if (!summaryBar) return;
  summaryBar.innerHTML = "";
  if (!policyResults || !policyResults.length) {
    summaryBar.innerHTML = '<div class="summary-empty">Run a simulation to compare policy metrics.</div>';
    return;
  }

  const baseline = policyResults.find((result) => result.policy_name === "learned") || policyResults[0];
  for (const result of policyResults) {
    const meta = POLICY_META[result.policy_name] || { label: result.policy_name, color: 0x888888 };
    const metrics = result.metrics || {};
    const card = document.createElement("div");
    card.className = "summary-card";
    card.style.setProperty("--summary-color", `#${meta.color.toString(16).padStart(6, "0")}`);

    const queue = pickMetric(metrics, ["mean_waiting_vehicles", "avg_queue"]);
    const throughput = pickMetric(metrics, ["throughput"]);
    const travelTime = pickMetric(metrics, ["average_travel_time", "travel_time"]);
    const totalReturn = pickMetric(metrics, ["total_episode_return", "total_return", "episode_return"]);
    const activePct = pickMetric(metrics, ["percent_steps_with_active_guidance"]);

    const queueDelta = baseline === result ? null : diffMetric(queue, pickMetric(baseline.metrics || {}, ["mean_waiting_vehicles", "avg_queue"]), true);
    const throughputDelta = baseline === result ? null : diffMetric(throughput, pickMetric(baseline.metrics || {}, ["throughput"]), false);

    card.innerHTML = `
      <div class="summary-card-header">
        <div class="summary-card-title">${meta.label}</div>
        <div class="summary-card-subtitle">${result.policy_name}</div>
      </div>
      <div class="summary-grid">
        <div>
          <div class="summary-stat-label">Return</div>
          <div class="summary-stat-value">${formatMetric(totalReturn, 2)}</div>
        </div>
        <div>
          <div class="summary-stat-label">Queue</div>
          <div class="summary-stat-value">${formatMetric(queue, 1)}</div>
          ${queueDelta ? `<div class="summary-delta ${queueDelta.className}">${queueDelta.label}</div>` : ""}
        </div>
        <div>
          <div class="summary-stat-label">Throughput</div>
          <div class="summary-stat-value">${formatMetric(throughput, 0)}</div>
          ${throughputDelta ? `<div class="summary-delta ${throughputDelta.className}">${throughputDelta.label}</div>` : ""}
        </div>
        <div>
          <div class="summary-stat-label">Travel Time</div>
          <div class="summary-stat-value">${formatMetric(travelTime, 1)}</div>
        </div>
        <div>
          <div class="summary-stat-label">Guidance Active</div>
          <div class="summary-stat-value">${activePct == null ? "—" : `${(activePct * 100).toFixed(0)}%`}</div>
        </div>
        <div>
          <div class="summary-stat-label">Runtime</div>
          <div class="summary-stat-value">${formatDuration(result.elapsed_ms)}</div>
        </div>
      </div>
    `;
    summaryBar.appendChild(card);
  }
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
    this.overlayContainer = null;
    this.legendEl = null;
    this._ready     = false;
  }

  get meta() { return POLICY_META[this.policyName] || { label: this.policyName, color: 0x888888 }; }

  init(roadnetData, districtMap) {
    this._buildElement();
    this._initPixi(roadnetData, districtMap);
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

    this.legendEl = document.createElement("div");
    this.legendEl.className = "panel-map-legend hidden";
    this.legendEl.innerHTML = `
      <div class="panel-map-legend-title">Map</div>
      <div class="panel-map-legend-row"><span class="panel-map-swatch districts"></span><span>Districts</span></div>
      <div class="panel-map-legend-row"><span class="panel-map-swatch entry"></span><span>Entry roads</span></div>
      <div class="panel-map-legend-row"><span class="panel-map-swatch exit"></span><span>Exit roads</span></div>
      <div class="panel-map-legend-row"><span class="panel-map-swatch gateway"></span><span>Gateway nodes</span></div>
    `;
    this.canvasWrapper.appendChild(this.legendEl);

    this.element.appendChild(header);
    this.element.appendChild(this.canvasWrapper);
  }

  _initPixi(roadnetData, districtMap) {
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
      this._drawRoadnet(roadnetData, districtMap);

      // Remove spinner.
      const spinner = wrapper.querySelector(".panel-spinner");
      if (spinner) spinner.classList.add("hidden");
      this._ready = true;
      console.log(`[viz] panel ready — policy: ${this.policyName}`);
    });
  }

  _drawRoadnet(roadnetJson, districtMap) {
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

    this._drawOverlayLayers(districtMap);
    if (this.legendEl) {
      this.legendEl.classList.toggle("hidden", !districtMap);
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

  _drawOverlayLayers(districtMap) {
    if (this.overlayContainer) {
      this.overlayContainer.destroy(true);
      this.overlayContainer = null;
    }
    if (!districtMap || !districtMap.intersection_to_district) return;

    this.overlayContainer = new PIXI.Container();
    this.simulatorContainer.addChild(this.overlayContainer);

    const overlayGraphics = new PIXI.Graphics();
    this.overlayContainer.addChild(overlayGraphics);

    const intersectionToDistrict = districtMap.intersection_to_district || {};
    const gatewayNodeIds = new Set(districtMap.gateway_intersections || []);
    const gatewayEdgeIds = new Set(districtMap.gateway_roads || []);
    const entryRoadIds = new Set();
    const exitRoadIds = new Set();
    for (const district of districtMap.districts || []) {
      for (const roadId of district.entry_roads || []) entryRoadIds.add(roadId);
      for (const roadId of district.exit_roads || []) exitRoadIds.add(roadId);
    }

    const districtToPoints = {};
    for (const nodeId in this.nodes) {
      const districtId = intersectionToDistrict[nodeId];
      if (!districtId) continue;
      if (!districtToPoints[districtId]) districtToPoints[districtId] = [];
      districtToPoints[districtId].push(this.nodes[nodeId].point);
    }
    drawDistrictRegionFills(overlayGraphics, districtToPoints);

    for (const edgeId in this.edges) {
      const edge = this.edges[edgeId];
      const fromDistrict = intersectionToDistrict[edge.from.id];
      const toDistrict = intersectionToDistrict[edge.to.id];
      if (fromDistrict && toDistrict) {
        if (fromDistrict === toDistrict) {
          drawEdgePolyline(overlayGraphics, edge, 1.2, getDistrictColor(fromDistrict), 0.35);
        } else {
          drawEdgePolyline(overlayGraphics, edge, 1.8, DISTRICT_BORDER_COLOR, 0.55);
        }
      }

      const fromGateway = gatewayNodeIds.has(edge.from.id) || String(edge.from.id).startsWith("g_");
      const toGateway = gatewayNodeIds.has(edge.to.id) || String(edge.to.id).startsWith("g_");
      const looksGateway = gatewayEdgeIds.has(edgeId) || String(edgeId).includes("_g_") || String(edgeId).startsWith("r_g_");
      if (!fromGateway && !toGateway && !looksGateway) continue;

      const isEntry = entryRoadIds.has(edgeId) || (fromGateway && !toGateway);
      const isExit = exitRoadIds.has(edgeId) || (!fromGateway && toGateway);
      const color = isEntry ? ENTRY_EDGE_COLOR : (isExit ? EXIT_EDGE_COLOR : GATEWAY_EDGE_COLOR);
      const width = isEntry || isExit ? 5.2 : 4.0;
      const alpha = isEntry || isExit ? 0.72 : 0.56;
      drawEdgePolyline(overlayGraphics, edge, width, color, alpha);
    }

    for (const nodeId in this.nodes) {
      const districtId = intersectionToDistrict[nodeId];
      if (districtId) {
        const point = this.nodes[nodeId].point;
        overlayGraphics.beginFill(getDistrictColor(districtId), 0.48);
        overlayGraphics.drawCircle(point.x, point.y, 2.0);
        overlayGraphics.endFill();
      }
      if (gatewayNodeIds.has(nodeId) || String(nodeId).startsWith("g_")) {
        const point = this.nodes[nodeId].point;
        overlayGraphics.lineStyle(0);
        overlayGraphics.beginFill(GATEWAY_NODE_COLOR, 0.22);
        overlayGraphics.drawCircle(point.x, point.y, 26.0);
        overlayGraphics.endFill();
        overlayGraphics.lineStyle(2.0, 0x1f2937, 0.95);
        overlayGraphics.beginFill(GATEWAY_NODE_COLOR, 0.9);
        overlayGraphics.drawCircle(point.x, point.y, 9.0);
        overlayGraphics.endFill();
        overlayGraphics.lineStyle(0);
        overlayGraphics.beginFill(0xffffff, 0.9);
        overlayGraphics.drawCircle(point.x, point.y, 3.6);
        overlayGraphics.endFill();
      }
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

function getDistrictColor(districtId) {
  return DISTRICT_PALETTE[stringHash(districtId) % DISTRICT_PALETTE.length];
}

function cross2D(o, a, b) {
  return (a.x - o.x) * (b.y - o.y) - (a.y - o.y) * (b.x - o.x);
}

function convexHull(points) {
  if (!points || points.length <= 2) return points ? points.slice() : [];
  const sorted = points
    .slice()
    .sort((p, q) => (p.x === q.x ? p.y - q.y : p.x - q.x));
  const lower = [];
  for (const point of sorted) {
    while (lower.length >= 2 && cross2D(lower[lower.length - 2], lower[lower.length - 1], point) <= 0) {
      lower.pop();
    }
    lower.push(point);
  }
  const upper = [];
  for (let i = sorted.length - 1; i >= 0; i--) {
    const point = sorted[i];
    while (upper.length >= 2 && cross2D(upper[upper.length - 2], upper[upper.length - 1], point) <= 0) {
      upper.pop();
    }
    upper.push(point);
  }
  lower.pop();
  upper.pop();
  return lower.concat(upper);
}

function drawDistrictRegionFills(graphics, districtToPoints) {
  for (const districtId in districtToPoints) {
    const pts = districtToPoints[districtId];
    if (!pts || pts.length === 0) continue;
    const color = getDistrictColor(districtId);
    if (pts.length >= 3) {
      const hull = convexHull(pts);
      if (hull.length >= 3) {
        graphics.lineStyle(1.2, color, 0.28);
        graphics.beginFill(color, 0.09);
        graphics.moveTo(hull[0].x, hull[0].y);
        for (let i = 1; i < hull.length; i++) graphics.lineTo(hull[i].x, hull[i].y);
        graphics.lineTo(hull[0].x, hull[0].y);
        graphics.endFill();
        continue;
      }
    }
    if (pts.length === 2) {
      graphics.lineStyle(16, color, 0.08);
      graphics.drawLine(pts[0], pts[1]);
      continue;
    }
    graphics.lineStyle(0);
    graphics.beginFill(color, 0.1);
    graphics.drawCircle(pts[0].x, pts[0].y, 16);
    graphics.endFill();
  }
}

function drawEdgePolyline(graphics, edge, width, color, alpha) {
  graphics.lineStyle(width, color, alpha);
  for (let i = 1; i < edge.points.length; i++) {
    graphics.drawLine(edge.points[i - 1], edge.points[i]);
  }
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

function pickMetric(metrics, keys) {
  for (const key of keys) {
    const value = metrics?.[key];
    if (typeof value === "number" && Number.isFinite(value)) return value;
  }
  return null;
}

function formatMetric(value, digits = 1) {
  if (value == null || !Number.isFinite(value)) return "—";
  return Number(value).toFixed(digits);
}

function formatDuration(elapsedMs) {
  const seconds = Number(elapsedMs || 0) / 1000;
  return `${seconds.toFixed(1)}s`;
}

function diffMetric(value, baseline, lowerIsBetter) {
  if (value == null || baseline == null) return null;
  const delta = value - baseline;
  const good = lowerIsBetter ? delta < 0 : delta > 0;
  const neutral = Math.abs(delta) < 1e-9;
  return {
    className: neutral ? "neutral" : (good ? "good" : "bad"),
    label: `${delta >= 0 ? "+" : ""}${delta.toFixed(1)} vs DQN`,
  };
}

function debugLog(message) {
  const line = `[${new Date().toLocaleTimeString()}] ${message}`;
  console.log("[frontend]", message);
  if (!debugLogEl) return;
  const entry = document.createElement("div");
  entry.textContent = line;
  debugLogEl.prepend(entry);
  while (debugLogEl.childNodes.length > 80) {
    debugLogEl.removeChild(debugLogEl.lastChild);
  }
}

// Extend PIXI.Graphics with drawLine if not already present.
if (PIXI.Graphics && !PIXI.Graphics.prototype.drawLine) {
  PIXI.Graphics.prototype.drawLine = function(pA, pB) {
    this.moveTo(pA.x, pA.y);
    this.lineTo(pB.x, pB.y);
  };
}
