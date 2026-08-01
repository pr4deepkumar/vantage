/**
 * VANTAGE Console Frontend Engine
 * Handles API integration, dynamic Plotly chart rendering, and interactive UI logic.
 */

document.addEventListener("DOMContentLoaded", () => {
  initNavigation();
  initEventListeners();
  loadCurrentTab("overview");
});

// State Management
const state = {
  currentTab: "overview",
  overviewData: null,
  traces: [],
  driftData: null,
  abData: []
};

/* ==========================================================================
   Navigation & UI Routing
   ========================================================================== */

function initNavigation() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach(btn => {
    btn.addEventListener("click", () => {
      const tabName = btn.getAttribute("data-tab");
      switchTab(tabName);
    });
  });
}

function switchTab(tabName) {
  state.currentTab = tabName;

  // Update Nav Buttons
  document.querySelectorAll(".nav-item").forEach(btn => {
    btn.classList.toggle("active", btn.getAttribute("data-tab") === tabName);
  });

  // Update Tab Content Panels
  document.querySelectorAll(".tab-content").forEach(panel => {
    panel.classList.toggle("active", panel.id === `tab-${tabName}`);
  });

  // Update Header Titles
  const titles = {
    overview: { title: "Performance Overview", subtitle: "Real-time metrics, cost distribution, and latency tracking" },
    sandbox: { title: "Interactive Live Sandbox", subtitle: "Zero-cost simulation playground for multi-agent traces & automated judges" },
    traces: { title: "Trace Explorer", subtitle: "Filterable trace log table & span tree inspector" },
    drift: { title: "Drift Audits", subtitle: "Statistical distribution drift & rolling baseline alerts" },
    abtest: { title: "Prompt A/B Testing", subtitle: "Side-by-side prompt version quality comparison" }
  };

  if (titles[tabName]) {
    document.getElementById("page-title").innerText = titles[tabName].title;
    document.getElementById("page-subtitle").innerText = titles[tabName].subtitle;
  }

  loadCurrentTab(tabName);
}

function loadCurrentTab(tabName) {
  if (tabName === "overview") fetchOverviewData();
  else if (tabName === "sandbox") setupSandboxDefault();
  else if (tabName === "traces") fetchTracesData();
  else if (tabName === "drift") fetchDriftData();
  else if (tabName === "abtest") fetchABTestData();
}

function initEventListeners() {
  document.getElementById("btn-refresh").addEventListener("click", () => {
    loadCurrentTab(state.currentTab);
  });

  document.getElementById("btn-simulate").addEventListener("click", async () => {
    const btn = document.getElementById("btn-simulate");
    btn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Generating...`;
    btn.disabled = true;

    try {
      await fetch("/api/simulate", { method: "POST" });
      loadCurrentTab(state.currentTab);
    } catch (err) {
      console.error("Simulation error:", err);
    } finally {
      btn.innerHTML = `<i class="fa-solid fa-arrows-rotate"></i> Generate Mock Traces`;
      btn.disabled = false;
    }
  });

  // Filter Listeners in Trace Explorer
  document.getElementById("trace-search").addEventListener("input", filterTracesTable);
  document.getElementById("filter-status").addEventListener("change", fetchTracesData);
  document.getElementById("filter-version").addEventListener("change", fetchTracesData);

  // Drawer Close
  document.getElementById("drawer-close").addEventListener("click", closeTraceDrawer);
  document.getElementById("drawer-backdrop").addEventListener("click", closeTraceDrawer);

  // Live Sandbox Listeners
  initSandboxListeners();
}

/* ==========================================================================
   Tab 1: Performance Overview
   ========================================================================== */

async function fetchOverviewData() {
  try {
    const res = await fetch("/api/overview");
    const data = await res.json();
    state.overviewData = data;

    // Update KPI Cards
    const m = data.metrics;
    document.getElementById("kpi-traces").innerText = m.total_traces.toLocaleString();
    document.getElementById("kpi-spend").innerText = `$${m.total_spend.toFixed(4)}`;
    document.getElementById("kpi-latency").innerText = `${m.avg_latency} ms`;
    document.getElementById("kpi-ttft").innerText = m.avg_ttft > 0 ? `${m.avg_ttft} ms` : "N/A";
    document.getElementById("kpi-success").innerText = `${m.success_rate}%`;
    document.getElementById("kpi-judge").innerText = m.avg_judge_score > 0 ? `${m.avg_judge_score} / 5` : "N/A";

    // Render Plotly Charts
    renderSpendTrendChart(data.spend_series);
    renderModelPieChart(data.model_distribution);

  } catch (err) {
    console.error("Failed to fetch overview data:", err);
  }
}

function renderSpendTrendChart(series) {
  if (!series || series.length === 0) return;

  const dates = series.map(s => s.date);
  const spends = series.map(s => s.daily_spend);
  const counts = series.map(s => s.trace_count);

  const trace1 = {
    x: dates,
    y: spends,
    type: "scatter",
    mode: "lines+markers",
    name: "Daily Spend ($)",
    line: { color: "#6366f1", width: 3 },
    marker: { size: 6 }
  };

  const trace2 = {
    x: dates,
    y: counts,
    type: "bar",
    name: "Trace Volume",
    yaxis: "y2",
    marker: { color: "rgba(255, 255, 255, 0.1)" }
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 10, r: 40, l: 40, b: 40 },
    showlegend: true,
    legend: { font: { color: "#a1a1aa" }, orientation: "h", y: 1.1 },
    xaxis: { gridcolor: "rgba(255, 255, 255, 0.05)", tickfont: { color: "#a1a1aa" } },
    yaxis: { gridcolor: "rgba(255, 255, 255, 0.05)", tickfont: { color: "#a1a1aa" }, title: "Spend ($)" },
    yaxis2: { overlaying: "y", side: "right", showgrid: false, tickfont: { color: "#a1a1aa" }, title: "Volume" }
  };

  Plotly.newPlot("chart-spend-trend", [trace2, trace1], layout, { responsive: true, displayModeBar: false });
}

function renderModelPieChart(dist) {
  if (!dist || Object.keys(dist).length === 0) return;

  const labels = Object.keys(dist);
  const values = Object.values(dist);

  const trace = {
    labels: labels,
    values: values,
    type: "pie",
    hole: 0.5,
    marker: { colors: ["#4f46e5", "#10b981", "#f59e0b", "#ef4444", "#8b5cf6"] },
    textinfo: "percent+label",
    textfont: { color: "#ffffff" }
  };

  const layout = {
    paper_bgcolor: "transparent",
    plot_bgcolor: "transparent",
    margin: { t: 10, r: 10, l: 10, b: 10 },
    showlegend: false
  };

  Plotly.newPlot("chart-models", [trace], layout, { responsive: true, displayModeBar: false });
}

/* ==========================================================================
   Tab 2: Trace Explorer
   ========================================================================== */

async function fetchTracesData() {
  const status = document.getElementById("filter-status").value;
  const version = document.getElementById("filter-version").value;

  let url = `/api/traces?status=${status}`;
  if (version !== "all") url += `&prompt_version=${version}`;

  try {
    const res = await fetch(url);
    const traces = await res.json();
    state.traces = traces;

    // Populate Version Dropdown Filter if empty
    populateVersionFilter(traces);

    renderTracesTable(traces);
  } catch (err) {
    console.error("Failed to fetch traces:", err);
  }
}

function populateVersionFilter(traces) {
  const select = document.getElementById("filter-version");
  if (select.children.length > 1) return; // Already populated

  const versions = [...new Set(traces.map(t => t.prompt_version).filter(Boolean))];
  versions.forEach(v => {
    const opt = document.createElement("option");
    opt.value = v;
    opt.innerText = v;
    select.appendChild(opt);
  });
}

function renderTracesTable(traces) {
  const tbody = document.getElementById("traces-tbody");
  tbody.innerHTML = "";

  if (traces.length === 0) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align: center; color: var(--text-dim);">No traces found matching criteria.</td></tr>`;
    return;
  }

  traces.forEach(t => {
    const tr = document.createElement("tr");

    const dateStr = new Date(t.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const statusBadge = t.status === "success" 
      ? `<span class="badge badge-success"><i class="fa-solid fa-check"></i> Success</span>`
      : `<span class="badge badge-danger"><i class="fa-solid fa-xmark"></i> ${t.error_type || 'Error'}</span>`;

    const feedbackBadge = t.user_feedback === "up" 
      ? `<span style="color: var(--success);"><i class="fa-solid fa-thumbs-up"></i></span>`
      : t.user_feedback === "down"
      ? `<span style="color: var(--danger);"><i class="fa-solid fa-thumbs-down"></i></span>`
      : `<span style="color: var(--text-dim);">-</span>`;

    tr.innerHTML = `
      <td style="color: var(--text-muted); font-size: 0.8rem;">${dateStr}</td>
      <td><code style="color: #6366f1;">${t.id.substring(0, 8)}...</code></td>
      <td><span class="badge" style="background: rgba(255,255,255,0.06);">${t.span_kind || 'llm'}</span></td>
      <td><strong>${t.model || 'N/A'}</strong> <br><small style="color: var(--text-dim);">${t.prompt_version || ''}</small></td>
      <td>${(t.latency_ms || 0).toFixed(0)} ms</td>
      <td>${t.input_tokens || 0} / ${t.output_tokens || 0}</td>
      <td>$${(t.cost || 0).toFixed(5)}</td>
      <td>${statusBadge}</td>
      <td>${feedbackBadge}</td>
      <td>
        <button class="btn btn-secondary" style="padding: 4px 8px; font-size: 0.75rem;" onclick="openTraceDrawer('${t.id}')">
          <i class="fa-solid fa-eye"></i> View
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

function filterTracesTable() {
  const query = document.getElementById("trace-search").value.toLowerCase();
  const filtered = state.traces.filter(t => 
    t.id.toLowerCase().includes(query) ||
    (t.input && t.input.toLowerCase().includes(query)) ||
    (t.output && t.output.toLowerCase().includes(query))
  );
  renderTracesTable(filtered);
}

/* ==========================================================================
   Trace Inspector Drawer
   ========================================================================== */

async function openTraceDrawer(traceId) {
  const backdrop = document.getElementById("drawer-backdrop");
  const drawer = document.getElementById("trace-drawer");
  const content = document.getElementById("drawer-content");

  content.innerHTML = `<div style="text-align: center; padding: 40px;"><i class="fa-solid fa-spinner fa-spin fa-2x"></i></div>`;
  backdrop.classList.add("active");
  drawer.classList.add("active");

  try {
    const res = await fetch(`/api/traces/${traceId}`);
    const data = await res.json();
    const t = data.trace;

    let childrenHtml = "";
    if (data.children && data.children.length > 0) {
      childrenHtml = `
        <h4 style="margin-top: 20px; margin-bottom: 10px;"><i class="fa-solid fa-sitemap"></i> Child Spans Waterfall (${data.children.length})</h4>
        <div style="display: flex; flex-direction: column; gap: 8px;">
          ${data.children.map(c => `
            <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 8px; padding: 10px;">
              <div style="display: flex; justify-content: space-between; font-size: 0.8rem;">
                <strong>${c.span_kind} - ${c.model || 'span'}</strong>
                <span>${c.latency_ms} ms</span>
              </div>
              <small style="color: var(--text-dim); display: block; margin-top: 4px;">ID: ${c.id}</small>
            </div>
          `).join('')}
        </div>
      `;
    }

    content.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 16px;">
        <span class="badge badge-success">${t.status}</span>
        <div style="display: flex; gap: 8px;">
          <button class="btn btn-secondary" onclick="rateTrace('${t.id}', 'up')"><i class="fa-solid fa-thumbs-up"></i></button>
          <button class="btn btn-secondary" onclick="rateTrace('${t.id}', 'down')"><i class="fa-solid fa-thumbs-down"></i></button>
        </div>
      </div>

      <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-bottom: 16px;">
        <div><small style="color: var(--text-dim);">Latency</small><br><strong>${t.latency_ms} ms</strong></div>
        <div><small style="color: var(--text-dim);">Cost</small><br><strong>$${t.cost.toFixed(5)}</strong></div>
        <div><small style="color: var(--text-dim);">Model</small><br><strong>${t.model || 'N/A'}</strong></div>
        <div><small style="color: var(--text-dim);">Prompt Version</small><br><strong>${t.prompt_version || 'N/A'}</strong></div>
      </div>

      <h4 style="margin-bottom: 6px;"><i class="fa-solid fa-arrow-right-to-bracket"></i> Prompt Input</h4>
      <div class="code-block">${escapeHtml(t.input || '')}</div>

      <h4 style="margin-top: 16px; margin-bottom: 6px;"><i class="fa-solid fa-arrow-right-from-bracket"></i> Model Output</h4>
      <div class="code-block">${escapeHtml(t.output || '')}</div>

      ${childrenHtml}
    `;

  } catch (err) {
    content.innerHTML = `<div style="color: var(--danger);">Failed to load trace details.</div>`;
  }
}

function closeTraceDrawer() {
  document.getElementById("drawer-backdrop").classList.remove("active");
  document.getElementById("trace-drawer").classList.remove("active");
}

async function rateTrace(traceId, feedback) {
  try {
    await fetch(`/api/traces/${traceId}/feedback`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ feedback })
    });
    openTraceDrawer(traceId);
    fetchTracesData();
  } catch (err) {
    console.error("Feedback rating failed:", err);
  }
}

/* ==========================================================================
   Tab 3: Drift Audits
   ========================================================================== */

async function fetchDriftData() {
  try {
    const res = await fetch("/api/drift");
    const data = await res.json();
    state.driftData = data;

    document.getElementById("psi-latency").innerText = data.metrics.latency_psi.toFixed(4);
    document.getElementById("psi-cost").innerText = data.metrics.cost_psi.toFixed(4);

    const alertsContainer = document.getElementById("drift-alerts-list");
    alertsContainer.innerHTML = "";

    if (!data.alerts || data.alerts.length === 0) {
      alertsContainer.innerHTML = `
        <div style="background: rgba(16, 185, 129, 0.05); border: 1px solid rgba(16, 185, 129, 0.2); padding: 16px; border-radius: 8px; color: var(--success);">
          <i class="fa-solid fa-circle-check"></i> All pipeline operational baselines are healthy. No statistical drift detected.
        </div>
      `;
      return;
    }

    data.alerts.forEach(alert => {
      const div = document.createElement("div");
      div.className = "alert-item";
      div.style.cssText = "background: rgba(245, 158, 11, 0.08); border: 1px solid rgba(245, 158, 11, 0.3); padding: 16px; border-radius: 8px; margin-bottom: 12px;";
      div.innerHTML = `
        <div style="display: flex; justify-content: space-between; align-items: center;">
          <strong style="color: var(--warning);"><i class="fa-solid fa-triangle-exclamation"></i> ${alert.metric} Drift Detected</strong>
          <span class="badge badge-warning">Z-Score: ${alert.z_score || 'High'}</span>
        </div>
        <p style="margin-top: 6px; font-size: 0.85rem; color: var(--text-main);">${alert.message || 'Metric shifted significantly beyond standard baseline distribution.'}</p>
      `;
      alertsContainer.appendChild(div);
    });

  } catch (err) {
    console.error("Failed to fetch drift data:", err);
  }
}

/* ==========================================================================
   Tab 4: Prompt A/B Testing
   ========================================================================== */

async function fetchABTestData() {
  try {
    const res = await fetch("/api/ab_test");
    const data = await res.json();
    state.abData = data;

    const container = document.getElementById("ab-cards-container");
    container.innerHTML = "";

    if (data.length === 0) {
      container.innerHTML = `<div style="color: var(--text-dim);">No prompt versions available for side-by-side comparison.</div>`;
      return;
    }

    data.forEach(v => {
      const card = document.createElement("div");
      card.className = "ab-card";
      card.innerHTML = `
        <div class="ab-card-header">
          <h4><i class="fa-solid fa-code-branch"></i> Version ${v.version}</h4>
          <span class="badge badge-success">${v.total_traces} Traces</span>
        </div>
        <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 12px;">
          <div><small style="color: var(--text-dim);">Avg Latency</small><br><strong>${v.avg_latency_ms} ms</strong></div>
          <div><small style="color: var(--text-dim);">Avg Cost / Request</small><br><strong>$${v.avg_cost_usd}</strong></div>
          <div><small style="color: var(--text-dim);">Success Rate</small><br><strong>${v.success_rate}%</strong></div>
          <div><small style="color: var(--text-dim);">Judge Quality Score</small><br><strong>${v.avg_judge_score} / 5</strong></div>
          <div><small style="color: var(--text-dim);">User Feedback</small><br><strong style="color: var(--success);">👍 ${v.up_votes}</strong> / <strong style="color: var(--danger);">👎 ${v.down_votes}</strong></div>
          <div><small style="color: var(--text-dim);">Hallucination Rate</small><br><strong style="color: ${v.hallucination_rate > 5 ? 'var(--danger)' : 'var(--success)'}">${v.hallucination_rate}%</strong></div>
        </div>
      `;
      container.appendChild(card);
    });

  } catch (err) {
    console.error("Failed to fetch A/B test data:", err);
  }
}

// Helper: Escape HTML
function escapeHtml(str) {
  return str ? String(str).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;") : "";
}

/* ==========================================================================
   Tab 5: Live Sandbox & Zero-Cost Simulation Engine
   ========================================================================== */

const SANDBOX_PRESETS = {
  rag: {
    prompt: "Summarize the key discoveries made by the James Webb Space Telescope in 2025.",
    rag: "Verified Dossier (2025): JWST discovered 3 temperate exoplanets in the habitable zone of star LHS-1140 and detected atmospheric carbon signatures.",
    scenario: "rag",
    model: "claude-3-5-sonnet-20241022",
    version: "v1.1.0"
  },
  hallucination: {
    prompt: "According to our enterprise policy, what is the refund window for annual plans?",
    rag: "Enterprise Terms (2025): Strict 14-day refund window from purchase. No exceptions after 14 days.",
    scenario: "rag",
    model: "claude-3-5-sonnet-20241022",
    version: "v1.0.0"
  },
  code: {
    prompt: "Write a high-performance Python class for an async rate limiter using token bucket algorithm with asyncio.",
    rag: "",
    scenario: "coder",
    model: "claude-3-5-sonnet-20241022",
    version: "v1.1.0"
  },
  support: {
    prompt: "A customer asks: 'How do I configure distributed tracing with contextvars in VANTAGE?'",
    rag: "VANTAGE Docs: Use `with trace_span('Agent Name', span_kind='agent'):` to auto-propagate parent span IDs across thread-safe contextvars.",
    scenario: "agent",
    model: "claude-3-5-haiku-20241022",
    version: "v1.1.0"
  }
};

let currentSelectedPreset = "rag";

function setupSandboxDefault() {
  if (!document.getElementById("sandbox-prompt").value) {
    applySandboxPreset("rag");
  }
}

function initSandboxListeners() {
  // Mode selection (Simulated vs Live)
  const modeRadios = document.querySelectorAll('input[name="sandbox-mode"]');
  modeRadios.forEach(radio => {
    radio.addEventListener("change", (e) => {
      const mode = e.target.value;
      document.querySelectorAll(".mode-option").forEach(opt => opt.classList.remove("active"));
      const parentLabel = document.getElementById(`mode-opt-${mode}`);
      if (parentLabel) parentLabel.classList.add("active");

      const apiKeyGroup = document.getElementById("api-key-group");
      if (apiKeyGroup) {
        apiKeyGroup.style.display = mode === "live" ? "block" : "none";
      }
    });
  });

  // Preset Chips
  const chips = document.querySelectorAll(".chip[data-preset]");
  chips.forEach(chip => {
    chip.addEventListener("click", () => {
      chips.forEach(c => c.classList.remove("active"));
      chip.classList.add("active");
      const presetKey = chip.getAttribute("data-preset");
      applySandboxPreset(presetKey);
    });
  });

  // Execute button
  const runBtn = document.getElementById("btn-run-sandbox");
  if (runBtn) {
    runBtn.addEventListener("click", executeSandboxTrace);
  }
}

function applySandboxPreset(key) {
  const preset = SANDBOX_PRESETS[key];
  if (!preset) return;
  currentSelectedPreset = key;

  document.getElementById("sandbox-prompt").value = preset.prompt;
  const ragInput = document.getElementById("sandbox-rag");
  if (ragInput) ragInput.value = preset.rag || "";

  const ragGroup = document.getElementById("rag-context-group");
  if (ragGroup) {
    ragGroup.style.display = preset.scenario === "coder" ? "none" : "block";
  }

  const modelSelect = document.getElementById("sandbox-model");
  if (modelSelect) modelSelect.value = preset.model;

  const versionSelect = document.getElementById("sandbox-version");
  if (versionSelect) versionSelect.value = preset.version;
}

async function executeSandboxTrace() {
  const prompt = document.getElementById("sandbox-prompt").value.trim();
  if (!prompt) {
    alert("Please enter a prompt to run.");
    return;
  }

  const mode = document.querySelector('input[name="sandbox-mode"]:checked').value;
  const apiKey = document.getElementById("sandbox-api-key") ? document.getElementById("sandbox-api-key").value.trim() : "";
  const model = document.getElementById("sandbox-model").value;
  const version = document.getElementById("sandbox-version").value;
  const ragContext = document.getElementById("sandbox-rag") ? document.getElementById("sandbox-rag").value.trim() : null;
  const scenario = SANDBOX_PRESETS[currentSelectedPreset] ? SANDBOX_PRESETS[currentSelectedPreset].scenario : "rag";

  if (mode === "live" && !apiKey) {
    alert("Please enter your Anthropic API Key for Live Mode, or switch back to ⚡ Free Simulated Mode.");
    return;
  }

  const runBtn = document.getElementById("btn-run-sandbox");
  const statusEl = document.getElementById("sandbox-status");
  const outputArea = document.getElementById("sandbox-output-area");

  runBtn.disabled = true;
  runBtn.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Executing Trace Pipeline...`;
  statusEl.innerHTML = `<span class="dot running"></span> Running Pipeline`;

  outputArea.innerHTML = `
    <div style="padding: 30px; text-align: center; color: var(--text-muted);">
      <i class="fa-solid fa-circle-notch fa-spin fa-2x" style="color: var(--primary-hover); margin-bottom: 12px;"></i>
      <p>Propagating thread-safe contextvars & recording telemetry spans...</p>
    </div>
  `;

  try {
    const res = await fetch("/api/sandbox/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        prompt: prompt,
        scenario_type: scenario,
        model: model,
        prompt_version: version,
        mode: mode,
        rag_context: ragContext,
        api_key: apiKey || null
      })
    });

    if (!res.ok) {
      const errData = await res.json();
      throw new Error(errData.detail || "Trace execution failed");
    }

    const data = await res.json();
    renderSandboxResult(data);
    statusEl.innerHTML = `<span class="dot ready"></span> Completed (200 OK)`;
  } catch (err) {
    console.error("Sandbox execution error:", err);
    outputArea.innerHTML = `
      <div style="padding: 20px; background: rgba(239, 68, 68, 0.1); border: 1px solid rgba(239, 68, 68, 0.3); border-radius: 8px; color: #f87171;">
        <strong><i class="fa-solid fa-circle-exclamation"></i> Execution Error:</strong>
        <p style="margin-top: 6px; font-size: 0.85rem;">${escapeHtml(err.message)}</p>
      </div>
    `;
    statusEl.innerHTML = `<span class="dot" style="background: var(--danger)"></span> Failed`;
  } finally {
    runBtn.disabled = false;
    runBtn.innerHTML = `<i class="fa-solid fa-play"></i> Execute Live Pipeline Trace`;
  }
}

function renderSandboxResult(data) {
  const outputArea = document.getElementById("sandbox-output-area");
  const recent = data.recent_traces || [];
  const primaryTrace = recent[0] || {};
  
  const latency = primaryTrace.latency_ms ? `${Math.round(primaryTrace.latency_ms)} ms` : "342 ms";
  const cost = primaryTrace.cost !== undefined ? `$${Number(primaryTrace.cost).toFixed(5)}` : "$0.00018";
  const tokens = primaryTrace.input_tokens ? `${primaryTrace.input_tokens} / ${primaryTrace.output_tokens}` : "42 / 128";
  const judgeScore = primaryTrace.judge_score !== null && primaryTrace.judge_score !== undefined ? primaryTrace.judge_score : 5;
  const isHallucinated = primaryTrace.hallucination_flag === true;

  const traceId = data.root_trace_id || (primaryTrace.id || "trc_live_demo");

  outputArea.innerHTML = `
    <div class="sandbox-result-card">
      <div class="result-kpi-row">
        <div class="result-kpi">
          <div class="result-kpi-label">Latency</div>
          <div class="result-kpi-val" style="color: #60a5fa;">${latency}</div>
        </div>
        <div class="result-kpi">
          <div class="result-kpi-label">Token Cost</div>
          <div class="result-kpi-val" style="color: #34d399;">${cost}</div>
        </div>
        <div class="result-kpi">
          <div class="result-kpi-label">Tokens (In/Out)</div>
          <div class="result-kpi-val">${tokens}</div>
        </div>
        <div class="result-kpi">
          <div class="result-kpi-label">Judge Quality</div>
          <div class="result-kpi-val" style="color: ${isHallucinated ? 'var(--danger)' : '#fbbf24'};">
            ${isHallucinated ? '⚠️ Flagged' : `⭐ ${judgeScore}/5`}
          </div>
        </div>
      </div>

      <div style="margin-bottom: 12px;">
        <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted); margin-bottom: 6px;">
          <i class="fa-solid fa-sitemap"></i> Distributed Span Waterfall
        </div>
        <div class="trace-flow-waterfall">
          <div class="flow-step">
            <span class="flow-title"><i class="fa-solid fa-robot text-primary"></i> ${escapeHtml(data.scenario.toUpperCase())} Agent Orchestrator</span>
            <span class="flow-meta"><span class="badge badge-success">200 OK</span></span>
          </div>
          ${data.scenario === "rag" || data.scenario === "agent" ? `
          <div class="flow-step tool">
            <span class="flow-title"><i class="fa-solid fa-database text-info"></i> Vector Index RAG Retrieval</span>
            <span class="flow-meta">12 ms</span>
          </div>` : ''}
          <div class="flow-step llm">
            <span class="flow-title"><i class="fa-solid fa-microchip text-warning"></i> ${escapeHtml(data.model)}</span>
            <span class="flow-meta">${latency}</span>
          </div>
        </div>
      </div>

      <div style="margin-bottom: 14px;">
        <div style="font-size: 0.8rem; font-weight: 600; color: var(--text-muted); margin-bottom: 4px;">
          Model Output Response:
        </div>
        <div class="code-block">${escapeHtml(primaryTrace.output || "Generated pipeline output recorded successfully.")}</div>
      </div>

      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span style="font-size: 0.75rem; color: var(--text-dim);">Trace ID: <code>${traceId}</code></span>
        <button class="btn btn-outline" style="padding: 6px 12px; font-size: 0.78rem;" onclick="inspectLiveTrace('${traceId}')">
          <i class="fa-solid fa-arrow-up-right-from-square"></i> Inspect Waterfall Tree
        </button>
      </div>
    </div>
  `;
}

function inspectLiveTrace(traceId) {
  switchTab("traces");
  setTimeout(() => {
    openTraceDrawer(traceId);
  }, 200);
}
