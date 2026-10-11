/**
 * VANTAGE console. Every element is built with DOM APIs and textContent, never innerHTML from data,
 * so span inputs, model names and prompt versions can't inject markup.
 */
(() => {
  "use strict";

  // ---------------------------------------------------------------------------
  // Formatting and geometry (pure; exported for tests)
  // ---------------------------------------------------------------------------

  const INT = new Intl.NumberFormat("en-US", { maximumFractionDigits: 0 });

  function formatInt(value) {
    return value == null ? "—" : INT.format(value);
  }

  function formatMs(value) {
    if (value == null) return "—";
    if (value > 0 && value < 1) return "<1 ms";
    return `${INT.format(value)} ms`;
  }

  /** Two significant digits under a dollar, cents above: $0.16, $0.00048, $12.40. */
  function formatUsd(value) {
    if (value == null) return "—";
    if (value === 0) return "$0";
    const abs = Math.abs(value);
    if (abs >= 1) return `$${value.toLocaleString("en-US", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}`;
    const decimals = Math.min(8, Math.max(2, 1 - Math.floor(Math.log10(abs))));
    return `$${value.toFixed(decimals)}`;
  }

  function formatPct(value, digits = 1) {
    return value == null ? "—" : `${value.toFixed(digits)}%`;
  }

  /** ISO timestamps carry microseconds; keep milliseconds so every browser parses them. */
  function parseTime(iso) {
    return Date.parse(String(iso).replace(/(\.\d{3})\d+/, "$1"));
  }

  function formatClock(iso) {
    return new Date(parseTime(iso)).toLocaleString("en-US", {
      month: "short",
      day: "numeric",
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
      hour12: false,
    });
  }

  function formatDay(day) {
    return new Date(`${day}T00:00:00Z`).toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" });
  }

  /** A readable name for a span: the step name, or the question an LLM call was asked. */
  function spanName(span) {
    const input = span.input || "";
    const step = input.match(/^Span step: ([\s\S]*)$/);
    if (step) return step[1];
    const task = input.match(/^Execute Agent Task: ([\s\S]*)$/);
    if (task) return task[1];
    const user = input.replace(/^\[System:[\s\S]*?\]\n/, "").replace(/^User: /, "");
    const question = user.match(/Question: ([^\n]*)/);
    return (question ? question[1] : user).split("\n")[0].trim() || span.id;
  }

  /** Output tokens per second while the model was generating (after the first token, if streamed). */
  function throughput(span) {
    if (!span.output_tokens || !span.latency_ms) return null;
    const generating = span.ttft_ms != null ? span.latency_ms - span.ttft_ms : span.latency_ms;
    return generating > 0 ? span.output_tokens / (generating / 1000) : null;
  }

  /** Clean axis ticks from 0 to at least ``max``. */
  function niceTicks(max, count = 4) {
    if (!(max > 0)) return [0, 1];
    const rough = max / count;
    const magnitude = 10 ** Math.floor(Math.log10(rough));
    const step = [1, 2, 2.5, 5, 10].map((m) => m * magnitude).find((s) => s >= rough);
    const top = Math.ceil(max / step - 1e-9) * step;
    const ticks = [];
    for (let i = 0; i * step <= top + step * 1e-6; i++) ticks.push(Number((i * step).toPrecision(12)));
    return ticks;
  }

  /**
   * Geometry of a trace's spans on one time axis. ``spans[0]`` is the root. Offsets and widths are
   * percentages of the trace's total duration; ``split`` is where the first token arrived, as a
   * percentage of an LLM span's own bar.
   */
  function waterfallRows(spans) {
    const root = spans[0];
    const t0 = parseTime(root.timestamp);
    const startOf = (span) => Math.max(0, parseTime(span.timestamp) - t0);
    const total = Math.max(1, root.latency_ms || 0, ...spans.map((s) => startOf(s) + (s.latency_ms || 0)));
    const rows = spans.map((span, i) => {
      const start = i === 0 ? 0 : startOf(span);
      const duration = span.latency_ms || 0;
      const ttft = span.span_kind === "llm" && span.ttft_ms != null ? Math.min(span.ttft_ms, duration) : null;
      return {
        span,
        depth: i === 0 ? 0 : 1,
        start,
        duration,
        ttft,
        left: (100 * start) / total,
        width: (100 * duration) / total,
        split: ttft != null && duration > 0 ? (100 * ttft) / duration : null,
      };
    });
    const [first, ...rest] = rows;
    rest.sort((a, b) => a.start - b.start);
    return { total, rows: [first, ...rest] };
  }

  /**
   * How ``value`` compares with ``base``. ``points`` compares percentages in percentage points.
   * Returns the text and whether the change is better, worse or flat.
   */
  function compareMetric(base, value, { lowerIsBetter = false, points = false, format = String } = {}) {
    if (base == null || value == null) return { text: "—", tone: "flat" };
    const diff = value - base;
    if (Math.abs(diff) < 1e-9) return { text: "No change", tone: "flat" };
    const sign = diff > 0 ? "+" : "−";
    let text;
    if (points) {
      text = `${sign}${Math.abs(diff).toFixed(1)} pts`;
    } else {
      text = `${sign}${format(Math.abs(diff))}`;
      if (base !== 0) text += ` (${sign}${Math.abs((100 * diff) / base).toFixed(0)}%)`;
    }
    return { text, tone: diff < 0 === lowerIsBetter ? "better" : "worse" };
  }

  /**
   * Which of ``count`` x positions get a label: every ``every``-th one, and always the last. When the
   * last would crowd the label before it, that earlier label goes instead.
   */
  function labelIndices(count, every) {
    const indices = [];
    for (let i = 0; i < count; i += every) indices.push(i);
    const last = count - 1;
    if (count > 0 && indices[indices.length - 1] !== last) {
      if (indices.length > 1 && last - indices[indices.length - 1] < every) indices.pop();
      indices.push(last);
    }
    return indices;
  }

  /** PSI bands: under 0.1 stable, 0.1 to 0.2 moderate, 0.2 or more significant. */
  function psiLevel(psi) {
    if (psi == null) return null;
    if (psi < 0.1) return "Stable";
    if (psi < 0.2) return "Moderate shift";
    return "Significant shift";
  }

  /** A smooth path through the points that never overshoots them (monotone cubic, as d3.curveMonotoneX). */
  function monotonePath(points) {
    const n = points.length;
    if (n === 0) return "";
    if (n === 1) return `M${points[0][0]},${points[0][1]}`;
    const dx = [];
    const slope = [];
    for (let i = 0; i < n - 1; i++) {
      dx.push(points[i + 1][0] - points[i][0]);
      slope.push((points[i + 1][1] - points[i][1]) / dx[i]);
    }
    const tangent = [slope[0]];
    for (let i = 1; i < n - 1; i++) tangent.push(slope[i - 1] * slope[i] > 0 ? (slope[i - 1] + slope[i]) / 2 : 0);
    tangent.push(slope[n - 2]);
    for (let i = 0; i < n - 1; i++) {
      if (slope[i] === 0) {
        tangent[i] = 0;
        tangent[i + 1] = 0;
        continue;
      }
      const a = tangent[i] / slope[i];
      const b = tangent[i + 1] / slope[i];
      const h = a * a + b * b;
      if (h > 9) {
        const t = 3 / Math.sqrt(h);
        tangent[i] = t * a * slope[i];
        tangent[i + 1] = t * b * slope[i];
      }
    }
    let d = `M${points[0][0]},${points[0][1]}`;
    for (let i = 0; i < n - 1; i++) {
      const [x0, y0] = points[i];
      const [x1, y1] = points[i + 1];
      const third = dx[i] / 3;
      d += `C${x0 + third},${y0 + tangent[i] * third},${x1 - third},${y1 - tangent[i + 1] * third},${x1},${y1}`;
    }
    return d;
  }

  // ---------------------------------------------------------------------------
  // DOM helpers
  // ---------------------------------------------------------------------------

  const SVG_NS = "http://www.w3.org/2000/svg";
  const $ = (id) => document.getElementById(id);

  function append(el, children) {
    for (const child of children.flat()) {
      if (child == null || child === false) continue;
      el.append(child instanceof Node ? child : document.createTextNode(String(child)));
    }
    return el;
  }

  function h(tag, props = {}, ...children) {
    const el = document.createElement(tag);
    for (const [key, value] of Object.entries(props)) {
      if (value == null || value === false) continue;
      if (key === "class") el.className = value;
      else if (key === "style") Object.assign(el.style, value);
      else if (key === "dataset") Object.assign(el.dataset, value);
      else if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value === true ? "" : value);
    }
    return append(el, children);
  }

  function s(tag, attrs = {}, ...children) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [key, value] of Object.entries(attrs)) {
      if (value == null) continue;
      if (key.startsWith("on")) el.addEventListener(key.slice(2), value);
      else el.setAttribute(key, value);
    }
    return append(el, children);
  }

  /** Replace an element's children, skipping empty ones (the native replaceChildren prints "null"). */
  function fill(el, ...children) {
    el.replaceChildren();
    return append(el, children);
  }

  function icon(name) {
    return s("svg", { class: "icon", "aria-hidden": "true" }, s("use", { href: `#i-${name}` }));
  }

  async function api(path, options) {
    const response = await fetch(path, options);
    const body = await response.json().catch(() => ({}));
    if (!response.ok) {
      const detail = Array.isArray(body.detail) ? body.detail.map((d) => d.msg).join("; ") : body.detail;
      throw new Error(detail || `${response.status} ${response.statusText}`);
    }
    return body;
  }

  // ---------------------------------------------------------------------------
  // Tooltip
  // ---------------------------------------------------------------------------

  function showTooltip(x, y, title, rows) {
    const tip = $("tooltip");
    tip.replaceChildren(
      h("div", { class: "tooltip-title" }, title),
      ...rows.map((row) =>
        h(
          "div",
          { class: "tooltip-row" },
          row.key ? h("span", { class: `key-line ${row.key}` }) : null,
          h("span", { class: "tooltip-value" }, row.value),
          row.name ? h("span", { class: "tooltip-name" }, row.name) : null,
        ),
      ),
    );
    tip.hidden = false;
    const { width, height } = tip.getBoundingClientRect();
    const left = Math.min(x + 12, window.innerWidth - width - 8);
    const top = y + 12 + height > window.innerHeight - 8 ? y - height - 12 : y + 12;
    tip.style.left = `${Math.max(8, left)}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }

  function hideTooltip() {
    $("tooltip").hidden = true;
  }

  function pointFor(event, target) {
    if (event && "clientX" in event && event.clientX) return [event.clientX, event.clientY];
    const box = target.getBoundingClientRect();
    return [box.left + box.width / 2, box.top];
  }

  // ---------------------------------------------------------------------------
  // Shared pieces
  // ---------------------------------------------------------------------------

  function stat({ label, value, unit, note, tone }) {
    return h(
      "div",
      { class: "stat" },
      h("div", { class: "stat-label" }, label),
      h("div", { class: "stat-value" }, value, unit ? h("span", { class: "stat-unit" }, unit) : null),
      note ? h("div", { class: `stat-note${tone ? ` is-${tone}` : ""}` }, note) : null,
    );
  }

  function statusLabel(span) {
    if (span.status === "error") return h("span", { class: "status is-error" }, span.error_type || "error");
    return h("span", { class: "status" }, span.status || "—");
  }

  function groundingText(span) {
    if (!span.rag_context) return { text: "No context to check", tone: null };
    if (span.hallucination_flag === true) return { text: "Unsupported claims", tone: "negative" };
    if (span.hallucination_flag === false) return { text: "Supported by the context", tone: "better" };
    return { text: "Not checked", tone: null };
  }

  function figure(label, value, tone) {
    return h("div", { class: "figure" }, h("dt", {}, label), h("dd", { class: tone ? `is-${tone}` : null }, value));
  }

  function llmFigures(span) {
    const grounding = groundingText(span);
    const rate = throughput(span);
    return h(
      "dl",
      { class: "figures" },
      figure("Latency", formatMs(span.latency_ms)),
      figure("Time to first token", span.ttft_ms != null ? formatMs(span.ttft_ms) : "Not streamed"),
      figure("Throughput", rate != null ? `${formatInt(rate)} tokens/s` : "—"),
      figure("Tokens in / out", `${formatInt(span.input_tokens)} / ${formatInt(span.output_tokens)}`),
      figure("Cost", formatUsd(span.cost)),
      figure("Stop reason", span.stop_reason || "—"),
      figure("Grounding", grounding.text, grounding.tone),
      span.judge_score != null ? figure("Judge score", `${span.judge_score} / 5`) : null,
    );
  }

  function renderWaterfall(spans, currentId) {
    const { total, rows } = waterfallRows(spans);
    const ticks = niceTicks(total, 4).filter((t) => t <= total);
    const gridLines = () => ticks.slice(1).map((t) => h("span", { class: "wf-grid", style: { left: `${(100 * t) / total}%` } }));

    const ruler = h(
      "div",
      { class: "wf-row wf-ruler", "aria-hidden": "true" },
      h("span"),
      h(
        "div",
        { class: "wf-lane" },
        ticks.map((t, i) =>
          h(
            "span",
            {
              class: "wf-tick",
              style:
                i === 0
                  ? { left: "0" }
                  : (100 * t) / total > 92
                    ? { right: `${100 - (100 * t) / total}%` }
                    : { left: `${(100 * t) / total}%`, transform: "translateX(-50%)" },
            },
            formatMs(t),
          ),
        ),
      ),
      h("span"),
    );

    const body = rows.map((row) => {
      const { span } = row;
      const name = spanName(span);
      const segments =
        row.split != null
          ? [
              h("span", { class: "wf-seg", style: { flex: `${row.split} 1 0` } }),
              h("span", { class: "wf-seg gen", style: { flex: `${100 - row.split} 1 0` } }),
            ]
          : [h("span", { class: `wf-seg${span.span_kind === "llm" ? " gen" : ""}`, style: { flex: "1 1 0" } })];
      const describe = [
        `${span.span_kind} ${name}`,
        `starts at ${formatMs(row.start)}`,
        `takes ${formatMs(row.duration)}`,
        row.ttft != null ? `first token at ${formatMs(row.ttft)}` : null,
      ]
        .filter(Boolean)
        .join(", ");
      const tipRows = () => [
        { value: formatMs(row.duration), name: "duration" },
        { value: `+${formatMs(row.start)}`, name: "from the start of the trace" },
        row.ttft != null ? { value: formatMs(row.ttft), name: "to the first token", key: "key-neutral" } : null,
        row.ttft != null ? { value: formatMs(row.duration - row.ttft), name: "generating", key: "key-accent" } : null,
        span.status === "error" ? { value: span.error_type || "error", name: "status" } : null,
      ].filter(Boolean);
      const bar = h(
        "span",
        {
          class: "wf-bar",
          tabindex: "0",
          role: "img",
          "aria-label": describe,
          style: { left: `${row.left}%`, width: `max(2px, ${row.width}%)` },
          onpointermove: (e) => showTooltip(e.clientX, e.clientY, `${span.span_kind} · ${name}`, tipRows()),
          onfocus: (e) => showTooltip(...pointFor(null, e.currentTarget), `${span.span_kind} · ${name}`, tipRows()),
          onpointerleave: hideTooltip,
          onblur: hideTooltip,
        },
        segments,
      );
      return h(
        "div",
        { class: `wf-row${span.id === currentId ? " is-current" : ""}`, style: { "--depth": String(row.depth) } },
        h("div", { class: "wf-name" }, h("span", { class: "wf-kind" }, span.span_kind), h("span", { class: "wf-label", title: name }, name)),
        h("div", { class: "wf-lane" }, gridLines(), bar),
        h("div", { class: "wf-dur" }, formatMs(row.duration)),
      );
    });

    const legend = h(
      "div",
      { class: "legend wf-legend" },
      h("span", { class: "legend-item" }, h("span", { class: "key-swatch key-accent" }), "Model generating"),
      h("span", { class: "legend-item" }, h("span", { class: "key-swatch key-neutral" }), "Waiting for the first token, or a pipeline step"),
    );
    return h("div", { class: "waterfall" }, ruler, body, legend);
  }

  // ---------------------------------------------------------------------------
  // Overview
  // ---------------------------------------------------------------------------

  let latencyDays = [];

  function renderLatencyChart(host, days) {
    host.replaceChildren();
    const points = days.filter((d) => d.p50_latency != null);
    if (!points.length) {
      host.append(h("p", { class: "empty" }, "No successful calls recorded yet"));
      return;
    }
    const width = Math.max(280, host.clientWidth);
    const height = host.clientHeight || 296;
    const narrow = width < 520; // end labels need room; the legend names the lines either way
    const m = { top: 12, right: narrow ? 16 : 104, bottom: 28, left: 56 };
    const iw = width - m.left - m.right;
    const ih = height - m.top - m.bottom;
    const series = [
      { key: "p95_latency", label: "p95", cls: "accent" },
      { key: "p50_latency", label: "p50", cls: "neutral" },
    ];
    const ticks = niceTicks(Math.max(...points.map((d) => d.p95_latency)), 4);
    const top = ticks[ticks.length - 1];
    const x = (i) => m.left + (points.length === 1 ? iw / 2 : (i / (points.length - 1)) * iw);
    const y = (v) => m.top + ih - (v / top) * ih;

    const last = points[points.length - 1];
    const svg = s("svg", {
      width,
      height,
      viewBox: `0 0 ${width} ${height}`,
      role: "img",
      "aria-label": `Latency per day over ${points.length} days. Latest p95 ${formatMs(last.p95_latency)}, p50 ${formatMs(last.p50_latency)}.`,
    });

    for (const t of ticks) {
      svg.append(s("line", { class: "grid-line", x1: m.left, x2: m.left + iw, y1: y(t), y2: y(t) }));
      svg.append(s("text", { class: "axis-label", x: m.left - 8, y: y(t), "text-anchor": "end", "dominant-baseline": "middle" }, formatInt(t)));
    }
    const every = Math.max(1, Math.ceil(points.length / Math.max(2, Math.floor(iw / 64))));
    for (const i of labelIndices(points.length, every)) {
      svg.append(
        s("text", { class: "axis-label", x: x(i), y: height - 8, "text-anchor": "middle" }, formatDay(points[i].date)),
      );
    }

    const crosshair = s("line", { class: "crosshair", y1: m.top, y2: m.top + ih, visibility: "hidden" });
    svg.append(crosshair);

    for (const line of series) {
      svg.append(s("path", { class: `series series-${line.cls}`, d: monotonePath(points.map((d, i) => [x(i), y(d[line.key])])) }));
    }
    const ends = series.map((line) => ({ ...line, y: y(last[line.key]), value: last[line.key] }));
    const labelsFit = !narrow && Math.abs(ends[0].y - ends[1].y) >= 16;
    for (const end of ends) {
      svg.append(s("circle", { class: `end-dot ${end.cls}`, cx: x(points.length - 1), cy: end.y, r: 4 }));
      if (labelsFit) {
        svg.append(s("text", { class: "end-label", x: x(points.length - 1) + 12, y: end.y, "dominant-baseline": "middle" }, `${end.label} ${formatMs(end.value)}`));
      }
    }

    // Hover and keyboard: one column per day; the crosshair snaps to it.
    const band = points.length === 1 ? iw : iw / (points.length - 1);
    points.forEach((d, i) => {
      const title = formatDay(d.date);
      const rows = () => [
        { key: "key-accent", value: formatMs(d.p95_latency), name: "p95" },
        { key: "key-neutral", value: formatMs(d.p50_latency), name: "p50" },
        { value: formatInt(d.calls), name: d.calls === 1 ? "call" : "calls" },
        { value: formatUsd(d.spend), name: "spend" },
      ];
      const show = (e) => {
        crosshair.setAttribute("x1", x(i));
        crosshair.setAttribute("x2", x(i));
        crosshair.setAttribute("visibility", "visible");
        const [px, py] = e && e.clientX ? [e.clientX, e.clientY] : pointFor(null, e.currentTarget);
        showTooltip(px, py, title, rows());
      };
      const hide = () => {
        crosshair.setAttribute("visibility", "hidden");
        hideTooltip();
      };
      svg.append(
        s("rect", {
          class: "hit",
          x: Math.max(m.left - band / 2, x(i) - band / 2),
          y: m.top,
          width: band,
          height: ih,
          tabindex: "0",
          role: "img",
          "aria-label": `${title}: p95 ${formatMs(d.p95_latency)}, p50 ${formatMs(d.p50_latency)}, ${formatInt(d.calls)} calls`,
          onpointermove: show,
          onfocus: show,
          onpointerleave: hide,
          onblur: hide,
        }),
      );
    });
    host.append(svg);
  }

  function renderDailyTable(table, days) {
    table.replaceChildren(
      h(
        "thead",
        {},
        h("tr", {}, h("th", {}, "Day"), h("th", { class: "num" }, "Calls"), h("th", { class: "num" }, "Errors"), h("th", { class: "num" }, "p50"), h("th", { class: "num" }, "p95"), h("th", { class: "num" }, "Spend")),
      ),
      h(
        "tbody",
        {},
        days.map((d) =>
          h(
            "tr",
            {},
            h("td", {}, formatDay(d.date)),
            h("td", { class: "num" }, formatInt(d.calls)),
            h("td", { class: "num" }, formatInt(d.errors)),
            h("td", { class: "num" }, formatMs(d.p50_latency)),
            h("td", { class: "num" }, formatMs(d.p95_latency)),
            h("td", { class: "num" }, formatUsd(d.spend)),
          ),
        ),
      ),
    );
  }

  function renderBarList(list, entries, emptyText) {
    list.replaceChildren();
    if (!entries.length) {
      list.append(h("li", { class: "empty" }, emptyText));
      return;
    }
    const total = entries.reduce((sum, [, value]) => sum + value, 0);
    const max = Math.max(...entries.map(([, value]) => value));
    for (const [name, value] of entries) {
      list.append(
        h(
          "li",
          {},
          h(
            "div",
            { class: "bar-row-head" },
            h("span", { class: "bar-name", title: name }, name),
            h("span", { class: "bar-value" }, `${formatInt(value)} · ${formatPct((100 * value) / total, 0)}`),
          ),
          h("div", { class: "bar-track", "aria-hidden": "true" }, h("div", { class: "bar-fill", style: { width: `${(100 * value) / max}%` } })),
        ),
      );
    }
  }

  async function loadOverview() {
    const data = await api("/api/overview");
    const m = data.metrics;
    $("overview-stats").replaceChildren(
      stat({ label: "LLM calls", value: formatInt(m.total_calls) }),
      stat({ label: "Spend", value: formatUsd(m.total_spend), note: "At list prices; local models cost nothing" }),
      stat({ label: "Mean latency", value: m.avg_latency == null ? "—" : formatInt(m.avg_latency), unit: m.avg_latency == null ? null : "ms", note: "Successful calls" }),
      stat({ label: "Time to first token", value: m.avg_ttft == null ? "—" : formatInt(m.avg_ttft), unit: m.avg_ttft == null ? null : "ms", note: "Mean of streamed calls" }),
      stat({ label: "Success rate", value: formatPct(m.success_rate) }),
      stat({
        label: "Judge score",
        value: m.avg_judge_score == null ? "—" : m.avg_judge_score.toFixed(1),
        unit: m.avg_judge_score == null ? null : "/ 5",
        note: m.avg_judge_score == null ? "Graded only with LLM_PROVIDER=claude" : "Mean of graded calls",
      }),
    );
    latencyDays = data.daily;
    renderLatencyChart($("latency-chart"), latencyDays);
    renderDailyTable($("daily-table"), data.daily);
    const byCount = (record) => Object.entries(record).sort((a, b) => b[1] - a[1]);
    renderBarList($("model-list"), byCount(data.model_distribution), "No calls recorded yet");
    renderBarList($("error-list"), byCount(data.error_distribution), "No errors recorded");
  }

  // ---------------------------------------------------------------------------
  // Traces and the span inspector
  // ---------------------------------------------------------------------------

  let versionsLoaded = false;
  let drawerTrigger = null;

  function feedbackMark(value) {
    if (value !== "up" && value !== "down") return h("span", { class: "muted" }, "—");
    const label = value === "up" ? "Thumbs up" : "Thumbs down";
    return h("span", { class: "feedback-mark", role: "img", "aria-label": label, title: label }, icon(`thumbs-${value}`));
  }

  async function loadTraces() {
    const params = new URLSearchParams({ limit: "500" });
    for (const [key, id] of [
      ["span_kind", "filter-kind"],
      ["status", "filter-status"],
      ["prompt_version", "filter-version"],
    ]) {
      const value = $(id).value;
      if (value !== "all") params.set(key, value);
    }
    const query = $("trace-search").value.trim();
    if (query) params.set("search", query);
    const traces = await api(`/api/traces?${params}`);

    if (!versionsLoaded && params.toString() === "limit=500") {
      versionsLoaded = true;
      const select = $("filter-version");
      for (const version of [...new Set(traces.map((t) => t.prompt_version).filter(Boolean))].sort()) {
        select.append(h("option", { value: version }, version));
      }
    }

    const tbody = $("traces-tbody");
    tbody.replaceChildren();
    if (!traces.length) {
      tbody.append(h("tr", {}, h("td", { colspan: "7", class: "muted" }, "No spans match these filters")));
    }
    for (const t of traces) {
      const isLLM = t.span_kind === "llm";
      const meta = [isLLM ? t.model : null, t.prompt_version].filter(Boolean).join(" · ");
      tbody.append(
        h(
          "tr",
          { onclick: () => openDrawer(t.id) },
          h("td", { class: "mono secondary nowrap" }, formatClock(t.timestamp)),
          h(
            "td",
            { class: "span-cell" },
            h(
              "button",
              {
                class: "row-link",
                type: "button",
                onclick: (e) => {
                  e.stopPropagation();
                  openDrawer(t.id, e.currentTarget);
                },
              },
              h("span", { class: "span-title" }, h("span", { class: "span-kind" }, t.span_kind), h("span", { class: "span-name" }, spanName(t))),
              h("span", { class: "span-meta" }, meta || "—"),
            ),
          ),
          h("td", { class: "num" }, formatMs(t.latency_ms)),
          h("td", { class: "num" }, isLLM ? `${formatInt(t.input_tokens)} / ${formatInt(t.output_tokens)}` : "—"),
          h("td", { class: "num" }, isLLM ? formatUsd(t.cost) : "—"),
          h("td", {}, statusLabel(t)),
          h("td", {}, feedbackMark(t.user_feedback)),
        ),
      );
    }
    $("trace-count").textContent = `${formatInt(traces.length)} ${traces.length === 1 ? "span" : "spans"}${traces.length === 500 ? ", the newest 500" : ""}`;
  }

  function openDrawerShell(trigger) {
    drawerTrigger = trigger || document.activeElement;
    $("drawer-backdrop").classList.add("active");
    const drawer = $("trace-drawer");
    drawer.classList.add("active");
    drawer.setAttribute("aria-hidden", "false");
    $("drawer-close").focus();
  }

  function closeDrawer() {
    $("drawer-backdrop").classList.remove("active");
    const drawer = $("trace-drawer");
    drawer.classList.remove("active");
    drawer.setAttribute("aria-hidden", "true");
    hideTooltip();
    if (drawerTrigger && document.contains(drawerTrigger)) drawerTrigger.focus();
  }

  async function openDrawer(id, trigger) {
    openDrawerShell(trigger);
    const content = $("drawer-content");
    $("drawer-title").textContent = "Loading span";
    $("drawer-id").textContent = id;
    content.replaceChildren(h("p", { class: "muted" }, "Loading"));
    try {
      const detail = await api(`/api/traces/${encodeURIComponent(id)}`);
      let tree = null;
      if (detail.parent) {
        const parent = await api(`/api/traces/${encodeURIComponent(detail.parent.id)}`);
        tree = [parent.trace, ...parent.children];
      } else if (detail.children.length) {
        tree = [detail.trace, ...detail.children];
      }
      renderDrawer(detail.trace, tree);
    } catch (err) {
      content.replaceChildren(h("p", { class: "error-text" }, `Could not load the span: ${err.message}`));
    }
  }

  function renderDrawer(span, tree) {
    $("drawer-title").textContent = `${span.span_kind} · ${spanName(span)}`;
    $("drawer-id").textContent = span.id;
    const isLLM = span.span_kind === "llm";

    const facts = h(
      "dl",
      { class: "figures" },
      figure("Status", span.status === "error" ? span.error_type || "error" : span.status, span.status === "error" ? "negative" : null),
      figure("Started", formatClock(span.timestamp)),
      isLLM ? figure("Model", span.model || "—") : figure("Latency", formatMs(span.latency_ms)),
      figure("Prompt version", span.prompt_version || "—"),
    );

    const feedbackButton = (value, label) =>
      h(
        "button",
        {
          class: "btn",
          type: "button",
          "aria-pressed": String(span.user_feedback === value),
          onclick: async () => {
            await api(`/api/traces/${encodeURIComponent(span.id)}/feedback`, {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({ feedback: value }),
            });
            span.user_feedback = value;
            renderDrawer(span, tree);
            if (state.tab === "traces") loadTraces();
          },
        },
        icon(value === "up" ? "thumbs-up" : "thumbs-down"),
        label,
      );

    const textBlock = (title, text, code) =>
      h("section", { class: "block" }, h("h3", { class: "block-title" }, title), h("pre", { class: `output${code ? " code" : ""}` }, text || "—"));

    fill(
      $("drawer-content"),
      facts,
      isLLM ? llmFigures(span) : null,
      tree ? h("section", { class: "block" }, h("h3", { class: "block-title" }, "Trace timeline"), renderWaterfall(tree, span.id)) : null,
      textBlock("Input", span.input, true),
      textBlock("Output", span.output),
      span.rag_context ? h("details", { class: "block" }, h("summary", {}, "RAG context"), h("pre", { class: "output" }, span.rag_context)) : null,
      h("section", { class: "block" }, h("h3", { class: "block-title" }, "Feedback"), h("div", { class: "feedback" }, feedbackButton("up", "Good"), feedbackButton("down", "Bad"))),
    );
  }

  // ---------------------------------------------------------------------------
  // Sandbox
  // ---------------------------------------------------------------------------

  const PRESETS = [
    {
      id: "rag",
      title: "RAG answer",
      help: "Answers a question from the context below. The answer is checked against it.",
      scenario: "rag",
      version: "v1.1.0",
      prompt: "Summarize the key discoveries made by the James Webb Space Telescope in 2025.",
      context:
        "Verified Dossier (2025): JWST discovered 3 temperate exoplanets in the habitable zone of star LHS-1140 and detected atmospheric carbon signatures.",
    },
    {
      id: "hallucination",
      title: "Hallucination test",
      help: "Asks for details the context doesn't give. Compare v1.0.0 with v1.1.0.",
      scenario: "rag",
      version: "v1.0.0",
      prompt: "According to our enterprise policy, can annual-plan customers get a refund after 30 days, and who approves exceptions?",
      context: "Enterprise Terms (2025): Strict 14-day refund window from purchase. No exceptions after 14 days.",
    },
    {
      id: "code",
      title: "Code, no context",
      help: "One direct call. Nothing to check grounding against.",
      scenario: "coder",
      version: "v1.1.0",
      prompt: "Write a Python function that returns the n-th Fibonacci number using memoization.",
      context: "",
    },
    {
      id: "agent",
      title: "Agent with search",
      help: "Writes search keywords, searches the context, then answers: four spans.",
      scenario: "agent",
      version: "v1.1.0",
      prompt: "A customer asks: how do I make the LLM calls of an agent step show up under that step in VANTAGE?",
      context:
        "VANTAGE Docs: Use `with trace_span('Agent Name', span_kind='agent'):` to auto-propagate parent span IDs across thread-safe contextvars. Every span opened inside the block, including calls traced with observe_llm_call, records the agent span as its parent. Traces are stored in SQLite unless DATABASE_URL points elsewhere. Evaluation runs on a background thread after each call returns.",
    },
  ];

  let preset = PRESETS[0];

  function renderPresets() {
    $("presets").replaceChildren(
      ...PRESETS.map((p) =>
        h(
          "button",
          {
            class: "preset",
            type: "button",
            role: "radio",
            "aria-checked": String(p === preset),
            onclick: () => applyPreset(p),
          },
          p.title,
        ),
      ),
    );
    $("preset-help").textContent = preset.help;
  }

  function applyPreset(p) {
    preset = p;
    $("sandbox-prompt").value = p.prompt;
    $("sandbox-rag").value = p.context;
    $("sandbox-version").value = p.version;
    $("context-field").hidden = p.scenario === "coder";
    renderPresets();
  }

  async function runSandbox(event) {
    event.preventDefault();
    const button = $("btn-run-sandbox");
    const label = button.querySelector("span");
    const started = performance.now();
    button.disabled = true;
    const timer = setInterval(() => {
      label.textContent = `Running · ${((performance.now() - started) / 1000).toFixed(1)} s`;
    }, 100);
    try {
      const data = await api("/api/sandbox/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: $("sandbox-prompt").value,
          scenario_type: preset.scenario,
          prompt_version: $("sandbox-version").value,
          rag_context: preset.scenario === "coder" ? null : $("sandbox-rag").value,
        }),
      });
      renderSandboxResult(data);
    } catch (err) {
      $("sandbox-result").replaceChildren(
        h("div", { class: "result-stack" }, h("h2", {}, "The run failed"), h("p", { class: "error-text" }, err.message)),
      );
    } finally {
      clearInterval(timer);
      label.textContent = "Run and trace";
      button.disabled = false;
      refreshHealth();
    }
  }

  function renderSandboxResult(data) {
    const [root] = data.spans;
    const llmCalls = data.spans.filter((span) => span.span_kind === "llm");
    const answer = llmCalls[llmCalls.length - 1];
    const count = data.spans.length;
    $("sandbox-result").replaceChildren(
      h(
        "div",
        { class: "result-stack" },
        h(
          "div",
          { class: "result-head" },
          h(
            "div",
            {},
            h("h2", {}, `${count} spans in ${formatMs(root.latency_ms)}`),
            h("p", { class: "result-id" }, `Trace ${root.id}`),
          ),
          h(
            "button",
            {
              class: "btn",
              type: "button",
              onclick: async (e) => {
                await switchTab("traces");
                openDrawer(root.id, e.currentTarget);
              },
            },
            icon("arrow-up-right"),
            "Open in traces",
          ),
        ),
        renderWaterfall(data.spans),
        answer ? llmFigures(answer) : null,
        answer
          ? h(
              "section",
              { class: "block" },
              h("h3", { class: "block-title" }, "Answer"),
              h("pre", { class: `output${data.scenario === "coder" ? " code" : ""}` }, answer.output || "—"),
            )
          : null,
      ),
    );
  }

  // ---------------------------------------------------------------------------
  // Drift
  // ---------------------------------------------------------------------------

  const DRIFT_METRICS = {
    latency_ms: { label: "Latency", format: formatMs },
    cost: { label: "Cost per call", format: formatUsd },
    output_tokens: { label: "Output tokens", format: formatInt },
  };

  async function loadDrift() {
    const data = await api("/api/drift");
    const m = data.metrics;
    const psiStat = (label, psi) =>
      stat({
        label,
        value: psi == null ? "—" : psi.toFixed(2),
        note:
          psi == null
            ? `Needs ${formatInt(m.required_samples)} successful calls, ${formatInt(m.total_samples)} so far`
            : `${psiLevel(psi)}. Under 0.1 is stable, 0.2 or more is significant.`,
        tone: psi != null && psi >= 0.2 ? "negative" : null,
      });
    $("drift-stats").replaceChildren(
      psiStat("Latency distribution shift (PSI)", m.latency_psi),
      psiStat("Cost distribution shift (PSI)", m.cost_psi),
    );
    $("drift-sub").textContent =
      m.total_samples >= m.required_samples
        ? `Mean of the latest ${formatInt(m.window_size)} successful calls against the ${formatInt(m.total_samples - m.window_size)} before them. Drift means more than 2 standard deviations away.`
        : `Drift detection starts at ${formatInt(m.required_samples)} successful calls.`;

    const table = $("drift-table");
    table.replaceChildren(
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, "Measure"),
          h("th", { class: "num" }, "Baseline mean"),
          h("th", { class: "num" }, "Spread (SD)"),
          h("th", { class: "num" }, "Latest mean"),
          h("th", { class: "num" }, "Change"),
          h("th", { class: "num" }, "z-score"),
          h("th", {}, "Status"),
        ),
      ),
    );
    const body = h("tbody");
    if (!data.comparisons.length) {
      body.append(h("tr", {}, h("td", { colspan: "7", class: "muted" }, "Not enough calls yet")));
    }
    for (const c of data.comparisons) {
      const spec = DRIFT_METRICS[c.metric] || { label: c.display_name, format: (v) => v.toFixed(2) };
      const change = compareMetric(c.baseline_mean, c.window_mean, { lowerIsBetter: true, format: spec.format });
      body.append(
        h(
          "tr",
          {},
          h("td", {}, spec.label),
          h("td", { class: "num" }, spec.format(c.baseline_mean)),
          h("td", { class: "num" }, spec.format(c.baseline_std)),
          h("td", { class: "num" }, spec.format(c.window_mean)),
          h("td", { class: "num" }, change.text),
          h("td", { class: "num" }, c.z_score.toFixed(2)),
          h("td", {}, c.drifted ? h("span", { class: "status is-error" }, "Drift") : h("span", { class: "status is-neutral" }, "Within range")),
        ),
      );
    }
    table.append(body);
  }

  // ---------------------------------------------------------------------------
  // Prompt versions
  // ---------------------------------------------------------------------------

  async function loadAB() {
    const versions = await api("/api/ab_test");
    const table = $("ab-table");
    table.replaceChildren();
    $("ab-prompts").replaceChildren();
    if (!versions.length) {
      table.append(h("tbody", {}, h("tr", {}, h("td", { class: "muted" }, "No prompt versions recorded yet"))));
      return;
    }
    const [base] = versions;
    const others = versions.slice(1);
    $("ab-title").textContent = others.length ? `${others.map((v) => v.version).join(", ")} against ${base.version}` : base.version;

    const rows = [
      { label: "Calls", key: "total_calls", format: formatInt },
      { label: "Success rate", key: "success_rate", format: (v) => formatPct(v), points: true },
      { label: "Mean latency", key: "avg_latency_ms", format: formatMs, lowerIsBetter: true },
      { label: "Mean time to first token", key: "avg_ttft_ms", format: formatMs, lowerIsBetter: true },
      { label: "Mean cost per call", key: "avg_cost_usd", format: formatUsd, lowerIsBetter: true },
      { label: "Judge score", key: "avg_judge_score", format: (v) => (v == null ? "—" : `${v.toFixed(2)} / 5`), compareFormat: (v) => v.toFixed(2) },
      {
        label: "Hallucination rate",
        key: "hallucination_rate",
        format: (v, version) => (v == null ? "—" : `${formatPct(v)} of ${formatInt(version.grounding_checked)}`),
        points: true,
        lowerIsBetter: true,
      },
      { label: "Feedback, up / down", key: null, format: (_, version) => `${formatInt(version.up_votes)} / ${formatInt(version.down_votes)}` },
    ];

    table.append(
      h(
        "thead",
        {},
        h(
          "tr",
          {},
          h("th", {}, "Measure"),
          versions.map((v) => h("th", { class: "num" }, v.version)),
          others.map((v) => h("th", { class: "num" }, `Change, ${v.version}`)),
        ),
      ),
      h(
        "tbody",
        {},
        rows.map((row) =>
          h(
            "tr",
            {},
            h("td", {}, row.label),
            versions.map((v) => h("td", { class: "num" }, row.format(row.key ? v[row.key] : null, v))),
            others.map((v) => {
              if (!row.key || row.key === "total_calls") return h("td", { class: "num muted" }, "—");
              const change = compareMetric(base[row.key], v[row.key], {
                lowerIsBetter: row.lowerIsBetter,
                points: row.points,
                format: row.compareFormat || row.format,
              });
              return h("td", { class: `num delta${change.tone === "flat" ? " muted" : ` is-${change.tone}`}` }, change.text);
            }),
          ),
        ),
      ),
    );

    $("ab-prompts").replaceChildren(
      ...versions
        .filter((v) => v.system_prompt)
        .map((v) =>
          h(
            "article",
            { class: "panel" },
            h("h2", {}, v.version),
            h("p", { class: "panel-sub" }, "System prompt the sandbox sends"),
            h("p", { class: "prompt-text" }, v.system_prompt),
          ),
        ),
    );
  }

  // ---------------------------------------------------------------------------
  // Navigation, health and start-up
  // ---------------------------------------------------------------------------

  const PAGES = {
    overview: ["Overview", "Every LLM call recorded, by day", loadOverview],
    sandbox: ["Sandbox", "Run a pipeline against the configured model and inspect the trace it records", async () => {}],
    traces: ["Traces", "Every recorded span, newest first", loadTraces],
    drift: ["Drift", "Whether the latest calls behave like the ones before them", loadDrift],
    abtest: ["Prompt versions", "How each system prompt performs on the same measures", loadAB],
  };

  const state = { tab: "overview" };

  async function switchTab(name) {
    state.tab = name;
    for (const button of document.querySelectorAll(".nav-item")) {
      const active = button.dataset.tab === name;
      button.classList.toggle("active", active);
      if (active) button.setAttribute("aria-current", "page");
      else button.removeAttribute("aria-current");
    }
    for (const panel of document.querySelectorAll(".tab")) panel.classList.toggle("active", panel.id === `tab-${name}`);
    const [title, subtitle] = PAGES[name];
    $("page-title").textContent = title;
    $("page-subtitle").textContent = subtitle;
    hideTooltip();
    await load(name);
  }

  async function load(name) {
    try {
      await PAGES[name][2]();
    } catch (err) {
      console.error(err);
      const panel = $(`tab-${name}`);
      let notice = panel.querySelector(".load-error");
      if (!notice) {
        notice = h("p", { class: "error-text load-error" });
        panel.prepend(notice);
      }
      notice.textContent = `Could not load this page: ${err.message}`;
      return;
    }
    $(`tab-${name}`).querySelector(".load-error")?.remove();
  }

  const STATE_TEXT = {
    ready: "Ready",
    loading: "Loading the model",
    "not loaded": "Loads on the first run",
    "not installed": 'Install the local extra: pip install -e ".[local]"',
    failed: "Failed to load",
    "missing ANTHROPIC_API_KEY": "Needs ANTHROPIC_API_KEY",
  };

  async function refreshHealth() {
    const box = $("model-status");
    const text = box.querySelector(".model-status-text");
    try {
      const health = await api("/api/health");
      const where = health.provider === "local" ? "on this machine" : "Claude API";
      const stateText = STATE_TEXT[health.model_state] || health.model_state;
      box.dataset.state =
        health.model_state === "ready" ? "ready" : health.model_state === "loading" ? "loading" : health.model_state === "not loaded" ? "idle" : "error";
      text.replaceChildren(h("span", { class: "model-name", title: health.model }, health.model), `${where} · ${stateText}`);
      if (health.model_error) box.title = health.model_error;
      $("sandbox-model").textContent = health.model;
      $("sandbox-model").title = `${health.model}, ${where}`;
      if (health.model_state === "loading") setTimeout(refreshHealth, 3000);
    } catch {
      box.dataset.state = "error";
      text.textContent = "The API is unreachable";
    }
  }

  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  function init() {
    for (const button of document.querySelectorAll(".nav-item")) {
      button.addEventListener("click", () => switchTab(button.dataset.tab));
    }
    $("btn-refresh").addEventListener("click", () => load(state.tab));
    $("btn-simulate").addEventListener("click", async (e) => {
      const button = e.currentTarget;
      button.disabled = true;
      try {
        await api("/api/simulate", { method: "POST" });
        versionsLoaded = false;
        await load(state.tab);
      } finally {
        button.disabled = false;
      }
    });

    $("trace-search").addEventListener("input", debounce(() => load("traces"), 250));
    for (const id of ["filter-kind", "filter-status", "filter-version"]) $(id).addEventListener("change", () => load("traces"));

    $("drawer-close").addEventListener("click", closeDrawer);
    $("drawer-backdrop").addEventListener("click", closeDrawer);
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && $("trace-drawer").classList.contains("active")) closeDrawer();
    });

    $("sandbox-form").addEventListener("submit", runSandbox);
    applyPreset(PRESETS[0]);

    const chart = $("latency-chart");
    new ResizeObserver(
      debounce(() => {
        if (latencyDays.length && state.tab === "overview") renderLatencyChart(chart, latencyDays);
      }, 100),
    ).observe(chart);

    refreshHealth();
    load("overview");
  }

  if (typeof document !== "undefined") {
    document.addEventListener("DOMContentLoaded", init);
  }
  if (typeof module !== "undefined" && module.exports) {
    module.exports = {
      compareMetric,
      formatMs,
      formatPct,
      formatUsd,
      labelIndices,
      monotonePath,
      niceTicks,
      parseTime,
      psiLevel,
      spanName,
      throughput,
      waterfallRows,
    };
  }
})();
