// Unit tests for the web console's pure helpers. Run with: node --test tests/ui
import assert from "node:assert/strict";
import { createRequire } from "node:module";
import { test } from "node:test";

const require = createRequire(import.meta.url);
const {
  compareMetric,
  formatMs,
  formatUsd,
  labelIndices,
  monotonePath,
  niceTicks,
  parseTime,
  psiLevel,
  spanName,
  throughput,
  waterfallRows,
} = require("../../src/vantage/web/js/app.js");

test("money keeps two significant digits under a dollar and cents above", () => {
  assert.equal(formatUsd(0.1603), "$0.16");
  assert.equal(formatUsd(0.00048), "$0.00048");
  assert.equal(formatUsd(0.0000196), "$0.000020");
  assert.equal(formatUsd(1234.5), "$1,234.50");
  assert.equal(formatUsd(0), "$0");
  assert.equal(formatUsd(null), "—");
});

test("durations are whole milliseconds with separators", () => {
  assert.equal(formatMs(1180.06), "1,180 ms");
  assert.equal(formatMs(0.0035), "<1 ms");
  assert.equal(formatMs(null), "—");
});

test("axis ticks are clean and reach the maximum", () => {
  assert.deepEqual(niceTicks(2431, 4), [0, 1000, 2000, 3000]);
  assert.deepEqual(niceTicks(0.9, 4), [0, 0.25, 0.5, 0.75, 1]);
  assert.deepEqual(niceTicks(0), [0, 1]);
});

test("the last day is always labeled, without crowding the label before it", () => {
  assert.deepEqual(labelIndices(8, 1), [0, 1, 2, 3, 4, 5, 6, 7]);
  assert.deepEqual(labelIndices(8, 3), [0, 3, 7]); // not 0, 3, 6, 7
  assert.deepEqual(labelIndices(7, 3), [0, 3, 6]);
  assert.deepEqual(labelIndices(1, 3), [0]);
  assert.deepEqual(labelIndices(0, 3), []);
});

test("microsecond timestamps parse in every browser", () => {
  assert.equal(parseTime("2026-11-17T14:40:21.466671+00:00"), Date.parse("2026-11-17T14:40:21.466Z"));
});

test("the waterfall places spans on the trace's timeline and splits LLM bars at the first token", () => {
  const spans = [
    { id: "root", span_kind: "agent", timestamp: "2026-11-17T10:00:00.000000+00:00", latency_ms: 1000 },
    { id: "answer", span_kind: "llm", timestamp: "2026-11-17T10:00:00.400000+00:00", latency_ms: 500, ttft_ms: 100 },
    { id: "search", span_kind: "tool", timestamp: "2026-11-17T10:00:00.100000+00:00", latency_ms: 200 },
  ];
  const { total, rows } = waterfallRows(spans);
  assert.equal(total, 1000);
  assert.deepEqual(
    rows.map((r) => r.span.id),
    ["root", "search", "answer"],
  );
  const answer = rows[2];
  assert.deepEqual([answer.left, answer.width, answer.split, answer.depth], [40, 50, 20, 1]);
  assert.equal(rows[1].split, null);
});

test("a child that outlives its root stretches the timeline instead of overflowing it", () => {
  const spans = [
    { id: "root", span_kind: "chain", timestamp: "2026-11-17T10:00:00Z", latency_ms: 100 },
    { id: "late", span_kind: "llm", timestamp: "2026-11-17T10:00:00.050Z", latency_ms: 150 },
  ];
  const { total, rows } = waterfallRows(spans);
  assert.equal(total, 200);
  assert.ok(rows[1].left + rows[1].width <= 100);
});

test("changes say which way is better", () => {
  assert.deepEqual(compareMetric(1000, 1355, { lowerIsBetter: true, format: formatMs }), {
    text: "+355 ms (+36%)",
    tone: "worse",
  });
  assert.deepEqual(compareMetric(95.1, 96, { points: true }), { text: "+0.9 pts", tone: "better" });
  assert.equal(compareMetric(3, 3).tone, "flat");
  assert.equal(compareMetric(null, 3).text, "—");
});

test("PSI bands", () => {
  assert.equal(psiLevel(0.05), "Stable");
  assert.equal(psiLevel(0.15), "Moderate shift");
  assert.equal(psiLevel(1.33), "Significant shift");
  assert.equal(psiLevel(null), null);
});

test("spans are named by their step or by the question asked", () => {
  assert.equal(spanName({ input: "Span step: Search context" }), "Search context");
  assert.equal(
    spanName({ input: "[System: Be careful.]\nUser: Context:\nRefunds take 14 days.\n\nQuestion: How long?" }),
    "How long?",
  );
  assert.equal(spanName({ input: "User: hello\nmore" }), "hello");
});

test("throughput counts generation time only", () => {
  assert.equal(throughput({ output_tokens: 50, latency_ms: 1100, ttft_ms: 100 }), 50);
  assert.equal(throughput({ output_tokens: 0, latency_ms: 1000 }), null);
});

test("the smooth line passes through every point", () => {
  const path = monotonePath([
    [0, 10],
    [10, 0],
    [20, 5],
  ]);
  assert.ok(path.startsWith("M0,10"));
  assert.ok(path.includes(",10,0C"));
  assert.ok(path.endsWith(",20,5"));
});
