"""Measure VANTAGE's tracing overhead, the PSI cost and, with the local extra, the local workload.

    python scripts/benchmark.py

Traces go to a throwaway SQLite database; nothing touches data/. The overhead numbers use a fake
client that answers instantly, so they are the cost of tracing alone.
"""

import os
import random
import statistics
import tempfile
import time

os.environ["DATABASE_URL"] = f"sqlite:///{tempfile.mkdtemp(prefix='vantage-bench-')}/bench.db"
os.environ["LLM_PROVIDER"] = "local"
os.environ["ANTHROPIC_API_KEY"] = ""

from types import SimpleNamespace as NS  # noqa: E402

from vantage.core import db  # noqa: E402
from vantage.core import instrumentation as ins  # noqa: E402
from vantage.core.settings import settings  # noqa: E402
from vantage.eval.drift import calculate_psi  # noqa: E402
from vantage.local.runtime import local_models_installed  # noqa: E402


def measure(fn, n):
    samples = []
    for _ in range(n):
        start = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - start) * 1000)
    samples.sort()
    return statistics.median(samples), samples[int(0.95 * (n - 1))]


def tracing_overhead():
    response = NS(
        content=[NS(type="text", text="Paris is the capital of France.")],
        usage=NS(input_tokens=12, output_tokens=8),
        stop_reason="end_turn",
    )
    client = NS(messages=NS(create=lambda **kwargs: response))
    messages = [{"role": "user", "content": "What is the capital of France?"}]

    def call():
        ins.observe_llm_call(client, "claude-haiku-5-5", messages, "v1.0.0")

    def span():
        with ins.trace_span("step", span_kind="tool"):
            pass

    for _ in range(100):
        call()
    call_ms, span_ms = measure(call, 3000), measure(span, 3000)
    ins.wait_for_evaluations(timeout=60)
    print(f"observe_llm_call: median {call_ms[0]:.2f} ms, p95 {call_ms[1]:.2f} ms per call")
    print(f"trace_span:       median {span_ms[0]:.2f} ms, p95 {span_ms[1]:.2f} ms per span")


def psi_cost():
    random.seed(7)
    baseline = [random.gauss(1000, 150) for _ in range(500)]
    window = [random.gauss(1200, 150) for _ in range(500)]
    print(
        f"calculate_psi:    median {measure(lambda: calculate_psi(baseline, window), 200)[0]:.2f} ms for 500 + 500 samples"
    )


def local_workload(runs=6):
    from vantage.local.llm import LocalChatClient
    from vantage.local.nli import load_grounding_checker

    start = time.perf_counter()
    client = LocalChatClient(settings.local_llm_model, device=settings.local_device)
    client.warm_up()
    print(
        f"\nload {settings.local_llm_model}: {time.perf_counter() - start:.1f} s on {client._backend.device}"
    )
    start = time.perf_counter()
    checker = load_grounding_checker(settings.local_nli_model, settings.local_device)
    print(f"load {settings.local_nli_model}: {time.perf_counter() - start:.1f} s")

    context = (
        "Verified Dossier (2025): JWST discovered 3 temperate exoplanets in the habitable zone of star "
        "LHS-1140 and detected atmospheric carbon signatures."
    )
    question = "Summarize the key discoveries made by the James Webb Space Telescope in 2025."
    results = []
    for run in range(runs):
        events = ins.observe_llm_call(
            client,
            settings.local_llm_model,
            [{"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
            "v1.1.0",
            system="You are a careful assistant. Keep answers short. When context is given, answer only from it.",
            rag_context=context,
            stream=True,
            evaluate=False,
            max_tokens=settings.local_max_new_tokens,
        )
        answer = "".join(e.delta.text for e in events if e.type == "content_block_delta")
        span = db.query_traces(span_kind="llm", model=settings.local_llm_model, limit=1)[0]
        start = time.perf_counter()
        checker.check(answer, context)
        check_ms = (time.perf_counter() - start) * 1000
        rate = span["output_tokens"] / ((span["latency_ms"] - span["ttft_ms"]) / 1000)
        results.append((span["ttft_ms"], rate, check_ms))
        print(
            f"run {run}: time to first token {span['ttft_ms']:.0f} ms, {rate:.0f} tokens/s, grounding check {check_ms:.0f} ms"
        )

    warm = results[1:]  # the first call after loading pays a one-off GPU set-up
    print(
        f"after the first call: time to first token {statistics.median(r[0] for r in warm):.0f} ms, "
        f"{statistics.median(r[1] for r in warm):.0f} tokens/s, grounding check {statistics.median(r[2] for r in warm):.0f} ms (medians)"
    )


if __name__ == "__main__":
    db.init_db()
    tracing_overhead()
    psi_cost()
    if local_models_installed():
        local_workload()
    else:
        print('\nSkipping the local workload: pip install -e ".[local]" to measure it.')
