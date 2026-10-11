"""VANTAGE REST API and web console."""

import logging
import os
import re
import threading
from concurrent.futures import TimeoutError as EvaluationTimeout
from contextlib import asynccontextmanager
from typing import Literal

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from vantage import __version__
from vantage.core.db import (
    get_children,
    get_trace_by_id,
    has_traces,
    init_db,
    query_traces,
    update_trace_feedback,
)
from vantage.core.instrumentation import observe_llm_call, trace_span, wait_for_evaluations
from vantage.core.settings import settings
from vantage.eval.drift import calculate_psi, check_for_drift, compare_windows
from vantage.local.runtime import INSTALL_HINT, LocalModelError, local_models_installed
from vantage.utils.simulate_traffic import generate_simulated_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vantage.api")


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    if not has_traces():
        logger.info("Empty trace store: adding synthetic demo traffic")
        generate_simulated_data()
    if settings.llm_provider == "local" and local_models_installed():
        # Load the models in the background so the console is usable while they load.
        threading.Thread(target=_warm_up, name="vantage-warm-up", daemon=True).start()
    yield


app = FastAPI(
    title="VANTAGE Telemetry Console API",
    description="REST API backend for VANTAGE LLM Telemetry & Observability Platform",
    version=__version__,
    lifespan=lifespan,
)


# --------------------------------------------------------------------------
# The traced model
# --------------------------------------------------------------------------

_client = None
_client_lock = threading.Lock()


def current_model() -> str:
    return settings.local_llm_model if settings.llm_provider == "local" else settings.chat_model


def get_llm_client():
    """The client the sandbox calls: the local model, or Claude. Created once per process."""
    global _client
    with _client_lock:
        if _client is None:
            if settings.llm_provider == "local":
                if not local_models_installed():
                    raise HTTPException(status_code=503, detail=INSTALL_HINT)
                from vantage.local.llm import LocalChatClient

                _client = LocalChatClient(
                    settings.local_llm_model,
                    device=settings.local_device,
                    max_new_tokens=settings.local_max_new_tokens,
                )
            else:
                if not settings.anthropic_api_key:
                    raise HTTPException(
                        status_code=503, detail="LLM_PROVIDER=claude needs ANTHROPIC_API_KEY."
                    )
                try:
                    import anthropic
                except ImportError as exc:
                    detail = 'Claude needs the optional dependency: pip install -e ".[claude]"'
                    raise HTTPException(status_code=503, detail=detail) from exc
                # The beta namespace carries server-side refusal fallbacks (see _request_options).
                _client = anthropic.Anthropic(api_key=settings.anthropic_api_key).beta
        return _client


def _warm_up() -> None:
    try:
        get_llm_client().warm_up()
        from vantage.local.nli import load_grounding_checker

        load_grounding_checker(settings.local_nli_model, settings.local_device)
    except Exception:
        logger.warning("Could not load the local models", exc_info=True)


def _request_options(*, local_max_tokens: int, claude_max_tokens: int) -> dict:
    if settings.llm_provider == "claude":
        # A request the model declines on policy grounds is re-run on a fallback model, server-side.
        return {
            "max_tokens": claude_max_tokens,
            "betas": ["server-side-fallback-2026-07-01"],
            "fallbacks": "default",
        }
    return {"max_tokens": local_max_tokens}


@app.get("/api/health")
def health():
    error = None
    if settings.llm_provider == "local":
        if not local_models_installed():
            state = "not installed"
        elif _client is None:
            state = "not loaded"
        else:
            state, error = _client.state, _client.error
    else:
        state = "ready" if settings.anthropic_api_key else "missing ANTHROPIC_API_KEY"
    return {
        "status": "ok",
        "version": __version__,
        "provider": settings.llm_provider,
        "model": current_model(),
        "model_state": state,
        "model_error": error,
    }


# --------------------------------------------------------------------------
# Dashboard metrics. Computed over LLM spans: agent, chain and tool spans are pipeline steps.
# --------------------------------------------------------------------------


def _llm_frame() -> pd.DataFrame:
    return pd.DataFrame(query_traces(span_kind="llm"))


def _mean(series: pd.Series, digits: int) -> float | None:
    series = series.dropna()
    return round(float(series.mean()), digits) if not series.empty else None


def _success_rate(frame: pd.DataFrame) -> float:
    return round(100 * float((frame["status"] == "success").mean()), 1)


@app.get("/api/overview")
def get_overview_data():
    df = _llm_frame()
    if df.empty:
        return {
            "metrics": {
                "total_calls": 0,
                "total_spend": 0.0,
                "avg_latency": None,
                "avg_ttft": None,
                "success_rate": None,
                "avg_judge_score": None,
            },
            "daily": [],
            "model_distribution": {},
            "error_distribution": {},
        }

    df["date"] = pd.to_datetime(df["timestamp"], format="ISO8601", utc=True).dt.strftime("%Y-%m-%d")
    # Latency is measured over successful calls; failures show in the success rate and error counts.
    successful = df[df["status"] == "success"]
    latency = successful.groupby("date")["latency_ms"]
    daily = (
        df.groupby("date")
        .agg(
            calls=("id", "count"),
            errors=("status", lambda s: int((s == "error").sum())),
            spend=("cost", "sum"),
        )
        .join(latency.quantile(0.5).rename("p50_latency"))
        .join(latency.quantile(0.95).rename("p95_latency"))
        .reset_index()
    )
    daily = daily.astype(object).where(daily.notna(), None)  # JSON has no NaN

    return {
        "metrics": {
            "total_calls": len(df),
            "total_spend": round(float(df["cost"].sum()), 4),
            "avg_latency": _mean(successful["latency_ms"], 1),
            "avg_ttft": _mean(successful["ttft_ms"], 1),
            "success_rate": _success_rate(df),
            "avg_judge_score": _mean(df["judge_score"], 2),
        },
        "daily": daily.to_dict(orient="records"),
        "model_distribution": df["model"].fillna("unknown").value_counts().to_dict(),
        "error_distribution": df[df["status"] == "error"]["error_type"]
        .fillna("unspecified")
        .value_counts()
        .to_dict(),
    }


@app.get("/api/traces")
def list_traces(
    status: str | None = Query(None),
    prompt_version: str | None = Query(None),
    model: str | None = Query(None),
    span_kind: str | None = Query(None),
    search: str | None = Query(None),
    limit: int = Query(500, ge=1, le=5000),
):
    def chosen(value: str | None) -> str | None:
        return value if value and value != "all" else None

    return query_traces(
        status=chosen(status),
        prompt_version=chosen(prompt_version),
        model=chosen(model),
        span_kind=chosen(span_kind),
        search=search or None,
        limit=limit,
    )


@app.get("/api/traces/{trace_id}")
def get_trace_detail(trace_id: str):
    trace = get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    parent = get_trace_by_id(trace["parent_id"]) if trace.get("parent_id") else None
    return {"trace": trace, "parent": parent, "children": get_children(trace_id)}


class FeedbackRequest(BaseModel):
    feedback: Literal["up", "down"]


@app.post("/api/traces/{trace_id}/feedback")
def set_feedback(trace_id: str, body: FeedbackRequest):
    if not update_trace_feedback(trace_id, body.feedback):
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"status": "success", "trace_id": trace_id, "feedback": body.feedback}


@app.get("/api/drift")
def get_drift_analysis(window_size: int = Query(30, ge=5, le=1000)):
    traces = query_traces(span_kind="llm")
    alerts = check_for_drift(traces, window_size=window_size)

    # Distribution shift (PSI) between the baseline and the latest window
    successful = sorted((t for t in traces if t["status"] == "success"), key=lambda t: t["timestamp"])
    latency_psi = cost_psi = None
    if len(successful) >= window_size * 2:
        baseline, target = successful[:-window_size], successful[-window_size:]
        latency_psi = round(
            calculate_psi([t["latency_ms"] for t in baseline], [t["latency_ms"] for t in target]), 4
        )
        priced = [[t["cost"] for t in spans if t["cost"] is not None] for spans in (baseline, target)]
        cost_psi = round(calculate_psi(*priced), 4)

    return {
        "alerts": alerts,
        "comparisons": compare_windows(traces, window_size=window_size),
        "metrics": {
            "latency_psi": latency_psi,
            "cost_psi": cost_psi,
            "total_samples": len(successful),
            "required_samples": window_size * 2,
            "window_size": window_size,
        },
    }


@app.get("/api/ab_test")
def get_ab_test_metrics():
    df = _llm_frame()
    if df.empty:
        return []

    versions = []
    for version, group in df.groupby("prompt_version"):
        evaluated = group["hallucination_flag"].dropna()
        versions.append(
            {
                "version": version,
                "total_calls": len(group),
                "success_rate": _success_rate(group),
                "avg_latency_ms": _mean(group["latency_ms"], 1),
                "avg_ttft_ms": _mean(group["ttft_ms"], 1),
                "avg_cost_usd": _mean(group["cost"], 6),
                "avg_judge_score": _mean(group["judge_score"], 2),
                # Share of the responses checked for grounding (calls with RAG context)
                "hallucination_rate": round(100 * float(evaluated.astype(bool).mean()), 1)
                if len(evaluated)
                else None,
                "grounding_checked": len(evaluated),
                "up_votes": int((group["user_feedback"] == "up").sum()),
                "down_votes": int((group["user_feedback"] == "down").sum()),
                "system_prompt": SYSTEM_PROMPTS.get(version),  # what the sandbox sends for this version
            }
        )
    return versions


@app.post("/api/simulate")
def trigger_simulation():
    generate_simulated_data()
    return {"status": "success", "message": "Simulated traces generated successfully"}


# --------------------------------------------------------------------------
# Sandbox: run a traced pipeline against the configured model
# --------------------------------------------------------------------------

# The prompt versions the A/B view compares.
SYSTEM_PROMPTS = {
    "v1.0.0": "You are a helpful assistant.",
    "v1.1.0": (
        "You are a careful assistant. Keep answers short. When context is given, answer only from it, "
        "and if it doesn't contain the answer, say that you don't know."
    ),
}
KEYWORDS_SYSTEM = (
    "Rewrite the user's question as a few search keywords. Reply with the keywords only, separated by spaces."
)
_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_WORD = re.compile(r"[^\W_]+")


class SandboxRunRequest(BaseModel):
    prompt: str = Field(max_length=4000)
    scenario_type: Literal["rag", "agent", "coder"] = "rag"
    prompt_version: Literal["v1.0.0", "v1.1.0"] = "v1.1.0"
    rag_context: str | None = Field(default=None, max_length=20000)


def _ask(client, *, version: str, question: str, context: str | None, system: str, **options) -> str:
    """One traced, streamed LLM call; returns the response text."""
    user = f"Context:\n{context}\n\nQuestion: {question}" if context else question
    events = observe_llm_call(
        client,
        current_model(),
        [{"role": "user", "content": user}],
        prompt_version=version,
        system=system,
        rag_context=context,
        stream=True,
        **options,
    )
    deltas = (event.delta for event in events if event.type == "content_block_delta")
    return "".join(delta.text for delta in deltas if getattr(delta, "type", None) == "text_delta")


def select_sentences(context: str, query: str, top: int = 3) -> tuple[list[str], int]:
    """The context sentences that share the most words with the query, in their original order."""
    sentences = [s.strip() for s in _SENTENCE.split(context) if s.strip()]
    query_words = {w.lower() for w in _WORD.findall(query) if len(w) > 2}
    scored = [(len(query_words & {w.lower() for w in _WORD.findall(s)}), i) for i, s in enumerate(sentences)]
    best = sorted(i for score, i in sorted(scored, reverse=True)[:top] if score > 0)
    return ([sentences[i] for i in best] or sentences), len(sentences)


def _run_pipeline(client, request: SandboxRunRequest, question: str, context: str | None) -> str:
    version = request.prompt_version
    answer_options = _request_options(local_max_tokens=settings.local_max_new_tokens, claude_max_tokens=16000)

    if request.scenario_type == "agent" and context:
        with trace_span("Agent run", span_kind="agent", prompt_version=version) as root:
            keywords = _ask(
                client,
                version=version,
                question=question,
                context=None,
                system=KEYWORDS_SYSTEM,
                evaluate=False,
                **_request_options(local_max_tokens=32, claude_max_tokens=2048),
            )
            with trace_span("Search context", span_kind="tool", prompt_version=version) as step:
                selected, total = select_sentences(context, f"{keywords} {question}")
                step.output = f"Selected {len(selected)} of {total} sentences for the keywords: {keywords.strip()[:120]}"
            _ask(
                client,
                version=version,
                question=question,
                context=" ".join(selected),
                system=SYSTEM_PROMPTS[version],
                **answer_options,
            )
    else:
        name = "RAG run" if context else "Direct run"
        with trace_span(name, span_kind="chain", prompt_version=version) as root:
            _ask(
                client,
                version=version,
                question=question,
                context=context,
                system=SYSTEM_PROMPTS[version],
                **answer_options,
            )
    return root.trace_id


@app.post("/api/sandbox/run")
def run_sandbox_trace(request: SandboxRunRequest):
    question = request.prompt.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")
    context = (request.rag_context or "").strip() or None
    if request.scenario_type == "coder":
        context = None

    client = get_llm_client()
    try:
        root_id = _run_pipeline(client, request, question, context)
    except LocalModelError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning("Sandbox run failed", exc_info=True)
        raise HTTPException(status_code=502, detail=f"The model call failed: {exc}") from exc

    try:
        wait_for_evaluations(timeout=120)  # the grounding check, so the result can show it
    except EvaluationTimeout:
        logger.warning("Evaluation of run %s is still running", root_id)

    return {
        "status": "success",
        "scenario": request.scenario_type,
        "provider": settings.llm_provider,
        "model": current_model(),
        "prompt_version": request.prompt_version,
        "root_trace_id": root_id,
        "spans": [get_trace_by_id(root_id), *get_children(root_id)],
    }


# --------------------------------------------------------------------------
# Serve Static Frontend Files (`src/vantage/web`)
# --------------------------------------------------------------------------

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(os.path.join(WEB_DIR, "index.html"))
