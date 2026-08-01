import os
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, HTTPException, Query
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
import pandas as pd
import numpy as np

from vantage.core.db import (
    init_db,
    get_all_traces,
    get_trace_by_id,
    update_trace_feedback,
    get_db,
    Trace
)
from vantage.core.instrumentation import trace_span, observe_llm_call
from vantage.eval.drift import check_for_drift, calculate_psi
from vantage.utils.simulate_traffic import generate_simulated_data

from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    traces = get_all_traces()
    if not traces:
        print("[VANTAGE API] No traces found in DB. Generating initial simulated telemetry data...")
        generate_simulated_data()
    yield

app = FastAPI(
    title="VANTAGE Telemetry Console API",
    description="REST API backend for VANTAGE LLM Telemetry & Observability Platform",
    version="0.1.0",
    lifespan=lifespan
)


class FeedbackRequest(BaseModel):
    feedback: str # "up" or "down"

class SandboxRunRequest(BaseModel):
    prompt: str
    scenario_type: Optional[str] = "rag"  # "rag", "agent", "coder", "direct"
    model: Optional[str] = "claude-3-5-sonnet-20241022"
    prompt_version: Optional[str] = "v1.1.0"
    mode: Optional[str] = "simulated"  # "simulated" or "live"
    rag_context: Optional[str] = None
    api_key: Optional[str] = None

# --------------------------------------------------------------------------
# API Endpoints
# --------------------------------------------------------------------------

@app.get("/api/overview")
def get_overview_data():
    traces = get_all_traces()
    if not traces:
        return {
            "metrics": {
                "total_traces": 0,
                "total_spend": 0.0,
                "avg_latency": 0.0,
                "avg_ttft": 0.0,
                "success_rate": 100.0,
                "avg_judge_score": 0.0,
            },
            "spend_series": [],
            "model_distribution": {},
            "error_distribution": {}
        }
    
    df = pd.DataFrame(traces)
    
    # Calculate main metrics
    total_traces = len(df)
    total_spend = float(df["cost"].sum()) if "cost" in df else 0.0
    avg_latency = float(df["latency_ms"].mean()) if "latency_ms" in df and not df["latency_ms"].dropna().empty else 0.0
    
    ttft_df = df["ttft_ms"].dropna() if "ttft_ms" in df else pd.Series()
    avg_ttft = float(ttft_df.mean()) if not ttft_df.empty else 0.0
    
    success_count = len(df[df["status"] == "success"]) if "status" in df else total_traces
    success_rate = round((success_count / total_traces) * 100, 1) if total_traces > 0 else 100.0
    
    judge_df = df["judge_score"].dropna() if "judge_score" in df else pd.Series()
    avg_judge_score = round(float(judge_df.mean()), 2) if not judge_df.empty else 0.0

    # Daily spend & trace volume time series
    df["date"] = pd.to_datetime(df["timestamp"]).dt.strftime("%Y-%m-%d")
    daily_stats = df.groupby("date").agg(
        daily_spend=("cost", "sum"),
        trace_count=("id", "count"),
        avg_latency=("latency_ms", "mean")
    ).reset_index()
    
    spend_series = daily_stats.to_dict(orient="records")

    # Model usage distribution
    model_counts = df["model"].fillna("unknown").value_counts().to_dict()
    
    # Error type breakdown
    error_counts = df[df["status"] == "error"]["error_type"].fillna("unspecified").value_counts().to_dict() if "status" in df else {}

    return {
        "metrics": {
            "total_traces": total_traces,
            "total_spend": round(total_spend, 4),
            "avg_latency": round(avg_latency, 1),
            "avg_ttft": round(avg_ttft, 1),
            "success_rate": success_rate,
            "avg_judge_score": avg_judge_score
        },
        "spend_series": spend_series,
        "model_distribution": model_counts,
        "error_distribution": error_counts
    }

@app.get("/api/traces")
def list_traces(
    status: Optional[str] = Query(None),
    prompt_version: Optional[str] = Query(None),
    model: Optional[str] = Query(None),
    search: Optional[str] = Query(None)
):
    traces = get_all_traces()
    if not traces:
        return []
    
    # Filter
    if status and status != "all":
        traces = [t for t in traces if t.get("status") == status]
    if prompt_version and prompt_version != "all":
        traces = [t for t in traces if t.get("prompt_version") == prompt_version]
    if model and model != "all":
        traces = [t for t in traces if t.get("model") == model]
    if search:
        search_lower = search.lower()
        traces = [
            t for t in traces 
            if (t.get("input") and search_lower in t["input"].lower()) 
            or (t.get("output") and search_lower in t["output"].lower())
            or (t.get("id") and search_lower in t["id"].lower())
        ]
    return traces

@app.get("/api/traces/{trace_id}")
def get_trace_detail(trace_id: str):
    trace = get_trace_by_id(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Trace not found")
    
    # Fetch child spans if this is a parent trace, or sibling spans if part of tree
    all_traces = get_all_traces()
    children = [t for t in all_traces if t.get("parent_id") == trace_id]
    parent = get_trace_by_id(trace["parent_id"]) if trace.get("parent_id") else None

    return {
        "trace": trace,
        "parent": parent,
        "children": children
    }

@app.post("/api/traces/{trace_id}/feedback")
def set_feedback(trace_id: str, body: FeedbackRequest):
    if body.feedback not in ["up", "down"]:
        raise HTTPException(status_code=400, detail="Feedback must be 'up' or 'down'")
    success = update_trace_feedback(trace_id, body.feedback)
    if not success:
        raise HTTPException(status_code=404, detail="Trace not found")
    return {"status": "success", "trace_id": trace_id, "feedback": body.feedback}

@app.get("/api/drift")
def get_drift_analysis(window_size: int = 30):
    traces = get_all_traces()
    alerts = check_for_drift(traces, window_size=window_size)
    
    # Compute distribution histograms and PSI for baseline vs current window
    sorted_traces = sorted(traces, key=lambda t: t.get("timestamp", ""))
    successful_traces = [t for t in sorted_traces if t.get("status") == "success"]
    
    latency_psi = 0.0
    cost_psi = 0.0
    
    if len(successful_traces) >= window_size * 2:
        baseline = successful_traces[:-window_size]
        target = successful_traces[-window_size:]
        
        b_lat = [t.get("latency_ms", 0.0) for t in baseline]
        t_lat = [t.get("latency_ms", 0.0) for t in target]
        latency_psi = calculate_psi(b_lat, t_lat)
        
        b_cost = [t.get("cost", 0.0) for t in baseline]
        t_cost = [t.get("cost", 0.0) for t in target]
        cost_psi = calculate_psi(b_cost, t_cost)

    return {
        "alerts": alerts,
        "metrics": {
            "latency_psi": round(latency_psi, 4),
            "cost_psi": round(cost_psi, 4),
            "total_samples": len(successful_traces),
            "window_size": window_size
        }
    }

@app.get("/api/ab_test")
def get_ab_test_metrics():
    traces = get_all_traces()
    if not traces:
        return []
    
    df = pd.DataFrame(traces)
    if "prompt_version" not in df or df["prompt_version"].dropna().empty:
        return []

    # Group by prompt version
    versions = []
    for version, group in df.groupby("prompt_version"):
        total = len(group)
        success = len(group[group["status"] == "success"])
        avg_lat = float(group["latency_ms"].mean())
        avg_cost = float(group["cost"].mean())
        avg_judge = float(group["judge_score"].dropna().mean()) if not group["judge_score"].dropna().empty else 0.0
        hallucination_rate = float((group["hallucination_flag"] == True).sum() / total * 100) if "hallucination_flag" in group else 0.0
        
        up_votes = len(group[group["user_feedback"] == "up"])
        down_votes = len(group[group["user_feedback"] == "down"])

        versions.append({
            "version": version,
            "total_traces": total,
            "success_rate": round((success / total) * 100, 1),
            "avg_latency_ms": round(avg_lat, 1),
            "avg_cost_usd": round(avg_cost, 5),
            "avg_judge_score": round(avg_judge, 2),
            "hallucination_rate": round(hallucination_rate, 1),
            "up_votes": up_votes,
            "down_votes": down_votes
        })
        
    return versions

@app.post("/api/simulate")
def trigger_simulation():
    generate_simulated_data()
    return {"status": "success", "message": "Simulated traces generated successfully"}

@app.post("/api/sandbox/run")
def run_sandbox_trace(request: SandboxRunRequest):
    if not request.prompt or not request.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt cannot be empty")

    client = None
    if request.mode == "live":
        if not request.api_key:
            raise HTTPException(status_code=400, detail="Anthropic API key required for live mode")
        try:
            import anthropic
            client = anthropic.Anthropic(api_key=request.api_key)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to initialize Anthropic client: {str(e)}")

    model = request.model or "claude-3-5-sonnet-20241022"
    version = request.prompt_version or "v1.1.0"
    scenario = request.scenario_type or "rag"
    
    rag_ctx = request.rag_context
    if not rag_ctx and scenario in ["rag", "agent"]:
        rag_ctx = "Verified Fact Sheet: VANTAGE is an open-source observability framework with sub-2ms tracing overhead and automated drift detection."

    created_root_id = None
    output_text = ""

    if scenario in ["rag", "agent"]:
        with trace_span(f"{scenario.upper()} Multi-Step Workflow", span_kind="agent", prompt_version=version) as root_span:
            created_root_id = root_span.trace_id
            with trace_span("Vector Semantic Retrieval", span_kind="tool", prompt_version=version):
                pass  # Simulated tool step
            
            response = observe_llm_call(
                client=client,
                model=model,
                messages=[{"role": "user", "content": request.prompt}],
                prompt_version=version,
                rag_context=rag_ctx
            )
            if hasattr(response, "content") and response.content:
                output_text = getattr(response.content[0], "text", str(response.content[0]))
            elif isinstance(response, str):
                output_text = response
    else:
        response = observe_llm_call(
            client=client,
            model=model,
            messages=[{"role": "user", "content": request.prompt}],
            prompt_version=version,
            rag_context=rag_ctx
        )
        if hasattr(response, "id"):
            created_root_id = response.id
        if hasattr(response, "content") and response.content:
            output_text = getattr(response.content[0], "text", str(response.content[0]))
        elif isinstance(response, str):
            output_text = response

    all_traces = get_all_traces()
    return {
        "status": "success",
        "mode": request.mode,
        "scenario": scenario,
        "model": model,
        "prompt_version": version,
        "root_trace_id": created_root_id,
        "recent_traces": all_traces[:5] if all_traces else []
    }

# --------------------------------------------------------------------------
# Serve Static Frontend Files (`src/vantage/web`)
# --------------------------------------------------------------------------

WEB_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "web"))
if os.path.exists(WEB_DIR):
    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

    @app.get("/")
    def serve_dashboard():
        index_path = os.path.join(WEB_DIR, "index.html")
        if os.path.exists(index_path):
            return FileResponse(index_path)
        return JSONResponse({"error": "Dashboard index.html not found"}, status_code=404)
