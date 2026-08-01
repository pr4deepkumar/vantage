import pytest
from fastapi.testclient import TestClient
from vantage.api.main import app

client = TestClient(app)

def test_api_overview():
    response = client.get("/api/overview")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "total_traces" in data["metrics"]
    assert "spend_series" in data

def test_api_traces_list():
    response = client.get("/api/traces")
    assert response.status_code == 200
    traces = response.json()
    assert isinstance(traces, list)
    if traces:
        trace_id = traces[0]["id"]
        detail = client.get(f"/api/traces/{trace_id}")
        assert detail.status_code == 200
        assert "trace" in detail.json()

def test_api_drift():
    response = client.get("/api/drift")
    assert response.status_code == 200
    data = response.json()
    assert "metrics" in data
    assert "alerts" in data

def test_api_ab_test():
    response = client.get("/api/ab_test")
    assert response.status_code == 200
    assert isinstance(response.json(), list)

def test_api_sandbox_run_simulated():
    payload = {
        "prompt": "Test sandbox query for RAG pipeline",
        "scenario_type": "rag",
        "model": "claude-3-5-sonnet-20241022",
        "prompt_version": "v1.1.0",
        "mode": "simulated",
        "rag_context": "Test knowledge fact"
    }
    response = client.post("/api/sandbox/run", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["mode"] == "simulated"
    assert "recent_traces" in data
    assert len(data["recent_traces"]) > 0

def test_api_sandbox_run_empty_prompt():
    payload = {
        "prompt": "   ",
        "mode": "simulated"
    }
    response = client.post("/api/sandbox/run", json=payload)
    assert response.status_code == 400
