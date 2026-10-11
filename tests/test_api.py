import pytest
from fastapi.testclient import TestClient

from tests.fakes import FakeBackend
from vantage.api import main
from vantage.core import db
from vantage.local.llm import LocalChatClient
from vantage.utils.simulate_traffic import generate_simulated_data

client = TestClient(main.app)  # without the lifespan: conftest has created the tables


@pytest.fixture
def fake_model(monkeypatch):
    """Answer sandbox calls with a scripted local model."""
    backend = FakeBackend("refund policy window", "The refund window is 14 days.")
    model = LocalChatClient(main.current_model(), backend=backend)
    monkeypatch.setattr(main, "get_llm_client", lambda: model)
    return backend


def test_health_reports_the_traced_model():
    health = client.get("/api/health").json()
    assert health["status"] == "ok"
    assert (health["provider"], health["model"]) == ("local", main.settings.local_llm_model)


def test_an_empty_store_reports_no_averages():
    overview = client.get("/api/overview").json()
    assert overview["metrics"]["total_calls"] == 0
    assert overview["metrics"]["avg_latency"] is None
    assert client.get("/api/ab_test").json() == []
    drift = client.get("/api/drift").json()["metrics"]
    assert (drift["latency_psi"], drift["required_samples"]) == (None, 60)


def test_overview_counts_llm_calls_and_their_spend_only():
    generate_simulated_data()
    llm_spans = db.query_traces(span_kind="llm")
    overview = client.get("/api/overview").json()

    assert overview["metrics"]["total_calls"] == len(llm_spans) == 120
    assert overview["metrics"]["total_spend"] == pytest.approx(round(sum(t["cost"] for t in llm_spans), 4))
    assert set(overview["model_distribution"]) <= {"claude-sonnet-5-5", "claude-haiku-5-5"}
    assert sum(day["calls"] for day in overview["daily"]) == 120
    assert all(day["p95_latency"] >= day["p50_latency"] for day in overview["daily"] if day["p50_latency"])


def test_drift_compares_every_metric_with_its_baseline():
    generate_simulated_data()  # the latest calls are slower and longer by construction
    drift = client.get("/api/drift").json()
    by_metric = {c["metric"]: c for c in drift["comparisons"]}
    assert set(by_metric) == {"latency_ms", "cost", "output_tokens"}
    assert by_metric["latency_ms"]["window_mean"] > by_metric["latency_ms"]["baseline_mean"]
    assert {alert["metric"] for alert in drift["alerts"]} == {m for m, c in by_metric.items() if c["drifted"]}


def test_ab_rates_are_over_llm_calls_checked_for_grounding():
    generate_simulated_data()
    for version in client.get("/api/ab_test").json():
        checked = [
            t
            for t in db.query_traces(span_kind="llm", prompt_version=version["version"])
            if t["hallucination_flag"] is not None
        ]
        assert version["grounding_checked"] == len(checked)
        expected = 100 * sum(t["hallucination_flag"] for t in checked) / len(checked)
        assert version["hallucination_rate"] == pytest.approx(expected, abs=0.05)


def test_trace_list_filters_and_the_detail_view_returns_children():
    generate_simulated_data()
    errors = client.get("/api/traces", params={"status": "error"}).json()
    assert errors and all(t["status"] == "error" for t in errors)
    assert len(client.get("/api/traces", params={"limit": 5}).json()) == 5
    assert client.get("/api/traces", params={"search": "100%_"}).json() == []  # wildcards are literal

    root = client.get("/api/traces", params={"span_kind": "agent", "limit": 1}).json()[0]
    detail = client.get(f"/api/traces/{root['id']}").json()
    assert [child["span_kind"] for child in detail["children"]] == ["tool", "llm"]
    assert all(child["timestamp"] >= root["timestamp"] for child in detail["children"])


def test_feedback_is_stored_and_validated():
    generate_simulated_data()
    trace_id = client.get("/api/traces", params={"limit": 1}).json()[0]["id"]
    assert client.post(f"/api/traces/{trace_id}/feedback", json={"feedback": "up"}).status_code == 200
    assert db.get_trace_by_id(trace_id)["user_feedback"] == "up"
    assert client.post(f"/api/traces/{trace_id}/feedback", json={"feedback": "meh"}).status_code == 422
    assert client.post("/api/traces/missing/feedback", json={"feedback": "up"}).status_code == 404


def test_a_rag_run_returns_the_spans_it_recorded(fake_model):
    response = client.post(
        "/api/sandbox/run",
        json={
            "prompt": "What is the refund window?",
            "scenario_type": "rag",
            "prompt_version": "v1.1.0",
            "rag_context": "Refunds are accepted within 14 days.",
        },
    )
    assert response.status_code == 200
    data = response.json()
    root, llm = data["spans"]
    assert (root["span_kind"], llm["span_kind"]) == ("chain", "llm")
    assert root["id"] == data["root_trace_id"] == llm["parent_id"]
    assert llm["output"] == "refund policy window"
    assert llm["ttft_ms"] is not None and llm["rag_context"] == "Refunds are accepted within 14 days."

    chat, _ = fake_model.chats[0]
    assert chat[0] == {"role": "system", "content": main.SYSTEM_PROMPTS["v1.1.0"]}
    assert chat[1]["content"].startswith("Context:\nRefunds are accepted within 14 days.")


def test_an_agent_run_records_each_step(fake_model):
    data = client.post(
        "/api/sandbox/run",
        json={
            "prompt": "How long do I have to ask for a refund?",
            "scenario_type": "agent",
            "rag_context": "Shipping takes a week. The refund window is 14 days. Support answers within a day.",
        },
    ).json()
    assert [span["span_kind"] for span in data["spans"]] == ["agent", "llm", "tool", "llm"]
    keywords, search, answer = data["spans"][1:]
    assert keywords["output"] == "refund policy window"
    assert search["output"].startswith("Selected 1 of 3 sentences")
    assert answer["rag_context"] == "The refund window is 14 days."
    assert answer["output"] == "The refund window is 14 days."


def test_the_sandbox_validates_its_input():
    assert client.post("/api/sandbox/run", json={"prompt": "   "}).status_code == 400
    assert client.post("/api/sandbox/run", json={"prompt": "hi", "scenario_type": "chaos"}).status_code == 422
    assert (
        client.post("/api/sandbox/run", json={"prompt": "hi", "prompt_version": "<b>x</b>"}).status_code
        == 422
    )


def test_without_the_local_extra_the_sandbox_explains_what_to_install(monkeypatch):
    monkeypatch.setattr(main, "_client", None)
    monkeypatch.setattr(main, "local_models_installed", lambda: False)
    response = client.post("/api/sandbox/run", json={"prompt": "hi"})
    assert response.status_code == 503
    assert "pip install" in response.json()["detail"]
