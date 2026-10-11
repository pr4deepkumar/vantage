import pytest

from tests.fakes import FakeClient, message, stream_events
from vantage.core import db
from vantage.core import instrumentation as ins
from vantage.core.instrumentation import (
    calculate_usd_cost,
    get_text_from_messages,
    observe_llm_call,
    trace_span,
    wait_for_evaluations,
)
from vantage.eval.judge import Judgement

MESSAGES = [{"role": "user", "content": "What is the capital of France?"}]


def llm_spans():
    return db.query_traces(span_kind="llm")


def test_calculate_usd_cost():
    assert calculate_usd_cost("claude-sonnet-5-5", 1_000_000, 1_000_000) == pytest.approx(12.00)  # $2 + $10
    assert calculate_usd_cost("a-model-without-a-price", 1000, 1000) is None


def test_get_text_from_messages_string():
    assert get_text_from_messages("Hello World") == "Hello World"


def test_get_text_from_messages_list():
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."},
    ]
    formatted = get_text_from_messages(messages)
    assert "User: What is Python?" in formatted
    assert "Assistant: Python is a programming language." in formatted


def test_trace_span_records_a_tree_with_start_times():
    with trace_span("pipeline", span_kind="agent") as root, trace_span("search", span_kind="tool") as step:
        step.output = "3 documents"

    parent, child = db.get_trace_by_id(root.trace_id), db.get_trace_by_id(step.trace_id)
    assert parent["parent_id"] is None
    assert child["parent_id"] == root.trace_id
    assert child["output"] == "3 documents"
    assert parent["timestamp"] <= child["timestamp"]  # spans are stamped when they start
    assert parent["latency_ms"] >= child["latency_ms"]


def test_trace_span_records_failures_and_reraises():
    with pytest.raises(TimeoutError), trace_span("flaky step", span_kind="tool") as span:
        raise TimeoutError("upstream timed out")
    recorded = db.get_trace_by_id(span.trace_id)
    assert (recorded["status"], recorded["error_type"]) == ("error", "timeout")


def test_llm_calls_inside_a_span_become_its_children():
    with trace_span("pipeline", span_kind="agent") as root:
        observe_llm_call(FakeClient(result=message("Paris.")), "m", MESSAGES, "v1.0.0")
    [span] = llm_spans()
    assert span["parent_id"] == root.trace_id


def test_streamed_call_passes_every_event_through_and_records_ttft_and_usage():
    events = list(stream_events())
    client = FakeClient(result=lambda: iter(events))

    received = list(
        observe_llm_call(client, "claude-sonnet-5-5", MESSAGES, "v1.0.0", stream=True, max_tokens=100)
    )

    assert received == events  # message_start and message_delta included
    assert client.requests[0]["stream"] is True and client.requests[0]["max_tokens"] == 100
    [span] = llm_spans()
    assert span["output"] == "Hello"
    assert (span["input_tokens"], span["output_tokens"], span["stop_reason"]) == (12, 3, "end_turn")
    assert span["status"] == "success"
    assert 0 <= span["ttft_ms"] <= span["latency_ms"]
    assert span["cost"] == pytest.approx(calculate_usd_cost("claude-sonnet-5-5", 12, 3))


def test_non_streamed_response_reads_every_text_block():
    response = message("Paris", " is the capital.")
    assert observe_llm_call(FakeClient(result=response), "m", MESSAGES, "v1.0.0") is response
    [span] = llm_spans()
    assert span["output"] == "Paris is the capital."
    assert span["status"] == "success"
    assert span["ttft_ms"] is None
    assert span["cost"] is None  # no price for this model


def test_a_storage_failure_never_reaches_the_caller(monkeypatch):
    def broken_store(_):
        raise RuntimeError("database is locked")

    monkeypatch.setattr(ins, "save_trace", broken_store)
    response = message("fine")
    assert observe_llm_call(FakeClient(result=response), "m", MESSAGES, "v1.0.0") is response
    events = list(stream_events())
    assert (
        list(observe_llm_call(FakeClient(result=lambda: iter(events)), "m", MESSAGES, "v1.0.0", stream=True))
        == events
    )


@pytest.mark.parametrize("stream", [False, True])
def test_failed_calls_are_recorded_and_reraised(stream):
    with pytest.raises(TimeoutError):
        observe_llm_call(FakeClient(error=TimeoutError("slow")), "m", MESSAGES, "v1.0.0", stream=stream)
    [span] = llm_spans()
    assert (span["status"], span["error_type"]) == ("error", "timeout")


def test_a_stream_that_breaks_midway_keeps_the_partial_output():
    def broken():
        yield from list(stream_events(pieces=("Par", "is")))[:6]  # up to the first text delta
        raise ConnectionError("connection reset")

    with pytest.raises(ConnectionError):
        list(observe_llm_call(FakeClient(result=broken), "m", MESSAGES, "v1.0.0", stream=True))
    [span] = llm_spans()
    assert (span["status"], span["output"], span["input_tokens"]) == ("error", "Par", 12)


def test_a_stream_abandoned_early_is_recorded_as_cancelled():
    stream = observe_llm_call(FakeClient(result=stream_events), "m", MESSAGES, "v1.0.0", stream=True)
    next(stream)
    stream.close()
    [span] = llm_spans()
    assert (span["status"], span["error_type"]) == ("error", "cancelled")


def test_refusals_and_truncation_are_visible():
    observe_llm_call(FakeClient(result=message("", stop_reason="refusal")), "m", MESSAGES, "v1.0.0")
    observe_llm_call(
        FakeClient(result=message("Partial ans", stop_reason="max_tokens")), "m", MESSAGES, "v1.0.0"
    )
    by_reason = {span["stop_reason"]: span for span in llm_spans()}
    assert (by_reason["refusal"]["status"], by_reason["refusal"]["error_type"]) == ("error", "refusal")
    assert by_reason["max_tokens"]["status"] == "success"


def test_json_mode_flags_malformed_output():
    observe_llm_call(FakeClient(result=message("{not json")), "m", MESSAGES, "v1.0.0", is_json_mode=True)
    [span] = llm_spans()
    assert span["error_type"] == "malformed_json"


def test_responses_are_evaluated_after_the_call_returns(monkeypatch):
    seen = []

    def fake_judge(prompt, response, rag_context):
        seen.append((response, rag_context))
        return Judgement(score=4, coherence=4, relevance=5, hallucinated=False)

    monkeypatch.setattr(ins, "judge_response", fake_judge)
    observe_llm_call(
        FakeClient(result=message("Paris.")), "m", MESSAGES, "v1.0.0", rag_context="Paris is the capital."
    )
    observe_llm_call(FakeClient(result=message("Skipped.")), "m", MESSAGES, "v1.0.0", evaluate=False)
    wait_for_evaluations(timeout=5)

    assert seen == [("Paris.", "Paris is the capital.")]
    evaluated = next(span for span in llm_spans() if span["output"] == "Paris.")
    assert (evaluated["judge_score"], evaluated["judge_relevance"], evaluated["hallucination_flag"]) == (
        4,
        5,
        False,
    )
