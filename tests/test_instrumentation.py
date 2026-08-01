import pytest
from vantage.core.instrumentation import calculate_usd_cost, get_text_from_messages, trace_span

def test_calculate_usd_cost():
    # Model with known pricing
    cost = calculate_usd_cost("claude-3-5-sonnet-latest", 1_000_000, 1_000_000)
    assert round(cost, 2) == 18.00  # $3.00 input + $15.00 output

def test_get_text_from_messages_string():
    assert get_text_from_messages("Hello World") == "Hello World"

def test_get_text_from_messages_list():
    messages = [
        {"role": "user", "content": "What is Python?"},
        {"role": "assistant", "content": "Python is a programming language."}
    ]
    formatted = get_text_from_messages(messages)
    assert "User: What is Python?" in formatted
    assert "Assistant: Python is a programming language." in formatted

def test_trace_span_context_manager():
    with trace_span("test_agent_step", span_kind="agent") as span:
        assert span.name == "test_agent_step"
        assert span.trace_id is not None