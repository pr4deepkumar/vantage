from tests.fakes import FakeBackend
from vantage.core import db
from vantage.core.instrumentation import observe_llm_call
from vantage.local.llm import LocalChatClient


def test_a_call_returns_a_messages_api_message():
    backend = FakeBackend("Hi there")
    client = LocalChatClient("local/chat-model", backend=backend)

    reply = client.messages.create(
        messages=[{"role": "user", "content": "hello"}], system="Be brief", max_tokens=20
    )

    assert [(block.type, block.text) for block in reply.content] == [("text", "Hi there")]
    assert (reply.usage.input_tokens, reply.usage.output_tokens, reply.stop_reason) == (7, 2, "end_turn")
    assert backend.chats == [
        ([{"role": "system", "content": "Be brief"}, {"role": "user", "content": "hello"}], 20)
    ]


def test_a_streamed_call_emits_the_messages_api_event_sequence():
    client = LocalChatClient("local/chat-model", backend=FakeBackend("Hi there"))
    events = list(client.messages.create(messages=[{"role": "user", "content": "hello"}], stream=True))
    assert [event.type for event in events] == [
        "message_start",
        "content_block_start",
        "content_block_delta",
        "content_block_delta",
        "content_block_stop",
        "message_delta",
        "message_stop",
    ]
    assert events[0].message.usage.input_tokens == 7
    assert (events[-2].delta.stop_reason, events[-2].usage.output_tokens) == ("end_turn", 2)


def test_a_traced_local_call_records_ttft_tokens_and_no_cost():
    client = LocalChatClient("local/traced-model", backend=FakeBackend("Paris is the capital."))
    stream = observe_llm_call(
        client,
        "local/traced-model",
        [{"role": "user", "content": "Capital of France?"}],
        "v1.0.0",
        stream=True,
    )
    list(stream)
    [span] = db.query_traces(span_kind="llm")
    assert span["output"] == "Paris is the capital."
    assert (span["input_tokens"], span["output_tokens"], span["cost"]) == (7, 4, 0.0)
    assert span["ttft_ms"] is not None
