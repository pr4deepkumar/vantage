"""Stand-ins for model clients, shaped like the Anthropic Messages API."""

from types import SimpleNamespace as NS


def stream_events(pieces=("Hel", "lo"), input_tokens=12, output_tokens=3, stop_reason="end_turn"):
    """A streamed response with a thinking block before the text, as current Claude models send."""
    yield NS(type="message_start", message=NS(usage=NS(input_tokens=input_tokens, output_tokens=1)))
    yield NS(type="content_block_start", index=0, content_block=NS(type="thinking", thinking=""))
    yield NS(type="content_block_delta", index=0, delta=NS(type="thinking_delta", thinking=""))
    yield NS(type="content_block_stop", index=0)
    yield NS(type="content_block_start", index=1, content_block=NS(type="text", text=""))
    for piece in pieces:
        yield NS(type="content_block_delta", index=1, delta=NS(type="text_delta", text=piece))
    yield NS(type="content_block_stop", index=1)
    yield NS(type="message_delta", delta=NS(stop_reason=stop_reason), usage=NS(output_tokens=output_tokens))
    yield NS(type="message_stop")


def message(*texts, input_tokens=10, output_tokens=5, stop_reason="end_turn"):
    content = [NS(type="thinking", thinking="")] + [NS(type="text", text=text) for text in texts]
    return NS(
        content=content,
        usage=NS(input_tokens=input_tokens, output_tokens=output_tokens),
        stop_reason=stop_reason,
    )


class FakeClient:
    """``messages.create`` returns ``result`` (called first if it's a function) or raises ``error``."""

    def __init__(self, result=None, error=None):
        self.messages = self
        self.result = result
        self.error = error
        self.requests = []

    def create(self, **request):
        self.requests.append(request)
        if self.error:
            raise self.error
        return self.result() if callable(self.result) else self.result


class FakeGeneration:
    def __init__(self, pieces, input_tokens=7, stop_reason="end_turn"):
        self._pieces = list(pieces)
        self.input_tokens = input_tokens
        self.output_tokens = len(self._pieces)
        self.stop_reason = stop_reason

    def __iter__(self):
        yield from self._pieces


class FakeBackend:
    """A LocalChatClient backend that answers each call with the next reply, split into words."""

    def __init__(self, *replies):
        self.replies = list(replies) or ["A short answer."]
        self.chats = []

    def start(self, chat, max_new_tokens):
        self.chats.append((chat, max_new_tokens))
        reply = self.replies[min(len(self.chats), len(self.replies)) - 1]
        words = reply.split(" ")
        return FakeGeneration([words[0]] + [" " + word for word in words[1:]])
