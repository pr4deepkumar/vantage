"""An open-weights chat model run in-process with Hugging Face transformers, answering in the shape of
the Anthropic Messages API (``client.messages.create``, the same response and stream events), so
``observe_llm_call`` traces it exactly like a Claude call: real latency, time to first token and
token counts, at no cost and without an API key."""

from __future__ import annotations

import logging
import threading
import uuid
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Protocol

from vantage.core.instrumentation import PRICES_PER_MTOK
from vantage.local.runtime import device_lock, import_torch, load_failure, pick_device

logger = logging.getLogger(__name__)


class Generation(Protocol):
    """One generation in progress. Iterate it for text pieces as they're generated; token counts
    and the stop reason are final once iteration ends."""

    input_tokens: int
    output_tokens: int
    stop_reason: str

    def __iter__(self) -> Iterator[str]: ...


class Backend(Protocol):
    def start(self, chat: list[dict], max_new_tokens: int) -> Generation: ...


class HuggingFaceBackend:
    """Greedy decoding with a chat model's own template, streamed token by token. Runs on CUDA or
    the Apple Silicon GPU when available, else the CPU."""

    def __init__(self, model_name: str, *, device: str = "auto") -> None:
        torch = import_torch()
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self.device = pick_device(torch, device)
        # bfloat16 halves memory on GPUs and is what Qwen-family models are trained in; the CPU
        # path stays in float32, which every CPU supports at full speed.
        dtype = torch.float32 if self.device == "cpu" else torch.bfloat16
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
        except OSError as exc:
            raise load_failure(model_name, exc) from exc
        with device_lock(self.device):  # the other model may be running on the GPU
            self.model = model.to(self.device).eval()
        eos = self.model.generation_config.eos_token_id
        self.eos_ids = set(eos) if isinstance(eos, list) else {eos}
        self.lock = threading.Lock()  # one generation at a time per model instance
        logger.info("Loaded %s on %s", model_name, self.device)

    def start(self, chat: list[dict], max_new_tokens: int) -> Generation:
        inputs = self.tokenizer.apply_chat_template(
            chat, add_generation_prompt=True, return_tensors="pt", return_dict=True
        ).to(self.device)
        return _HuggingFaceGeneration(self, inputs, max_new_tokens)


class _HuggingFaceGeneration:
    def __init__(self, backend: HuggingFaceBackend, inputs, max_new_tokens: int) -> None:
        from transformers import TextIteratorStreamer

        self.input_tokens = int(inputs["input_ids"].shape[1])
        self.output_tokens = 0
        self.stop_reason = "end_turn"
        self._backend = backend
        self._inputs = inputs
        self._max_new_tokens = max_new_tokens
        self._error: BaseException | None = None
        self._streamer = TextIteratorStreamer(backend.tokenizer, skip_prompt=True, skip_special_tokens=True)
        self._thread = threading.Thread(target=self._generate, name="vantage-generate", daemon=True)
        self._thread.start()

    def _generate(self) -> None:
        backend = self._backend
        try:
            with backend.lock, device_lock(backend.device), backend._torch.inference_mode():
                output = backend.model.generate(
                    **self._inputs,
                    streamer=self._streamer,
                    max_new_tokens=self._max_new_tokens,
                    do_sample=False,  # greedy: the same prompt always gets the same answer
                    repetition_penalty=1.05,
                    pad_token_id=backend.tokenizer.eos_token_id,
                )
            new_tokens = output[0][self.input_tokens :].tolist()
            self.output_tokens = len(new_tokens)
            hit_limit = self.output_tokens >= self._max_new_tokens and new_tokens[-1] not in backend.eos_ids
            self.stop_reason = "max_tokens" if hit_limit else "end_turn"
        except BaseException as exc:
            self._error = exc
            self._streamer.end()  # unblock the reader

    def __iter__(self) -> Iterator[str]:
        for piece in self._streamer:
            if piece:
                yield piece
        self._thread.join()
        if self._error is not None:
            raise self._error


def _chat(system: str | None, messages: list[dict]) -> list[dict]:
    chat = [{"role": "system", "content": system}] if system else []
    for message in messages:
        content = message["content"]
        if not isinstance(content, str):  # content blocks
            content = "".join(
                block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
                for block in content
            )
        chat.append({"role": message["role"], "content": content})
    return chat


def _events(generation: Generation, model: str) -> Iterator[SimpleNamespace]:
    """The event sequence of a streamed Messages API response."""
    message = SimpleNamespace(
        id=f"msg_local_{uuid.uuid4().hex[:24]}",
        type="message",
        role="assistant",
        model=model,
        content=[],
        stop_reason=None,
        usage=SimpleNamespace(input_tokens=generation.input_tokens, output_tokens=0),
    )
    yield SimpleNamespace(type="message_start", message=message)
    yield SimpleNamespace(
        type="content_block_start", index=0, content_block=SimpleNamespace(type="text", text="")
    )
    for piece in generation:
        yield SimpleNamespace(
            type="content_block_delta", index=0, delta=SimpleNamespace(type="text_delta", text=piece)
        )
    yield SimpleNamespace(type="content_block_stop", index=0)
    yield SimpleNamespace(
        type="message_delta",
        delta=SimpleNamespace(stop_reason=generation.stop_reason),
        usage=SimpleNamespace(output_tokens=generation.output_tokens),
    )
    yield SimpleNamespace(type="message_stop")


def _collect(events: Iterator[SimpleNamespace]) -> SimpleNamespace:
    """The complete message a non-streamed Messages API call returns."""
    message, parts = None, []
    for event in events:
        if event.type == "message_start":
            message = event.message
        elif event.type == "content_block_delta":
            parts.append(event.delta.text)
        elif event.type == "message_delta":
            message.stop_reason = event.delta.stop_reason
            message.usage.output_tokens = event.usage.output_tokens
    message.content = [SimpleNamespace(type="text", text="".join(parts))]
    return message


class LocalChatClient:
    """Drop-in for ``anthropic.Anthropic`` as far as ``messages.create`` goes. The model loads on
    first use, or in ``warm_up``."""

    def __init__(
        self,
        model_name: str,
        *,
        device: str = "auto",
        max_new_tokens: int = 512,
        backend: Backend | None = None,
    ) -> None:
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.messages = _Messages(self)
        self.state = "ready" if backend else "not loaded"  # then "loading", "ready" or "failed"
        self.error: str | None = None
        self._device = device
        self._backend = backend
        self._load_lock = threading.Lock()
        PRICES_PER_MTOK.setdefault(model_name, (0.0, 0.0))  # runs on this machine: no per-token price

    def warm_up(self) -> None:
        self._load()

    def _load(self) -> Backend:
        with self._load_lock:
            if self._backend is None:
                self.state = "loading"
                try:
                    self._backend = HuggingFaceBackend(self.model_name, device=self._device)
                except Exception as exc:
                    self.state, self.error = "failed", str(exc)
                    raise
                self.state, self.error = "ready", None
        return self._backend


class _Messages:
    def __init__(self, client: LocalChatClient) -> None:
        self._client = client

    def create(
        self,
        *,
        messages: list[dict],
        model: str | None = None,
        system: str | None = None,
        max_tokens: int | None = None,
        stream: bool = False,
    ):
        generation = self._client._load().start(
            _chat(system, messages), max_tokens or self._client.max_new_tokens
        )
        events = _events(generation, model or self._client.model_name)
        return events if stream else _collect(events)
