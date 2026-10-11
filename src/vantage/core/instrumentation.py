"""Tracing for LLM pipelines.

``trace_span`` records one pipeline step (an agent, a chain, a tool call) and makes it the parent of
every span opened inside it, through a ``contextvars.ContextVar``, so nested steps form a trace tree
without passing ids around.

``observe_llm_call`` wraps ``client.messages.create`` of any client shaped like the Anthropic
Messages API: the Anthropic SDK client, or ``vantage.local.LocalChatClient``, which runs an open
model on this machine. Each call is recorded with its start time, latency, time to first token
(streamed calls), token usage, cost, stop reason and error type. Evaluation (grounding check, judge
scores) runs on a background thread after the response is returned, so it adds no latency to the call.

Telemetry never breaks the traced application: a span that can't be recorded is logged, not raised.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
import uuid
from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from vantage.core.db import save_trace, update_trace_judgement
from vantage.eval.judge import judge_response

logger = logging.getLogger(__name__)

# The span that new spans attach to as children.
active_parent_span_id: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "active_parent_span_id", default=None
)

# USD per million (input, output) tokens: Anthropic list prices as of 2026-11-03. Long-prompt price
# tiers and prompt-caching discounts are not modeled. A model without an entry is recorded without a
# cost; local models register a price of zero.
PRICES_PER_MTOK: dict[str, tuple[float, float]] = {
    "claude-fable-5-2": (8.00, 40.00),
    "claude-opus-5-5": (4.00, 20.00),
    "claude-opus-5": (5.00, 25.00),
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5-5": (2.00, 10.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-5-5": (0.10, 0.50),
    "claude-haiku-4-5": (1.00, 5.00),
}


def calculate_usd_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    prices = PRICES_PER_MTOK.get(model)
    if prices is None:
        return None
    return (input_tokens * prices[0] + output_tokens * prices[1]) / 1_000_000


def get_text_from_messages(messages):
    """Utility to convert messages input format into a single string for DB logging."""
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        text_parts = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if isinstance(content, list):
                block_texts = []
                for block in content:
                    if isinstance(block, dict):
                        block_texts.append(block.get("text", ""))
                    elif hasattr(block, "text"):
                        block_texts.append(block.text)
                content_str = " ".join(block_texts)
            else:
                content_str = str(content)
            text_parts.append(f"{role.capitalize()}: {content_str}")
        return "\n".join(text_parts)
    return str(messages)


# Safe imports for Anthropic exception classes
try:
    import anthropic
except ImportError:
    anthropic = None


def classify_exception(exc: BaseException) -> str:
    if anthropic:
        if isinstance(exc, anthropic.APITimeoutError):
            return "timeout"
        if isinstance(exc, anthropic.RateLimitError):
            return "rate_limit"
        if isinstance(exc, anthropic.APIStatusError):
            if exc.status_code == 429:
                return "rate_limit"
            if exc.status_code == 408:
                return "timeout"

    lower_name = type(exc).__name__.lower()
    if "timeout" in lower_name:
        return "timeout"
    if "rate" in lower_name or "429" in lower_name:
        return "rate_limit"

    return "other"


def _record(**fields) -> None:
    try:
        save_trace(fields)
    except Exception:
        logger.warning("Could not record span %s", fields.get("id"), exc_info=True)


class trace_span:
    """
    Context manager to trace logical steps (Agents, Chains, or Tools) in a pipeline.
    Automatically handles parent-child context propagation. Set ``span.output`` inside the block
    to record what the step produced.
    """

    def __init__(
        self, name: str, span_kind: str = "chain", prompt_version: str = "v1.0.0", parent_id: str = None
    ):
        self.name = name
        self.span_kind = span_kind
        self.prompt_version = prompt_version
        self.parent_id = parent_id
        self.trace_id = str(uuid.uuid4())
        self.output = None
        self.token = None
        self.started_at = None
        self.start_time = None
        self.inherited_parent = None

    def __enter__(self):
        self.started_at = datetime.now(timezone.utc)
        self.start_time = time.perf_counter()
        self.inherited_parent = self.parent_id or active_parent_span_id.get()
        # Set current span as active parent for nested sub-calls
        self.token = active_parent_span_id.set(self.trace_id)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        active_parent_span_id.reset(self.token)
        failed = exc_type is not None
        _record(
            id=self.trace_id,
            timestamp=self.started_at,
            prompt_version=self.prompt_version,
            model="system" if self.span_kind == "tool" else "orchestrator",
            parent_id=self.inherited_parent,
            span_kind=self.span_kind,
            input=f"Span step: {self.name}",
            output=f"Step failed: {exc_val}" if failed else (self.output or "Completed step successfully."),
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency_ms=(time.perf_counter() - self.start_time) * 1000,
            status="error" if failed else "success",
            error_type=classify_exception(exc_val) if failed else None,
        )


class _LLMCall:
    """Collects the telemetry of one LLM call and records it as a span."""

    def __init__(
        self, *, model, prompt_version, input_text, parent_id, span_kind, rag_context, is_json_mode, evaluate
    ):
        self.trace_id = str(uuid.uuid4())
        self.started_at = datetime.now(timezone.utc)
        self._start = time.perf_counter()
        self.ttft_ms = None
        self.model = model
        self.prompt_version = prompt_version
        self.input_text = input_text
        self.parent_id = parent_id
        self.span_kind = span_kind
        self.rag_context = rag_context
        self.is_json_mode = is_json_mode
        self.evaluate = evaluate

    def elapsed_ms(self) -> float:
        return (time.perf_counter() - self._start) * 1000

    def record(
        self, *, status, error_type=None, output=None, input_tokens=0, output_tokens=0, stop_reason=None
    ):
        _record(
            id=self.trace_id,
            timestamp=self.started_at,
            prompt_version=self.prompt_version,
            model=self.model,
            parent_id=self.parent_id,
            span_kind=self.span_kind,
            input=self.input_text,
            output=output,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost=calculate_usd_cost(self.model, input_tokens, output_tokens),
            latency_ms=self.elapsed_ms(),
            ttft_ms=self.ttft_ms,
            status=status,
            error_type=error_type,
            stop_reason=stop_reason,
            rag_context=self.rag_context,
        )
        if status == "success" and self.evaluate:
            _queue_evaluation(self.trace_id, self.input_text, output, self.rag_context)

    def succeed(self, text: str, input_tokens: int, output_tokens: int, stop_reason: str | None) -> None:
        status, error_type = "success", None
        if stop_reason == "refusal":
            status, error_type = "error", "refusal"
        elif not text:
            status, error_type = "error", "empty_response"
        elif self.is_json_mode:
            try:
                json.loads(text)
            except json.JSONDecodeError:
                status, error_type = "error", "malformed_json"
        self.record(
            status=status,
            error_type=error_type,
            output=text,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            stop_reason=stop_reason,
        )

    def fail(self, exc, *, error_type=None, output=None, input_tokens=0, output_tokens=0) -> None:
        self.record(
            status="error",
            error_type=error_type or classify_exception(exc),
            output=output or None,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def observe_response(self, response) -> None:
        # Read every text block: with thinking on, the first block is often a thinking block.
        blocks = getattr(response, "content", None) or []
        text = "".join(block.text for block in blocks if getattr(block, "type", None) == "text")
        usage = getattr(response, "usage", None)
        self.succeed(
            text,
            getattr(usage, "input_tokens", 0) or 0,
            getattr(usage, "output_tokens", 0) or 0,
            getattr(response, "stop_reason", None),
        )

    def observe_stream(self, events: Iterable) -> Iterator:
        """Pass every event through unchanged, measuring as they go by."""
        parts: list[str] = []
        input_tokens = output_tokens = 0
        stop_reason = None
        try:
            for event in events:
                kind = getattr(event, "type", None)
                if kind == "content_block_delta":
                    delta = getattr(event, "delta", None)
                    if getattr(delta, "type", None) == "text_delta":
                        if self.ttft_ms is None:
                            self.ttft_ms = self.elapsed_ms()
                        parts.append(delta.text)
                elif kind == "message_start":
                    usage = getattr(getattr(event, "message", None), "usage", None)
                    input_tokens = getattr(usage, "input_tokens", 0) or 0
                elif kind == "message_delta":
                    stop_reason = getattr(getattr(event, "delta", None), "stop_reason", None) or stop_reason
                    # Cumulative for the whole message.
                    output_tokens = (
                        getattr(getattr(event, "usage", None), "output_tokens", None) or output_tokens
                    )
                yield event
        except GeneratorExit:  # the caller stopped reading before the end
            self.fail(None, error_type="cancelled", output="".join(parts), input_tokens=input_tokens)
            raise
        except Exception as exc:
            self.fail(exc, output="".join(parts), input_tokens=input_tokens, output_tokens=output_tokens)
            raise
        finally:
            close = getattr(events, "close", None)
            if callable(close):
                close()
        self.succeed("".join(parts), input_tokens, output_tokens, stop_reason)


def observe_llm_call(
    client,
    model: str,
    messages: list,
    prompt_version: str,
    system: str = None,
    rag_context: str = None,
    stream: bool = False,
    is_json_mode: bool = False,
    parent_id: str = None,
    span_kind: str = "llm",
    evaluate: bool = True,
    **kwargs,
):
    """
    Call ``client.messages.create`` and record the call as a span. Returns what the client returns:
    the message or, with ``stream=True``, an iterator over the same events. Extra keyword arguments
    (``max_tokens``, ...) are passed to the client. ``evaluate=False`` skips the grounding check and
    judge for this call.
    """
    input_text = get_text_from_messages(messages)
    if system:
        input_text = f"[System: {system}]\n" + input_text

    call = _LLMCall(
        model=model,
        prompt_version=prompt_version,
        input_text=input_text,
        parent_id=parent_id or active_parent_span_id.get(),
        span_kind=span_kind,
        rag_context=rag_context,
        is_json_mode=is_json_mode,
        evaluate=evaluate,
    )
    request = dict(model=model, messages=messages, **kwargs)
    if system is not None:
        request["system"] = system

    try:
        result = (
            client.messages.create(**request, stream=True) if stream else client.messages.create(**request)
        )
    except Exception as exc:
        call.fail(exc)
        raise

    if stream:
        return call.observe_stream(result)
    try:
        call.observe_response(result)
    except Exception:
        logger.warning("Could not record LLM call %s", call.trace_id, exc_info=True)
    return result


# One worker: evaluations run in order, and a local model must not be driven from several threads.
_evaluator = ThreadPoolExecutor(max_workers=1, thread_name_prefix="vantage-eval")


def _queue_evaluation(trace_id: str, prompt: str, response: str, rag_context: str | None) -> None:
    try:
        _evaluator.submit(_evaluate, trace_id, prompt, response, rag_context)
    except RuntimeError:  # the interpreter is shutting down
        logger.warning("Skipped evaluation of span %s", trace_id)


def _evaluate(trace_id: str, prompt: str, response: str, rag_context: str | None) -> None:
    try:
        judgement = judge_response(prompt, response, rag_context)
        fields = {
            "judge_score": judgement.score,
            "judge_coherence": judgement.coherence,
            "judge_relevance": judgement.relevance,
            "hallucination_flag": judgement.hallucinated,
        }
        fields = {key: value for key, value in fields.items() if value is not None}
        if fields:
            update_trace_judgement(trace_id, **fields)
    except Exception:
        logger.warning("Could not evaluate span %s", trace_id, exc_info=True)


def wait_for_evaluations(timeout: float | None = None) -> None:
    """Block until every evaluation queued so far has finished (the queue runs in order)."""
    _evaluator.submit(lambda: None).result(timeout)
