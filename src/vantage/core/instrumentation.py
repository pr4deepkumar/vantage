import os
import time
import uuid
import json
import logging
import contextvars
from datetime import datetime, timezone

# Import DB function
from vantage.core.db import save_trace

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("vantage.instrumentation")

# Thread-safe ContextVar to manage parent-child span relations automatically
active_parent_span_id = contextvars.ContextVar("active_parent_span_id", default=None)

# Pricing Table (USD per token)
PRICING_TABLE = {
    "claude-3-5-sonnet-20241022": (3.00 / 1e6, 15.00 / 1e6),
    "claude-3-5-sonnet-latest": (3.00 / 1e6, 15.00 / 1e6),
    "claude-3-5-sonnet": (3.00 / 1e6, 15.00 / 1e6),
    "claude-3-5-haiku-20241022": (0.80 / 1e6, 4.00 / 1e6),
    "claude-3-5-haiku-latest": (0.80 / 1e6, 4.00 / 1e6),
    "claude-3-5-haiku": (0.80 / 1e6, 4.00 / 1e6),
    "claude-3-opus-20240229": (15.00 / 1e6, 75.00 / 1e6),
    "claude-3-opus-latest": (15.00 / 1e6, 75.00 / 1e6),
    "claude-3-opus": (15.00 / 1e6, 75.00 / 1e6),
    "claude-3-haiku-20240307": (0.25 / 1e6, 1.25 / 1e6),
    "claude-3-haiku": (0.25 / 1e6, 1.25 / 1e6),
}

DEFAULT_PRICING = (3.00 / 1e6, 15.00 / 1e6)

def calculate_usd_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    pricing = PRICING_TABLE.get(model, DEFAULT_PRICING)
    return (input_tokens * pricing[0]) + (output_tokens * pricing[1])

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
    ANTHROPIC_ERRORS = (
        anthropic.APITimeoutError,
        anthropic.RateLimitError,
        anthropic.APIStatusError,
        anthropic.APIConnectionError
    )
except ImportError:
    anthropic = None
    ANTHROPIC_ERRORS = ()

def classify_exception(exc: Exception) -> str:
    exc_name = type(exc).__name__
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
    
    lower_name = exc_name.lower()
    if "timeout" in lower_name:
        return "timeout"
    if "rate" in lower_name or "429" in lower_name:
        return "rate_limit"
    
    return "other"

def log_trace(
    trace_id: str,
    prompt_version: str,
    model: str,
    parent_id: str,
    span_kind: str,
    input_str: str,
    output_str: str,
    input_tokens: int,
    output_tokens: int,
    cost: float,
    latency_ms: float,
    ttft_ms: float,
    status: str,
    error_type: str,
    rag_context: str,
    judge_score=None,
    judge_coherence=None,
    judge_relevance=None,
    hallucination_flag=None
):
    """Unified helper function to save traces to database."""
    trace_data = {
        "id": trace_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "prompt_version": prompt_version,
        "model": model,
        "parent_id": parent_id,
        "span_kind": span_kind,
        "input": input_str,
        "output": output_str,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
        "latency_ms": latency_ms,
        "ttft_ms": ttft_ms,
        "status": status,
        "error_type": error_type,
        "user_feedback": None,
        "judge_score": judge_score,
        "judge_coherence": judge_coherence,
        "judge_relevance": judge_relevance,
        "hallucination_flag": hallucination_flag,
        "rag_context": rag_context
    }
    save_trace(trace_data)

class trace_span:
    """
    Context manager to trace logical steps (Agents, Chains, or Tools) in a pipeline.
    Automatically handles parent-child context propagation.
    """
    def __init__(self, name: str, span_kind: str = "chain", prompt_version: str = "v1.0.0", parent_id: str = None):
        self.name = name
        self.span_kind = span_kind
        self.prompt_version = prompt_version
        self.parent_id = parent_id
        self.trace_id = str(uuid.uuid4())
        self.token = None
        self.start_time = None
        self.inherited_parent = None
        
    def __enter__(self):
        self.start_time = time.perf_counter()
        self.inherited_parent = self.parent_id or active_parent_span_id.get()
        # Set current span as active parent for nested sub-calls
        self.token = active_parent_span_id.set(self.trace_id)
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        active_parent_span_id.reset(self.token)
        latency = (time.perf_counter() - self.start_time) * 1000
        
        status = "success" if exc_type is None else "error"
        error_type = classify_exception(exc_val) if exc_type else None
        
        output_txt = f"Completed step successfully."
        if exc_type:
            output_txt = f"Step failed: {str(exc_val)}"
            
        log_trace(
            trace_id=self.trace_id,
            prompt_version=self.prompt_version,
            model="system" if self.span_kind == "tool" else "orchestrator",
            parent_id=self.inherited_parent,
            span_kind=self.span_kind,
            input_str=f"Span step: {self.name}",
            output_str=output_txt,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency_ms=latency,
            ttft_ms=None,
            status=status,
            error_type=error_type,
            rag_context=None
        )

# A simulated mock response for offline demo/testing
class MockUsage:
    def __init__(self, input_tokens=15, output_tokens=30):
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens

class MockTextBlock:
    def __init__(self, text):
        self.text = text
        self.type = "text"

class MockMessageResponse:
    def __init__(self, id, model, content_text, input_tokens, output_tokens):
        self.id = id
        self.model = model
        self.type = "message"
        self.role = "assistant"
        self.content = [MockTextBlock(content_text)]
        self.usage = MockUsage(input_tokens, output_tokens)

class MockStreamChunk:
    def __init__(self, type, text=None, input_tokens=0, output_tokens=0):
        self.type = type
        if type == "content_block_delta" and text:
            self.delta = type('', (object,), {"text": text})()
        if type == "message_delta" and (input_tokens or output_tokens):
            self.usage = MockUsage(input_tokens, output_tokens)

def simulate_llm_call(model, messages, stream=False):
    input_str = get_text_from_messages(messages)
    
    if "hello" in input_str.lower() or "hi" in input_str.lower():
        response_text = "Hello! I am Claude (simulated model). How can I assist you with your observability tests today?"
    elif "capital of france" in input_str.lower():
        response_text = "The capital of France is Paris, located along the Seine River in the north-central part of the country."
    elif "coherence" in input_str.lower():
        response_text = "This response is designed to be highly coherent, structured, and logical for demonstrating judge scores."
    elif "hallucinate" in input_str.lower() or "rag" in input_str.lower():
        response_text = "Based on the provided documents, France's capital is Berlin. (This is a simulated hallucination for testing purposes.)"
    elif "json" in input_str.lower():
        response_text = '{"status": "success", "data": {"message": "Simulated structured response", "records": [1, 2, 3]}}'
    else:
        response_text = f"This is a simulated response from {model} evaluating the input query of length {len(input_str)} characters."

    in_tokens = max(10, len(input_str) // 4)
    out_tokens = max(15, len(response_text) // 4)
    
    return response_text, in_tokens, out_tokens

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
    **kwargs
):
    """
    Wraps standard or streaming calls to the Anthropic API.
    If client is None, operates in simulated offline mode.
    """
    trace_id = str(uuid.uuid4())
    input_str = get_text_from_messages(messages)
    if system:
        input_str = f"[System: {system}]\n" + input_str
        
    inherited_parent_id = parent_id or active_parent_span_id.get()
    start_time = time.perf_counter()
    
    # Offline Mock Mode
    if client is None:
        logger.info("No Anthropic client provided. Running in simulated offline mode.")
        mock_response_text, in_tok, out_tok = simulate_llm_call(model, messages, stream=stream)
        simulated_latency = 0.5 + (0.01 * out_tok)
        
        if stream:
            def mock_stream_generator():
                nonlocal start_time
                time.sleep(0.15)
                ttft_ms = (time.perf_counter() - start_time) * 1000
                
                words = mock_response_text.split(" ")
                accumulated_text = ""
                for i, word in enumerate(words):
                    time.sleep(0.02)
                    space = " " if i > 0 else ""
                    yield MockStreamChunk("content_block_delta", text=space + word)
                    accumulated_text += space + word
                
                time.sleep(0.05)
                end_time = time.perf_counter()
                latency_ms = (end_time - start_time) * 1000
                cost = calculate_usd_cost(model, in_tok, out_tok)
                
                from vantage.eval.judge import judge_response
                judge_score, judge_coherence, judge_relevance, hallucination_flag = judge_response(
                    input_str, accumulated_text, rag_context
                )
                
                log_trace(
                    trace_id=trace_id,
                    prompt_version=prompt_version,
                    model=model,
                    parent_id=inherited_parent_id,
                    span_kind=span_kind,
                    input_str=input_str,
                    output_str=accumulated_text,
                    input_tokens=in_tok,
                    output_tokens=out_tok,
                    cost=cost,
                    latency_ms=latency_ms,
                    ttft_ms=ttft_ms,
                    status="success",
                    error_type=None,
                    rag_context=rag_context,
                    judge_score=judge_score,
                    judge_coherence=judge_coherence,
                    judge_relevance=judge_relevance,
                    hallucination_flag=hallucination_flag
                )
                
            return mock_stream_generator()
        else:
            time.sleep(simulated_latency)
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            status = "success"
            error_type = None
            if is_json_mode:
                try:
                    json.loads(mock_response_text)
                except json.JSONDecodeError:
                    status = "error"
                    error_type = "malformed_json"
            
            if not mock_response_text:
                status = "error"
                error_type = "empty_response"
                
            cost = calculate_usd_cost(model, in_tok, out_tok)
            
            from vantage.eval.judge import judge_response
            judge_score, judge_coherence, judge_relevance, hallucination_flag = None, None, None, None
            if status == "success":
                judge_score, judge_coherence, judge_relevance, hallucination_flag = judge_response(
                    input_str, mock_response_text, rag_context
                )
                
            log_trace(
                trace_id=trace_id,
                prompt_version=prompt_version,
                model=model,
                parent_id=inherited_parent_id,
                span_kind=span_kind,
                input_str=input_str,
                output_str=mock_response_text,
                input_tokens=in_tok,
                output_tokens=out_tok,
                cost=cost,
                latency_ms=latency_ms,
                ttft_ms=None,
                status=status,
                error_type=error_type,
                rag_context=rag_context,
                judge_score=judge_score,
                judge_coherence=judge_coherence,
                judge_relevance=judge_relevance,
                hallucination_flag=hallucination_flag
            )
            return MockMessageResponse(trace_id, model, mock_response_text, in_tok, out_tok)

    # Real SDK Integration Mode
    try:
        if stream:
            create_kwargs = dict(model=model, messages=messages, stream=True, **kwargs)
            if system is not None:
                create_kwargs["system"] = system
            response_generator = client.messages.create(**create_kwargs)
            
            def real_stream_generator():
                nonlocal start_time
                ttft_ms = None
                accumulated_text = ""
                input_tok = 0
                output_tok = 0
                
                try:
                    for chunk in response_generator:
                        chunk_type = getattr(chunk, "type", None)
                        
                        if chunk_type == "content_block_delta":
                            if ttft_ms is None:
                                ttft_ms = (time.perf_counter() - start_time) * 1000
                            
                            delta_text = ""
                            if hasattr(chunk, "delta") and hasattr(chunk.delta, "text"):
                                delta_text = chunk.delta.text
                            yield chunk
                            accumulated_text += delta_text
                            
                        elif chunk_type == "message_start":
                            message = getattr(chunk, "message", None)
                            if message and hasattr(message, "usage"):
                                input_tok = getattr(message.usage, "input_tokens", 0)
                                
                        elif chunk_type == "message_delta":
                            usage = getattr(chunk, "usage", None)
                            if usage:
                                output_tok = getattr(usage, "output_tokens", 0)
                        else:
                            yield chunk
                            
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    cost = calculate_usd_cost(model, input_tok, output_tok)
                    
                    status = "success"
                    error_type = None
                    if is_json_mode:
                        try:
                            json.loads(accumulated_text)
                        except json.JSONDecodeError:
                            status = "error"
                            error_type = "malformed_json"
                    
                    if not accumulated_text:
                        status = "error"
                        error_type = "empty_response"
                        
                    from vantage.eval.judge import judge_response
                    judge_score, judge_coherence, judge_relevance, hallucination_flag = None, None, None, None
                    if status == "success":
                        judge_score, judge_coherence, judge_relevance, hallucination_flag = judge_response(
                            input_str, accumulated_text, rag_context
                        )
                        
                    log_trace(
                        trace_id=trace_id,
                        prompt_version=prompt_version,
                        model=model,
                        parent_id=inherited_parent_id,
                        span_kind=span_kind,
                        input_str=input_str,
                        output_str=accumulated_text,
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        cost=cost,
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        status=status,
                        error_type=error_type,
                        rag_context=rag_context,
                        judge_score=judge_score,
                        judge_coherence=judge_coherence,
                        judge_relevance=judge_relevance,
                        hallucination_flag=hallucination_flag
                    )
                    
                except Exception as stream_exc:
                    latency_ms = (time.perf_counter() - start_time) * 1000
                    err_type = classify_exception(stream_exc)
                    log_trace(
                        trace_id=trace_id,
                        prompt_version=prompt_version,
                        model=model,
                        parent_id=inherited_parent_id,
                        span_kind=span_kind,
                        input_str=input_str,
                        output_str=accumulated_text if accumulated_text else None,
                        input_tokens=input_tok,
                        output_tokens=output_tok,
                        cost=calculate_usd_cost(model, input_tok, output_tok),
                        latency_ms=latency_ms,
                        ttft_ms=ttft_ms,
                        status="error",
                        error_type=err_type,
                        rag_context=rag_context
                    )
                    raise stream_exc
                    
            return real_stream_generator()
            
        else:
            create_kwargs = dict(model=model, messages=messages, **kwargs)
            if system is not None:
                create_kwargs["system"] = system
            response = client.messages.create(**create_kwargs)
            
            latency_ms = (time.perf_counter() - start_time) * 1000
            
            content_text = ""
            if hasattr(response, "content") and response.content:
                content_text = response.content[0].text
                
            input_tok = 0
            output_tok = 0
            if hasattr(response, "usage") and response.usage:
                input_tok = getattr(response.usage, "input_tokens", 0)
                output_tok = getattr(response.usage, "output_tokens", 0)
                
            status = "success"
            error_type = None
            
            if is_json_mode:
                try:
                    json.loads(content_text)
                except json.JSONDecodeError:
                    status = "error"
                    error_type = "malformed_json"
                    
            if not content_text:
                status = "error"
                error_type = "empty_response"
                
            cost = calculate_usd_cost(model, input_tok, output_tok)
            
            from vantage.eval.judge import judge_response
            judge_score, judge_coherence, judge_relevance, hallucination_flag = None, None, None, None
            if status == "success":
                judge_score, judge_coherence, judge_relevance, hallucination_flag = judge_response(
                    input_str, content_text, rag_context
                )
                
            log_trace(
                trace_id=trace_id,
                prompt_version=prompt_version,
                model=model,
                parent_id=inherited_parent_id,
                span_kind=span_kind,
                input_str=input_str,
                output_str=content_text,
                input_tokens=input_tok,
                output_tokens=output_tok,
                cost=cost,
                latency_ms=latency_ms,
                ttft_ms=None,
                status=status,
                error_type=error_type,
                rag_context=rag_context,
                judge_score=judge_score,
                judge_coherence=judge_coherence,
                judge_relevance=judge_relevance,
                hallucination_flag=hallucination_flag
            )
            return response
            
    except Exception as exc:
        if stream:
            raise exc
            
        latency_ms = (time.perf_counter() - start_time) * 1000
        err_type = classify_exception(exc)
        
        log_trace(
            trace_id=trace_id,
            prompt_version=prompt_version,
            model=model,
            parent_id=inherited_parent_id,
            span_kind=span_kind,
            input_str=input_str,
            output_str=None,
            input_tokens=0,
            output_tokens=0,
            cost=0.0,
            latency_ms=latency_ms,
            ttft_ms=None,
            status="error",
            error_type=err_type,
            rag_context=rag_context
        )
        raise exc
