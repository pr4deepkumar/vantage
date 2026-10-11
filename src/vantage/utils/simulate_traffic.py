"""Synthetic demo traffic: a week of agent runs and direct LLM calls, so the dashboard has history
(drift detection needs at least 60 calls) before any real call is traced. Everything here is made
up: latencies, token counts and judge scores are drawn at random, and responses come from fixed
templates, some of them deliberately wrong ("hallucinated")."""

import random
import uuid
from datetime import datetime, timedelta, timezone

from vantage.core.db import init_db, save_trace
from vantage.core.instrumentation import calculate_usd_cost

# Each response is (text, faithful): faithful responses agree with the context; the others contradict it.
TEMPLATES = [
    {
        "prompt": "Summarize the key achievements of space exploration in 2025.",
        "context": "Space exploration in 2025 saw the launch of the Artemis IV mission, which placed a new habitat module in lunar orbit. Private firms successfully landed cargo on Mars, and the James Webb telescope discovered three new temperate exoplanets.",
        "responses": [
            (
                "In 2025, space exploration achieved major milestones including the lunar Artemis IV orbit habitat launch, Mars cargo landings by private corporations, and the identification of three temperate exoplanets by the James Webb Space Telescope.",
                True,
            ),
            (
                "The year 2025 was historic for space: Artemis IV module reached lunar orbit, commercial firms landed cargo on Mars, and James Webb discovered three temperate exoplanets.",
                True,
            ),
            (
                "In 2025, space exploration was highlighted by astronauts landing on Mars for the first time, while the James Webb telescope was decommissioned and crashed into the ocean.",
                False,
            ),
        ],
    },
    {
        "prompt": "Write a Python function to compute the Fibonacci sequence using memoization.",
        "context": None,
        "responses": [
            (
                "Here is the memoized Fibonacci function in Python:\n\n```python\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n```\nThis implementation has a time complexity of O(n) and space complexity of O(n).",
                True,
            ),
            (
                "Below is a memoized Fibonacci implementation:\n\n```python\nclass Fib:\n    def __init__(self):\n        self.cache = {}\n    def get(self, n):\n        if n <= 1: return n\n        if n not in self.cache:\n            self.cache[n] = self.get(n-1) + self.get(n-2)\n        return self.cache[n]\n```",
                True,
            ),
        ],
    },
    {
        "prompt": "What are the primary symptoms of dehydration?",
        "context": "Dehydration symptoms range from mild to severe. Common symptoms include extreme thirst, dry mouth, less frequent urination, dark-colored urine, fatigue, dizziness, and confusion.",
        "responses": [
            (
                "Common symptoms of dehydration include dry mouth, fatigue, dizziness, dark urine, decreased urination frequency, and intense thirst.",
                True,
            ),
            (
                "Dehydration manifests primarily as dry mouth, fatigue, dark urine, extreme thirst, and dizziness. Severe cases can cause confusion.",
                True,
            ),
            (
                "Dehydration primarily causes high fever, chest pain, coughing up blood, and sudden weight gain.",
                False,
            ),
        ],
    },
    {
        "prompt": "Translate 'Could you please point me to the nearest train station?' to German.",
        "context": None,
        "responses": [
            ("In German, you would say: 'Könnten Sie mir bitte den Weg zum nächsten Bahnhof zeigen?'", True),
            (
                "A polite German translation is: 'Könnten Sie mir bitte sagen, wo der nächste Bahnhof ist?'",
                True,
            ),
        ],
    },
    {
        "prompt": "Create a JSON schema for a user profile with name, email, and age.",
        "context": None,
        "responses": [
            (
                '{\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "type": "object",\n  "properties": {\n    "name": { "type": "string" },\n    "email": { "type": "string", "format": "email" },\n    "age": { "type": "integer", "minimum": 0 }\n  },\n  "required": ["name", "email"]\n}',
                True,
            ),
            (
                '{\n  "title": "User",\n  "type": "object",\n  "properties": {\n    "name": {"type": "string"},\n    "email": {"type": "string"},\n    "age": {"type": "integer"}\n  }\n}',
                True,
            ),
        ],
    },
]

MODELS = ["claude-sonnet-5-5", "claude-haiku-5-5"]
BASE_LATENCY_MS = {"claude-sonnet-5-5": 1200.0, "claude-haiku-5-5": 450.0}


def _judge_scores(faithful: bool, has_context: bool) -> dict:
    """Made-up judge results consistent with the response's label."""
    coherence = random.choice([4, 5]) if faithful else random.choice([3, 4])
    relevance = random.choice([4, 5]) if faithful else random.choice([2, 3])
    return {
        "judge_score": round((coherence + relevance) / 2),
        "judge_coherence": coherence,
        "judge_relevance": relevance,
        "hallucination_flag": (not faithful) if has_context else None,
    }


def _span(**fields) -> dict:
    base = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cost": 0.0,
        "ttft_ms": None,
        "error_type": None,
        "stop_reason": None,
        "user_feedback": None,
        "rag_context": None,
    }
    return base | fields


def _feedback(hallucinated: bool | None, down_rate: float) -> str | None:
    r = random.random()
    if r < 0.15:
        return "up"
    if r < 0.15 + down_rate or (hallucinated and r < 0.70):
        return "down"
    return None


def generate_simulated_data():
    print("Initializing database...")
    init_db()

    print("Generating ~200 simulated traces with parent-child relationships...")

    current_time = datetime.now(timezone.utc) - timedelta(days=7)
    time_increment = timedelta(days=7) / 100
    total_generated = 0

    # 1. 40 agent runs (120 spans): agent -> vector search tool -> LLM call
    for i in range(40):
        prompt_version = "v1.0.0" if i < 20 else "v1.1.0"
        is_drifted = i >= 30

        template = random.choice(TEMPLATES)
        prompt = template["prompt"]
        rag_context = template["context"]

        is_error = random.random() < 0.05
        error_type = random.choice(["timeout", "rate_limit", "other"]) if is_error else None
        status = "error" if is_error else "success"
        model = random.choice(MODELS)

        tool_latency = random.uniform(40, 100)
        if is_error:
            llm_latency = 10000.0 if error_type == "timeout" else random.uniform(50, 200)
            response_text, faithful = None, True
            in_tokens, out_tokens = 0, 0
        else:
            base_latency = BASE_LATENCY_MS[model] * (2.5 if is_drifted else 1.0)
            llm_latency = max(100.0, random.normalvariate(base_latency, 150.0))
            response_text, faithful = random.choice(template["responses"])
            in_tokens = len(prompt) // 4 + random.randint(10, 30)
            out_tokens = len(response_text) // 4 + random.randint(15, 45)
            if is_drifted:
                in_tokens = int(in_tokens * 2.2)
                out_tokens = int(out_tokens * 2.0)

        # Children start inside the agent span: retrieval first, then the LLM call.
        tool_start = current_time + timedelta(milliseconds=random.uniform(2, 8))
        llm_start = tool_start + timedelta(milliseconds=tool_latency + random.uniform(2, 8))
        total_latency = (
            (llm_start - current_time).total_seconds() * 1000 + llm_latency + random.uniform(5, 15)
        )

        agent_span_id = str(uuid.uuid4())
        save_trace(
            _span(
                id=agent_span_id,
                timestamp=current_time,
                prompt_version=prompt_version,
                model="orchestrator",
                parent_id=None,
                span_kind="agent",
                input=f"Execute Agent Task: {prompt}",
                output=f"Agent completed workflow with status {status}"
                if status == "success"
                else f"Agent crashed: {error_type}",
                latency_ms=total_latency,
                status=status,
                error_type=error_type,
            )
        )
        save_trace(
            _span(
                id=str(uuid.uuid4()),
                timestamp=tool_start,
                prompt_version=prompt_version,
                model="system",
                parent_id=agent_span_id,
                span_kind="tool",
                input=f"Query vector index: {prompt[:30]}",
                output=f"Retrieved Context: {(rag_context or 'no documents found')[:60]}...",
                latency_ms=tool_latency,
                status="success",
            )
        )
        judgement = _judge_scores(faithful, rag_context is not None) if status == "success" else {}
        save_trace(
            _span(
                id=str(uuid.uuid4()),
                timestamp=llm_start,
                prompt_version=prompt_version,
                model=model,
                parent_id=agent_span_id,
                span_kind="llm",
                input=prompt,
                output=response_text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost=calculate_usd_cost(model, in_tokens, out_tokens) if status == "success" else 0.0,
                latency_ms=llm_latency,
                ttft_ms=random.uniform(150, 300) if status == "success" else None,
                status=status,
                error_type=error_type,
                stop_reason="end_turn" if status == "success" else None,
                user_feedback=_feedback(judgement.get("hallucination_flag"), 0.05)
                if status == "success"
                else None,
                rag_context=rag_context,
                **judgement,
            )
        )

        current_time += time_increment
        total_generated += 3

    # 2. 80 direct LLM calls (80 spans), not streamed
    for i in range(80):
        prompt_version = "v1.0.0" if i < 40 else "v1.1.0"
        is_drifted = i >= 60

        template = random.choice(TEMPLATES)
        prompt = template["prompt"]

        is_error = random.random() < 0.08
        error_type = (
            random.choice(["timeout", "rate_limit", "malformed_json", "empty_response", "other"])
            if is_error
            else None
        )
        status = "error" if is_error else "success"
        model = random.choice(MODELS)

        if is_error:
            latency_ms = 10000.0 if error_type == "timeout" else random.uniform(50, 200)
            response_text = "Invalid JSON output { ... }" if error_type == "malformed_json" else None
            in_tokens, out_tokens = 0, 0
        else:
            base_latency = BASE_LATENCY_MS[model] * (2.4 if is_drifted else 1.0)
            latency_ms = max(100.0, random.normalvariate(base_latency, 120.0))
            response_text = random.choice([text for text, faithful in template["responses"] if faithful])
            in_tokens = len(prompt) // 4 + random.randint(10, 30)
            out_tokens = len(response_text) // 4 + random.randint(15, 45)
            if is_drifted:
                in_tokens = int(in_tokens * 2.3)
                out_tokens = int(out_tokens * 2.1)

        save_trace(
            _span(
                id=str(uuid.uuid4()),
                timestamp=current_time,
                prompt_version=prompt_version,
                model=model,
                parent_id=None,
                span_kind="llm",
                input=prompt,
                output=response_text,
                input_tokens=in_tokens,
                output_tokens=out_tokens,
                cost=calculate_usd_cost(model, in_tokens, out_tokens) if status == "success" else 0.0,
                latency_ms=latency_ms,
                status=status,
                error_type=error_type,
                stop_reason="end_turn" if status == "success" else None,
                user_feedback=_feedback(None, 0.03) if status == "success" else None,
                **(_judge_scores(True, False) if status == "success" else {}),
            )
        )

        current_time += time_increment / 2
        total_generated += 1

    print(f"Successfully generated {total_generated} trace spans in the database.")


if __name__ == "__main__":
    generate_simulated_data()
