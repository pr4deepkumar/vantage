import random
import time
from datetime import datetime, timedelta
import uuid

from vantage.core.db import init_db, save_trace
from vantage.core.instrumentation import calculate_usd_cost, trace_span, observe_llm_call
from vantage.eval.judge import heuristic_mock_judge

# Realistic templates for simulated requests
TEMPLATES = [
    {
        "prompt": "Summarize the key achievements of space exploration in 2025.",
        "context": "Space exploration in 2025 saw the launch of the Artemis IV mission, which placed a new habitat module in lunar orbit. Private firms successfully landed cargo on Mars, and the James Webb telescope discovered three new temperate exoplanets.",
        "responses": [
            "In 2025, space exploration achieved major milestones including the lunar Artemis IV orbit habitat launch, Mars cargo landings by private corporations, and the identification of three temperate exoplanets by the James Webb Space Telescope.",
            "The year 2025 was historic for space: Artemis IV module reached lunar orbit, commercial firms landed cargo on Mars, and James Webb discovered three temperate exoplanets.",
            "In 2025, space exploration was highlighted by astronauts landing on Mars for the first time, while the James Webb telescope was decommissioned and crashed into the ocean."
        ]
    },
    {
        "prompt": "Write a Python function to compute the Fibonacci sequence using memoization.",
        "context": None,
        "responses": [
            "Here is the memoized Fibonacci function in Python:\n\n```python\ndef fib(n, memo={}):\n    if n in memo: return memo[n]\n    if n <= 1: return n\n    memo[n] = fib(n-1, memo) + fib(n-2, memo)\n    return memo[n]\n```\nThis implementation has a time complexity of O(n) and space complexity of O(n).",
            "Below is a memoized Fibonacci implementation:\n\n```python\nclass Fib:\n    def __init__(self):\n        self.cache = {}\n    def get(self, n):\n        if n <= 1: return n\n        if n not in self.cache:\n            self.cache[n] = self.get(n-1) + self.get(n-2)\n        return self.cache[n]\n```"
        ]
    },
    {
        "prompt": "What are the primary symptoms of dehydration?",
        "context": "Dehydration symptoms range from mild to severe. Common symptoms include extreme thirst, dry mouth, less frequent urination, dark-colored urine, fatigue, dizziness, and confusion.",
        "responses": [
            "Common symptoms of dehydration include dry mouth, fatigue, dizziness, dark urine, decreased urination frequency, and intense thirst.",
            "Dehydration manifests primarily as dry mouth, fatigue, dark urine, extreme thirst, and dizziness. Severe cases can cause confusion.",
            "Dehydration primarily causes high fever, chest pain, coughing up blood, and sudden weight gain."
        ]
    },
    {
        "prompt": "Translate 'Could you please point me to the nearest train station?' to German.",
        "context": None,
        "responses": [
            "In German, you would say: 'Könnten Sie mir bitte den Weg zum nächsten Bahnhof zeigen?'",
            "A polite German translation is: 'Könnten Sie mir bitte sagen, wo der nächste Bahnhof ist?'"
        ]
    },
    {
        "prompt": "Create a JSON schema for a user profile with name, email, and age.",
        "context": None,
        "responses": [
            '{\n  "$schema": "http://json-schema.org/draft-07/schema#",\n  "type": "object",\n  "properties": {\n    "name": { "type": "string" },\n    "email": { "type": "string", "format": "email" },\n    "age": { "type": "integer", "minimum": 0 }\n  },\n  "required": ["name", "email"]\n}',
            '{\n  "title": "User",\n  "type": "object",\n  "properties": {\n    "name": {"type": "string"},\n    "email": {"type": "string"},\n    "age": {"type": "integer"}\n  }\n}'
        ]
    }
]

def generate_simulated_data():
    print("Initializing database...")
    init_db()
    
    print("Generating ~200 simulated traces with parent-child relationships...")
    
    start_date = datetime.utcnow() - timedelta(days=7)
    time_increment = timedelta(days=7) / 100
    
    total_generated = 0
    current_time = start_date
    
    def get_timestamp(offset_seconds=0):
        nonlocal current_time
        return current_time + timedelta(seconds=offset_seconds)

    # 1. Generate 40 nested Agent runs (120 traces)
    for i in range(40):
        prompt_version = "v1.0.0" if i < 20 else "v1.1.0"
        is_drifted = (i >= 30)
        
        template = random.choice(TEMPLATES)
        prompt = template["prompt"]
        rag_context = template["context"] or "Default pipeline facts."
        
        is_error = (random.random() < 0.05)
        error_type = random.choice(["timeout", "rate_limit", "other"]) if is_error else None
        status = "error" if is_error else "success"
        
        model = random.choice(["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"])
        
        agent_span_id = str(uuid.uuid4())
        tool_span_id = str(uuid.uuid4())
        llm_span_id = str(uuid.uuid4())
        
        tool_latency = random.uniform(40, 100)
        
        if is_error:
            llm_latency = 10000.0 if error_type == "timeout" else random.uniform(50, 200)
            response_text = None
            in_tokens, out_tokens = 0, 0
        else:
            base_latency = 1200.0 if model == "claude-3-5-sonnet-latest" else 450.0
            if is_drifted:
                base_latency *= 2.5
            llm_latency = max(100.0, random.normalvariate(base_latency, 150.0))
            
            response_text = random.choice(template["responses"])
            in_tokens = len(prompt) // 4 + random.randint(10, 30)
            out_tokens = len(response_text) // 4 + random.randint(15, 45)
            if is_drifted:
                in_tokens = int(in_tokens * 2.2)
                out_tokens = int(out_tokens * 2.0)
                
        total_latency = tool_latency + llm_latency + random.uniform(10, 30)
        cost = calculate_usd_cost(model, in_tokens, out_tokens) if status == "success" else 0.0
        
        save_trace({
            "id": agent_span_id,
            "timestamp": get_timestamp(),
            "prompt_version": prompt_version,
            "model": "orchestrator",
            "parent_id": None,
            "span_kind": "agent",
            "input": f"Execute Agent Task: {prompt}",
            "output": f"Agent completed workflow with status {status}" if status == "success" else f"Agent crashed: {error_type}",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": cost,
            "latency_ms": total_latency,
            "ttft_ms": None,
            "status": status,
            "error_type": error_type,
            "user_feedback": None,
            "judge_score": None,
            "judge_coherence": None,
            "judge_relevance": None,
            "hallucination_flag": None,
            "rag_context": None
        })
        
        save_trace({
            "id": tool_span_id,
            "timestamp": get_timestamp(1),
            "prompt_version": prompt_version,
            "model": "system",
            "parent_id": agent_span_id,
            "span_kind": "tool",
            "input": f"Query vector index: {prompt[:30]}",
            "output": f"Retrieved Context: {rag_context[:60]}...",
            "input_tokens": 0,
            "output_tokens": 0,
            "cost": 0.0,
            "latency_ms": tool_latency,
            "ttft_ms": None,
            "status": "success",
            "error_type": None,
            "user_feedback": None,
            "judge_score": None,
            "judge_coherence": None,
            "judge_relevance": None,
            "hallucination_flag": None,
            "rag_context": None
        })
        
        judge_score, coherence, relevance, hallucinated = None, None, None, None
        if status == "success":
            judge_score, coherence, relevance, hallucinated = heuristic_mock_judge(prompt, response_text, rag_context)
            
        fb_rand = random.random()
        user_feedback = None
        if status == "success":
            if fb_rand < 0.15:
                user_feedback = "up"
            elif fb_rand < 0.20 or (hallucinated and fb_rand < 0.70):
                user_feedback = "down"

        save_trace({
            "id": llm_span_id,
            "timestamp": get_timestamp(2),
            "prompt_version": prompt_version,
            "model": model,
            "parent_id": agent_span_id,
            "span_kind": "llm",
            "input": prompt,
            "output": response_text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost": cost,
            "latency_ms": llm_latency,
            "ttft_ms": random.uniform(150, 300) if status == "success" else None,
            "status": status,
            "error_type": error_type,
            "user_feedback": user_feedback,
            "judge_score": judge_score,
            "judge_coherence": coherence,
            "judge_relevance": relevance,
            "hallucination_flag": hallucinated,
            "rag_context": rag_context
        })
        
        current_time += time_increment
        total_generated += 3

    # 2. Generate 80 Flat LLM Calls (80 traces)
    for i in range(80):
        prompt_version = "v1.0.0" if i < 40 else "v1.1.0"
        is_drifted = (i >= 60)
        
        template = random.choice(TEMPLATES)
        prompt = template["prompt"]
        
        is_error = (random.random() < 0.08)
        error_type = random.choice(["timeout", "rate_limit", "malformed_json", "empty_response", "other"]) if is_error else None
        status = "error" if is_error else "success"
        
        model = random.choice(["claude-3-5-sonnet-latest", "claude-3-5-haiku-latest"])
        
        if is_error:
            latency_ms = 10000.0 if error_type == "timeout" else random.uniform(50, 200)
            response_text = None
            in_tokens, out_tokens = 0, 0
            if error_type == "malformed_json":
                response_text = "Invalid JSON output { ... }"
                status = "error"
        else:
            base_latency = 1200.0 if model == "claude-3-5-sonnet-latest" else 450.0
            if is_drifted:
                base_latency *= 2.4
            latency_ms = max(100.0, random.normalvariate(base_latency, 120.0))
            
            response_text = random.choice(template["responses"][:-1])
            in_tokens = len(prompt) // 4 + random.randint(10, 30)
            out_tokens = len(response_text) // 4 + random.randint(15, 45)
            if is_drifted:
                in_tokens = int(in_tokens * 2.3)
                out_tokens = int(out_tokens * 2.1)
                
        cost = calculate_usd_cost(model, in_tokens, out_tokens) if status == "success" else 0.0
        
        judge_score, coherence, relevance, hallucinated = None, None, None, None
        if status == "success":
            judge_score, coherence, relevance, hallucinated = heuristic_mock_judge(prompt, response_text, None)
            
        fb_rand = random.random()
        user_feedback = None
        if status == "success":
            if fb_rand < 0.15:
                user_feedback = "up"
            elif fb_rand < 0.18:
                user_feedback = "down"

        save_trace({
            "id": str(uuid.uuid4()),
            "timestamp": current_time,
            "prompt_version": prompt_version,
            "model": model,
            "parent_id": None,
            "span_kind": "llm",
            "input": prompt,
            "output": response_text,
            "input_tokens": in_tokens,
            "output_tokens": out_tokens,
            "cost": cost,
            "latency_ms": latency_ms,
            "ttft_ms": None,
            "status": status,
            "error_type": error_type,
            "user_feedback": user_feedback,
            "judge_score": judge_score,
            "judge_coherence": coherence,
            "judge_relevance": relevance,
            "hallucination_flag": hallucinated,
            "rag_context": None
        })
        
        current_time += (time_increment / 2)
        total_generated += 1
        
    print(f"Successfully generated {total_generated} trace spans in the database.")

if __name__ == "__main__":
    generate_simulated_data()
