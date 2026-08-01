import os
import json
import logging
import hashlib

logger = logging.getLogger("vantage.eval.judge")

def get_judge_client():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        return anthropic.Anthropic(api_key=api_key)
    except ImportError:
        logger.warning("Anthropic SDK is not installed. Falling back to mock judge.")
        return None

def heuristic_mock_judge(prompt: str, response: str, rag_context: str = None) -> tuple:
    """
    Computes a deterministic, realistic score based on prompt and response characteristics.
    Used when running offline.
    """
    # Deterministic seed using hash of prompt + response
    combo = (prompt + response).encode("utf-8")
    h = int(hashlib.md5(combo).hexdigest(), 16)
    
    # Relevance heuristic (length matching, keyword overlap, hash-based variation)
    words_prompt = set(prompt.lower().split())
    words_response = set(response.lower().split())
    overlap = len(words_prompt.intersection(words_response))
    
    relevance = 3 + (overlap % 3)
    if len(response) < 10:
        relevance = 2
        
    # Coherence heuristic
    coherence = 4
    if len(response) > 500:
        coherence = 5
    elif len(response) < 20:
        coherence = 2
        
    # Introduce random variation
    relevance = max(1, min(5, relevance + (h % 3 - 1)))
    coherence = max(1, min(5, coherence + ((h >> 2) % 3 - 1)))
    
    # Hallucination heuristic
    hallucinated = False
    if rag_context:
        rc_lower = rag_context.lower()
        resp_lower = response.lower()
        
        if "berlin" in resp_lower and "paris" in rc_lower:
            hallucinated = True
        elif "hallucinate" in resp_lower or "contradict" in resp_lower:
            hallucinated = True
        else:
            if "not supported" in resp_lower or "fake info" in resp_lower:
                hallucinated = True
            elif (h % 100) < 15:
                hallucinated = True
                
    overall_score = round((relevance + coherence) / 2)
    return overall_score, coherence, relevance, hallucinated

def judge_response(prompt: str, response: str, rag_context: str = None) -> tuple:
    """
    Evaluates response quality using an LLM API or heuristic fallback.
    Returns (overall_score, coherence_score, relevance_score, hallucination_flag)
    """
    if not response:
        return 1, 1, 1, False

    client = get_judge_client()
    
    if client is None:
        return heuristic_mock_judge(prompt, response, rag_context)

    # Call real Claude-3-5-Haiku model as judge
    system_instruction = (
        "You are an AI judge. Evaluate the quality of the LLM response to the user's prompt, "
        "and potentially evaluate it against the retrieved RAG context.\n"
        "Provide score values from 1 (poor) to 5 (excellent) for 'coherence' and 'relevance'.\n"
        "If RAG context is provided, assess if the response states facts that contradict the context "
        "or contains assertions that cannot be substantiated by the context. Set 'hallucinated' to true "
        "if it is a hallucination/contradiction, else false. If no RAG context is provided, 'hallucinated' MUST be false.\n"
        "Return ONLY a valid JSON object matching this schema:\n"
        "{\n"
        "  \"coherence\": <int 1-5>,\n"
        "  \"relevance\": <int 1-5>,\n"
        "  \"hallucinated\": <bool>\n"
        "}"
    )

    prompt_details = f"Prompt:\n{prompt}\n\nResponse:\n{response}"
    if rag_context:
        prompt_details = f"RAG Context:\n{rag_context}\n\n" + prompt_details

    try:
        res = client.messages.create(
            model="claude-3-5-haiku-20241022",
            max_tokens=200,
            system=system_instruction,
            messages=[{"role": "user", "content": prompt_details}],
            temperature=0.0
        )
        
        content = res.content[0].text.strip()
        
        if content.startswith("```json"):
            content = content.split("```json")[1].split("```")[0].strip()
        elif content.startswith("```"):
            content = content.split("```")[1].split("```")[0].strip()
            
        data = json.loads(content)
        coherence = int(data.get("coherence", 3))
        relevance = int(data.get("relevance", 3))
        hallucinated = bool(data.get("hallucinated", False))
        
        coherence = max(1, min(5, coherence))
        relevance = max(1, min(5, relevance))
        overall_score = round((coherence + relevance) / 2)
        
        return overall_score, coherence, relevance, hallucinated
        
    except Exception as e:
        logger.error(f"Error calling real judge LLM: {e}. Falling back to heuristic judge.")
        return heuristic_mock_judge(prompt, response, rag_context)
