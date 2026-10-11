"""Evaluation of traced LLM responses. Runs on a background thread after each call has returned.

- Grounding: when the call had RAG context, every sentence of the response is checked against it by
  a local NLI model (``vantage.local.nli``, needs the ``local`` extra). This sets the hallucination flag.
- Quality scores (coherence, relevance and overall, 1-5): an LLM judge on Claude, only with
  LLM_PROVIDER=claude, so local mode never makes paid calls. Otherwise they stay empty; there is no
  stand-in score.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

from vantage.core.settings import settings
from vantage.local.runtime import local_models_installed

logger = logging.getLogger("vantage.eval.judge")

JUDGE_SYSTEM = (
    "You are an AI judge. Evaluate the quality of the LLM response to the user's prompt, "
    "and potentially evaluate it against the retrieved RAG context.\n"
    "Provide score values from 1 (poor) to 5 (excellent) for 'coherence' and 'relevance'.\n"
    "If RAG context is provided, assess if the response states facts that contradict the context "
    "or contains assertions that cannot be substantiated by the context. Set 'hallucinated' to true "
    "if it is a hallucination/contradiction, else false. If no RAG context is provided, 'hallucinated' MUST be false."
)

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "coherence": {"type": "integer"},
        "relevance": {"type": "integer"},
        "hallucinated": {"type": "boolean"},
    },
    "required": ["coherence", "relevance", "hallucinated"],
    "additionalProperties": False,
}


@dataclass
class Judgement:
    score: int | None = None
    coherence: int | None = None
    relevance: int | None = None
    hallucinated: bool | None = None


def judge_response(prompt: str, response: str, rag_context: str | None = None) -> Judgement:
    judgement = Judgement()
    if not response:
        return judgement

    if rag_context and local_models_installed():
        try:
            from vantage.local.nli import load_grounding_checker

            checker = load_grounding_checker(settings.local_nli_model, settings.local_device)
            judgement.hallucinated = checker.check(response, rag_context).hallucinated
        except Exception:
            logger.warning("Grounding check failed", exc_info=True)

    if settings.llm_provider == "claude" and settings.anthropic_api_key:
        scores = _claude_judge(prompt, response, rag_context)
        if scores:
            judgement.coherence = max(1, min(5, int(scores["coherence"])))
            judgement.relevance = max(1, min(5, int(scores["relevance"])))
            judgement.score = round((judgement.coherence + judgement.relevance) / 2)
            if judgement.hallucinated is None and rag_context:
                judgement.hallucinated = bool(scores["hallucinated"])
    return judgement


_client = None


def _claude_client():
    global _client
    if _client is None:
        import anthropic

        _client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
    return _client


def _claude_judge(prompt: str, response: str, rag_context: str | None) -> dict | None:
    details = f"Prompt:\n{prompt}\n\nResponse:\n{response}"
    if rag_context:
        details = f"RAG Context:\n{rag_context}\n\n" + details
    try:
        res = _claude_client().messages.create(
            model=settings.judge_model,
            max_tokens=2048,
            system=JUDGE_SYSTEM,
            messages=[{"role": "user", "content": details}],
            output_config={"effort": "low", "format": {"type": "json_schema", "schema": JUDGE_SCHEMA}},
        )
        if res.stop_reason == "refusal":
            return None
        # Read the text block: the first block can be a thinking block.
        text = next((block.text for block in res.content if block.type == "text"), None)
        return json.loads(text) if text else None
    except Exception:
        logger.warning("Claude judge call failed", exc_info=True)
        return None
