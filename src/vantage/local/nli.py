"""Grounding check for RAG responses with an open-source NLI model, run locally: no API key, no cost.

The response is split into sentences by rule. Each sentence is checked against the RAG context by a
three-way natural language inference (NLI) cross-encoder, against the whole context and against
overlapping two-sentence windows of it (NLI models are trained mostly on short premises and read
long ones less reliably; the approach of SummaC, Laban et al., 2022):

- supported: some premise entails the sentence;
- partially supported: no premise entails or contradicts it, but one shares most of its content
  words, so it discusses the point without entailing all of it;
- contradicted: no premise entails it and some premise contradicts it;
- unsupported: anything else.

A response counts as a hallucination when any sentence is contradicted or unsupported. Sentences that
only say what the context lacks ("The context doesn't mention ...") make no claim and are skipped.

Adapted from the VERITAS claim verifier. The default model, MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli
(MIT licence), is downloaded from the Hugging Face Hub on first use. Requires the ``local`` extra.
"""

from __future__ import annotations

import logging
import re
import threading
from collections.abc import Callable
from dataclasses import dataclass

from vantage.local.runtime import LocalModelError, device_lock, import_torch, load_failure, pick_device

logger = logging.getLogger(__name__)

# (premise, hypothesis) pairs -> one {"entailment": p, "neutral": p, "contradiction": p} each.
Classifier = Callable[[list[tuple[str, str]]], list[dict[str, float]]]

NLI_LABELS = {"entailment", "neutral", "contradiction"}
FLAGGED = {"contradicted", "unsupported"}
MIN_CLAIM_WORDS = 3
# Share of a sentence's content words found in a premise at which a neutral verdict counts as
# partially supported.
PARTIAL_OVERLAP = 0.5
WINDOW_SENTENCES = 2
MAX_WINDOWS = 24

_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])|\n+")
_LIST_BULLET = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")
_MARKUP = re.compile(r"[*`]+|^#+\s*")
_NOT = r"(?:\s+not|n['\N{RIGHT SINGLE QUOTATION MARK}]t)\b"
# Sentences that only describe what the context lacks. Only reporting verbs count: "The warranty does
# not cover water damage." is a claim about the warranty, not about the sources.
_META_SENTENCE = re.compile(
    rf"\b(?:do|does|did){_NOT}\s+(?:\w+\s+)?"
    r"(?:say|state|mention|specify|report|disclose|discuss|describe|explain|indicate|clarify|detail)\b"
    rf"|\b(?:is|are|was|were){_NOT}\s+(?:\w+\s+)?"
    r"(?:mentioned|stated|specified|reported|disclosed|discussed|described|explained|indicated)\b"
    rf"|\b(?:sources?|documents?|passages?|evidence|context)\b[^.]*\b(?:do|does|did){_NOT}"
    r"|\bno (?:information|evidence|details?|data|mention)\b"
    rf"|\bI do{_NOT}\s+know\b",
    re.IGNORECASE,
)
_WORD = re.compile(r"[^\W_]+")
_PREMISE_SENTENCE = re.compile(r"(?<=[.!?])\s+|\n+")
_STOPWORDS = frozenset(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "their",
        "this",
        "to",
        "was",
        "were",
        "will",
        "with",
    ]
)


@dataclass(frozen=True)
class ClaimCheck:
    text: str
    verdict: str  # supported, partially_supported, contradicted or unsupported
    probability: float  # of the NLI label behind the verdict


@dataclass(frozen=True)
class GroundingReport:
    claims: tuple[ClaimCheck, ...]

    @property
    def hallucinated(self) -> bool:
        return any(claim.verdict in FLAGGED for claim in self.claims)


def split_claims(response: str) -> list[str]:
    claims = []
    for raw_sentence in _SENTENCE_BREAK.split(response):
        sentence = _MARKUP.sub("", _LIST_BULLET.sub("", raw_sentence)).strip()
        sentence = re.sub(r"[.!?]+$", "", sentence).strip()
        if len(_WORD.findall(sentence)) >= MIN_CLAIM_WORDS and not _META_SENTENCE.search(sentence):
            claims.append(sentence)
    return claims


def premise_windows(text: str) -> list[str]:
    """Overlapping two-sentence windows of the context. Contexts of two sentences or fewer need none."""
    sentences = [s.strip() for s in _PREMISE_SENTENCE.split(text) if s.strip()]
    if len(sentences) <= WINDOW_SENTENCES:
        return []
    windows = [
        " ".join(sentences[i : i + WINDOW_SENTENCES]) for i in range(len(sentences) - WINDOW_SENTENCES + 1)
    ]
    return windows[:MAX_WINDOWS]


def _hypothesis(text: str) -> str:
    # A list item can lack its subject ("reduces latency by 40%").
    return text if text[:1].isupper() or text[:1].isdigit() else f"It {text}"


def _content_words(text: str) -> set[str]:
    return {word for word in _WORD.findall(text.lower()) if word not in _STOPWORDS}


def _overlap(claim: str, premise: str) -> float:
    words = _content_words(claim)
    return len(words & _content_words(premise)) / len(words) if words else 0.0


def _argmax(scores: dict[str, float]) -> str:
    return max(scores, key=scores.__getitem__)


def _decide(claim: str, premises: list[str], scores: list[dict[str, float]]) -> ClaimCheck:
    entailed = [s["entailment"] for s in scores if _argmax(s) == "entailment"]
    if entailed:
        return ClaimCheck(claim, "supported", max(entailed))
    contradicted = [s["contradiction"] for s in scores if _argmax(s) == "contradiction"]
    if contradicted:
        return ClaimCheck(claim, "contradicted", max(contradicted))
    best = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
    verdict = "partially_supported" if _overlap(claim, premises[best]) >= PARTIAL_OVERLAP else "unsupported"
    return ClaimCheck(claim, verdict, scores[best]["entailment"])


class GroundingChecker:
    def __init__(self, classifier: Classifier) -> None:
        self.classify = classifier

    def check(self, response: str, context: str) -> GroundingReport:
        claims = split_claims(response)
        premises = [context, *premise_windows(context)]
        pairs = [(premise, _hypothesis(claim)) for claim in claims for premise in premises]
        scores = self.classify(pairs) if pairs else []
        n = len(premises)
        return GroundingReport(
            tuple(_decide(claim, premises, scores[i * n : (i + 1) * n]) for i, claim in enumerate(claims))
        )


class HuggingFaceNLI:
    """A three-way NLI cross-encoder from the Hugging Face Hub, run with PyTorch."""

    def __init__(self, model_name: str, batch_size: int = 16, device: str = "auto") -> None:
        torch = import_torch()
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.device = pick_device(torch, device)
        logger.info("Loading NLI model %s on %s (downloaded on first use)", model_name, self.device)
        self._torch = torch
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        except OSError as exc:
            raise load_failure(model_name, exc) from exc
        with device_lock(self.device):  # the chat model may be running on the GPU
            self.model = model.to(self.device).eval()  # float32 on every device
        config = self.model.config
        self.labels = [config.id2label[i].lower() for i in range(config.num_labels)]
        if set(self.labels) != NLI_LABELS:
            raise LocalModelError(f"{model_name} is not a three-way NLI model (labels: {self.labels})")
        self.batch_size = batch_size

    def __call__(self, pairs: list[tuple[str, str]]) -> list[dict[str, float]]:
        # Batch pairs of similar length together so short windows aren't padded to the whole context.
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        results: list[dict[str, float]] = [{} for _ in pairs]
        with device_lock(self.device), self._torch.inference_mode():
            for start in range(0, len(order), self.batch_size):
                indices = order[start : start + self.batch_size]
                batch = [pairs[i] for i in indices]
                encoded = self.tokenizer(
                    [premise for premise, _ in batch],
                    [hypothesis for _, hypothesis in batch],
                    truncation="only_first",
                    max_length=512,
                    padding=True,
                    return_tensors="pt",
                ).to(self.device)
                probabilities = self.model(**encoded).logits.float().softmax(dim=-1).tolist()
                for i, row in zip(indices, probabilities, strict=True):
                    results[i] = dict(zip(self.labels, row, strict=True))
        return results


_cache: dict[tuple[str, str], GroundingChecker] = {}
_cache_lock = threading.Lock()


def load_grounding_checker(model_name: str, device: str = "auto") -> GroundingChecker:
    """Load (once per process) the checker backed by ``model_name``."""
    with _cache_lock:
        if (model_name, device) not in _cache:
            _cache[model_name, device] = GroundingChecker(HuggingFaceNLI(model_name, device=device))
        return _cache[model_name, device]
