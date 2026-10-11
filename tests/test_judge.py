from types import SimpleNamespace as NS

import pytest

from vantage.eval import judge
from vantage.local.nli import GroundingChecker, premise_windows, split_claims

CONTEXT = "The refund window is 14 days. Refunds go to the original card. Support answers within a day."


def scripted(labels):
    """An NLI classifier that labels a pair by the first key found in its hypothesis (default: neutral)."""

    def classify(pairs):
        scores = []
        for _, hypothesis in pairs:
            label = next((value for key, value in labels.items() if key in hypothesis), "neutral")
            scores.append(
                {name: 0.9 if name == label else 0.05 for name in ("entailment", "neutral", "contradiction")}
            )
        return scores

    return classify


def test_split_claims_keeps_sentences_that_assert_something():
    response = (
        "The refund window is 14 days. Refunds go to the original card!\n"
        "- Support answers within a day\n"
        "Sure. The context does not mention fees. I don't know who approves exceptions."
    )
    assert split_claims(response) == [
        "The refund window is 14 days",
        "Refunds go to the original card",
        "Support answers within a day",
    ]


def test_long_contexts_are_also_read_in_two_sentence_windows():
    assert premise_windows(CONTEXT) == [
        "The refund window is 14 days. Refunds go to the original card.",
        "Refunds go to the original card. Support answers within a day.",
    ]
    assert premise_windows("One sentence only.") == []


def test_entailed_responses_are_grounded():
    report = GroundingChecker(scripted({"14 days": "entailment"})).check(
        "The refund window is 14 days.", CONTEXT
    )
    assert [claim.verdict for claim in report.claims] == ["supported"]
    assert not report.hallucinated


def test_a_contradicted_sentence_flags_the_response():
    checker = GroundingChecker(scripted({"14 days": "entailment", "30 days": "contradiction"}))
    report = checker.check("The refund window is 14 days. Exceptions extend it to 30 days.", CONTEXT)
    assert [claim.verdict for claim in report.claims] == ["supported", "contradicted"]
    assert report.hallucinated


def test_neutral_sentences_are_partial_when_the_context_covers_their_words():
    checker = GroundingChecker(scripted({}))
    report = checker.check("Refunds go to the original card quickly. A manager approves exceptions.", CONTEXT)
    assert [claim.verdict for claim in report.claims] == ["partially_supported", "unsupported"]
    assert report.hallucinated


def test_a_response_without_claims_is_not_a_hallucination():
    assert not GroundingChecker(scripted({})).check("I don't know.", CONTEXT).hallucinated


def test_empty_responses_get_no_judgement():
    assert judge.judge_response("prompt", "", "context") == judge.Judgement()


def test_local_mode_never_calls_the_paid_judge(monkeypatch):
    monkeypatch.setattr(judge.settings, "llm_provider", "local")
    monkeypatch.setattr(judge.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(judge, "_claude_judge", lambda *args: pytest.fail("called Claude in local mode"))
    assert judge.judge_response("prompt", "An answer.", "context") == judge.Judgement()


def test_the_claude_judge_reads_structured_scores(monkeypatch):
    res = NS(
        stop_reason="end_turn",
        content=[
            NS(type="thinking", thinking=""),
            NS(type="text", text='{"coherence": 7, "relevance": 3, "hallucinated": true}'),
        ],
    )
    requests = []
    fake = NS(messages=NS(create=lambda **request: requests.append(request) or res))
    monkeypatch.setattr(judge.settings, "llm_provider", "claude")
    monkeypatch.setattr(judge.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(judge, "_claude_client", lambda: fake)

    assert judge.judge_response("prompt", "An answer.", "context") == judge.Judgement(
        score=4, coherence=5, relevance=3, hallucinated=True
    )
    assert requests[0]["model"] == judge.settings.judge_model
    assert requests[0]["output_config"]["format"]["schema"] == judge.JUDGE_SCHEMA
    assert "temperature" not in requests[0]  # current models reject sampling parameters


def test_a_refused_judgement_leaves_the_scores_empty(monkeypatch):
    fake = NS(messages=NS(create=lambda **request: NS(stop_reason="refusal", content=[])))
    monkeypatch.setattr(judge.settings, "llm_provider", "claude")
    monkeypatch.setattr(judge.settings, "anthropic_api_key", "sk-test")
    monkeypatch.setattr(judge, "_claude_client", lambda: fake)
    assert judge.judge_response("prompt", "An answer.") == judge.Judgement()
