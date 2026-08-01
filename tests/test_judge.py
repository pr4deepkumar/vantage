import pytest
from vantage.eval.judge import heuristic_mock_judge, judge_response

def test_heuristic_mock_judge():
    prompt = "What is the capital of France?"
    response = "The capital of France is Paris."
    score, coherence, relevance, hallucinated = heuristic_mock_judge(prompt, response)
    
    assert 1 <= score <= 5
    assert 1 <= coherence <= 5
    assert 1 <= relevance <= 5
    assert isinstance(hallucinated, bool)

def test_heuristic_hallucination_detection():
    prompt = "What is the capital of France?"
    rag_context = "France's capital city is Paris."
    hallucinated_response = "France's capital is Berlin."
    
    _, _, _, hallucinated = heuristic_mock_judge(prompt, hallucinated_response, rag_context)
    assert hallucinated is True

def test_judge_response_empty_input():
    score, coherence, relevance, hallucinated = judge_response("Any prompt", "")
    assert score == 1
    assert coherence == 1
    assert relevance == 1
    assert hallucinated is False
