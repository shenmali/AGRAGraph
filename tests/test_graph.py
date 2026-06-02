from src.graph.state import Document
from src.graph.edges import route_after_hallucination, route_after_relevance


def test_route_hallucination_hallucinated():
    state = {"hallucination_check": "hallucinated", "retry_count": 0, "max_retries": 2}
    assert route_after_hallucination(state) == "refine_query"


def test_route_hallucination_grounded():
    state = {"hallucination_check": "grounded", "retry_count": 0, "max_retries": 2}
    assert route_after_hallucination(state) == "check_relevance"


def test_route_hallucination_max_retries_exceeded():
    state = {"hallucination_check": "hallucinated", "retry_count": 2, "max_retries": 2}
    assert route_after_hallucination(state) == "check_relevance"


def test_route_relevance_relevant():
    state = {"relevance_check": "relevant", "retry_count": 0, "max_retries": 2}
    assert route_after_relevance(state) == "finalize"


def test_route_relevance_irrelevant():
    state = {"relevance_check": "irrelevant", "retry_count": 0, "max_retries": 2}
    assert route_after_relevance(state) == "refine_query"
