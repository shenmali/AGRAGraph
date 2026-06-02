def route_after_hallucination(state: dict) -> str:
    check = state.get("hallucination_check", "hallucinated")
    retry = state.get("retry_count", 0)
    max_r = state.get("max_retries", 2)

    if check == "hallucinated" and retry < max_r:
        return "refine_query"
    return "check_relevance"


def route_after_relevance(state: dict) -> str:
    check = state.get("relevance_check", "irrelevant")
    retry = state.get("retry_count", 0)
    max_r = state.get("max_retries", 2)

    if check == "irrelevant" and retry < max_r:
        return "refine_query"
    return "finalize"


def route_after_retrieval(state: dict) -> str:
    if state.get("retrieved_chunks"):
        return "rerank"
    return "refine_query"


def route_after_classify(state: dict) -> str:
    qtype = state.get("query_type", "factual")
    return qtype
