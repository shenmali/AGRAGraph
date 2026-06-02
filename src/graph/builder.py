from langgraph.graph import StateGraph, START, END
from src.graph.state import AgentState
from src.graph.nodes import (
    classify_query,
    retrieve_bm25,
    retrieve_dense,
    rerank,
    generate,
    check_hallucination,
    check_relevance,
    refine_query,
    finalize,
)
from src.graph.edges import (
    route_after_hallucination,
    route_after_relevance,
    route_after_retrieval,
    route_after_classify,
)
from src.config import config


def build_graph() -> StateGraph:
    builder = StateGraph(AgentState)

    builder.add_node("classify_query", classify_query)
    builder.add_node("retrieve_bm25", retrieve_bm25)
    builder.add_node("retrieve_dense", retrieve_dense)
    builder.add_node("rerank", rerank)
    builder.add_node("generate", generate)
    builder.add_node("check_hallucination", check_hallucination)
    builder.add_node("check_relevance", check_relevance)
    builder.add_node("refine_query", refine_query)
    builder.add_node("finalize", finalize)

    builder.add_conditional_edges(
        "classify_query",
        route_after_classify,
        {
            "factual": "retrieve_bm25",
            "analytical": "retrieve_dense",
            "creative": "retrieve_dense",
        },
    )

    builder.add_conditional_edges(
        "retrieve_bm25",
        route_after_retrieval,
        {
            "rerank": "rerank",
            "refine_query": "refine_query",
        },
    )

    builder.add_conditional_edges(
        "retrieve_dense",
        route_after_retrieval,
        {
            "rerank": "rerank",
            "refine_query": "refine_query",
        },
    )

    builder.add_edge("rerank", "generate")
    builder.add_edge("generate", "check_hallucination")

    builder.add_conditional_edges(
        "check_hallucination",
        route_after_hallucination,
        {
            "refine_query": "refine_query",
            "check_relevance": "check_relevance",
        },
    )

    builder.add_conditional_edges(
        "check_relevance",
        route_after_relevance,
        {
            "refine_query": "refine_query",
            "finalize": "finalize",
        },
    )

    builder.add_edge("refine_query", "rerank")
    builder.add_edge("finalize", END)

    builder.set_entry_point("classify_query")

    return builder.compile()


graph = build_graph()
