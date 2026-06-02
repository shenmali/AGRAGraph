import json
from src.graph.builder import graph


def run(query: str, docs: list = None) -> dict:
    initial_state = {
        "query": query,
        "query_type": None,
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "generated_answer": None,
        "hallucination_check": None,
        "relevance_check": None,
        "retry_count": 0,
        "max_retries": 2,
        "confidence": 0.0,
        "citations": [],
        "intermediate_results": [],
        "error": None,
    }

    result = graph.invoke(initial_state, {"recursion_limit": 25})

    return {
        "query": result["query"],
        "query_type": result.get("query_type", "unknown"),
        "answer": result.get("generated_answer", "No answer generated."),
        "confidence": result.get("confidence", 0.0),
        "citations": result.get("citations", []),
        "hallucination_check": result.get("hallucination_check", "unknown"),
        "relevance_check": result.get("relevance_check", "unknown"),
        "retry_count": result.get("retry_count", 0),
        "intermediate_results": result.get("intermediate_results", []),
    }


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m src.main <query>")
        sys.exit(1)

    query = " ".join(sys.argv[1:])
    result = run(query)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
