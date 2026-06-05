from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from demo.sample_data import load_sample_corpus
from evals.baselines.naive_rag import llm_is_configured, run_naive_rag
from evals.metrics import extract_source_ids
from evals.report import build_report_payload, write_markdown_report
from src.main import run as run_agentic_graph
from src.retrievers.bm25_retriever import BM25Retriever


def load_dataset(path: Path) -> dict:
    return json.loads(path.read_text())


def _elapsed_ms(start: float) -> int:
    return int((time.perf_counter() - start) * 1000)


def _retrieval_smoke(query: str, k: int = 5) -> dict:
    start = time.perf_counter()
    chunks = BM25Retriever().retrieve(query, k=k)
    observed_source_ids = [doc.metadata.get("source_id") or doc.metadata.get("source", "unknown") for doc in chunks]
    return {
        "answer": "",
        "citations": [],
        "observed_source_ids": observed_source_ids,
        "retrieved_count": len(chunks),
        "latency_ms": _elapsed_ms(start),
        "error": None,
    }


def _run_naive_record(query: str) -> dict:
    start = time.perf_counter()
    try:
        result = run_naive_rag(query)
        result["latency_ms"] = _elapsed_ms(start)
        result["error"] = None
        return result
    except Exception as exc:
        return {
            "answer": "",
            "citations": [],
            "observed_source_ids": [],
            "retrieved_count": 0,
            "latency_ms": _elapsed_ms(start),
            "error": str(exc),
        }


def _run_agentic_record(query: str) -> dict:
    start = time.perf_counter()
    try:
        result = run_agentic_graph(query)
        observed_source_ids = extract_source_ids(result.get("citations", []))
        return {
            "answer": result.get("answer", ""),
            "citations": result.get("citations", []),
            "observed_source_ids": observed_source_ids,
            "retry_count": result.get("retry_count", 0),
            "confidence": result.get("confidence", 0.0),
            "latency_ms": _elapsed_ms(start),
            "error": None,
        }
    except Exception as exc:
        return {
            "answer": "",
            "citations": [],
            "observed_source_ids": [],
            "retry_count": 0,
            "confidence": 0.0,
            "latency_ms": _elapsed_ms(start),
            "error": str(exc),
        }


def run_eval(dataset_path: Path, output_path: Path) -> dict:
    dataset = load_dataset(dataset_path)
    load_sample_corpus(reset=True)

    skipped_llm = not llm_is_configured()
    records = []

    for item in dataset["questions"]:
        query = item["query"]
        if skipped_llm:
            naive = _retrieval_smoke(query)
            agentic = {
                "answer": "",
                "citations": [],
                "observed_source_ids": [],
                "retry_count": 0,
                "confidence": 0.0,
                "latency_ms": 0,
                "error": "LLM configuration unavailable; agentic graph skipped.",
            }
        else:
            naive = _run_naive_record(query)
            agentic = _run_agentic_record(query)

        records.append(
            {
                "id": item["id"],
                "query": query,
                "expected_answer": item["expected_answer"],
                "expected_source_ids": item["expected_source_ids"],
                "naive": naive,
                "agentic": agentic,
            }
        )

    payload = build_report_payload(dataset["name"], records, skipped_llm=skipped_llm)
    write_markdown_report(payload, output_path)
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run AGRAGraph sample benchmark.")
    parser.add_argument("--dataset", default="evals/datasets/sample_qa.json")
    parser.add_argument("--out", default="docs/benchmark-results.md")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    payload = run_eval(Path(args.dataset), Path(args.out))
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
