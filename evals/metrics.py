from __future__ import annotations

from typing import Any


def answer_produced(answer: str | None) -> bool:
    return bool(answer and answer.strip())


def extract_source_ids(values: list[Any]) -> list[str]:
    source_ids: list[str] = []
    for value in values:
        if isinstance(value, dict):
            source_id = value.get("source_id") or value.get("source")
            if source_id:
                source_ids.append(str(source_id))
        elif isinstance(value, str):
            if " from " in value:
                source_ids.append(value.rsplit(" from ", 1)[-1].strip())
            elif value.strip():
                source_ids.append(value.strip())
    return source_ids


def citation_hit(expected_source_ids: list[str], observed_values: list[Any]) -> bool:
    expected = set(expected_source_ids)
    observed = set(extract_source_ids(observed_values))
    return bool(expected & observed)


def _rate(count: int, total: int) -> float:
    if total == 0:
        return 0.0
    return round(count / total, 3)


def summarize_records(records: list[dict]) -> dict:
    total = len(records)
    naive_answer_count = 0
    agentic_answer_count = 0
    naive_citation_hits = 0
    agentic_citation_hits = 0
    agentic_total_retries = 0

    for record in records:
        expected = record.get("expected_source_ids", [])
        naive = record.get("naive", {})
        agentic = record.get("agentic", {})

        if answer_produced(naive.get("answer")):
            naive_answer_count += 1
        if answer_produced(agentic.get("answer")):
            agentic_answer_count += 1
        if citation_hit(expected, naive.get("observed_source_ids", [])):
            naive_citation_hits += 1
        if citation_hit(expected, agentic.get("observed_source_ids", [])):
            agentic_citation_hits += 1

        agentic_total_retries += int(agentic.get("retry_count", 0) or 0)

    return {
        "total_questions": total,
        "naive_answer_rate": _rate(naive_answer_count, total),
        "agentic_answer_rate": _rate(agentic_answer_count, total),
        "naive_citation_hit_rate": _rate(naive_citation_hits, total),
        "agentic_citation_hit_rate": _rate(agentic_citation_hits, total),
        "agentic_total_retries": agentic_total_retries,
    }
