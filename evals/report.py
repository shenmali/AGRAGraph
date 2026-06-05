from __future__ import annotations

import json
from pathlib import Path

from evals.metrics import summarize_records


def write_json_report(payload: dict, markdown_path: Path) -> Path:
    json_path = markdown_path.with_suffix(".json")
    json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return json_path


def write_markdown_report(payload: dict, markdown_path: Path) -> Path:
    summary = payload["summary"]
    records = payload["records"]
    lines = [
        "# AGRAGraph Sample Benchmark Results",
        "",
        f"Dataset: `{payload['dataset_name']}`",
        f"Questions: {summary['total_questions']}",
        f"LLM-dependent steps skipped: `{payload['skipped_llm']}`",
        "",
        "This is a small local portfolio benchmark. It is not an academic benchmark.",
        "",
        "## Summary",
        "",
        "| Metric | Naive RAG | Agentic Graph |",
        "| --- | ---: | ---: |",
        f"| Answer rate | {summary['naive_answer_rate']:.3f} | {summary['agentic_answer_rate']:.3f} |",
        f"| Citation hit rate | {summary['naive_citation_hit_rate']:.3f} | {summary['agentic_citation_hit_rate']:.3f} |",
        f"| Agentic total retries | n/a | {summary['agentic_total_retries']} |",
        "",
        "## Records",
        "",
    ]

    for record in records:
        lines.extend(
            [
                f"### {record['id']}: {record['query']}",
                "",
                f"- Expected sources: `{', '.join(record['expected_source_ids'])}`",
                f"- Naive observed sources: `{', '.join(record['naive'].get('observed_source_ids', []))}`",
                f"- Agentic observed sources: `{', '.join(record['agentic'].get('observed_source_ids', []))}`",
                f"- Naive latency: `{record['naive'].get('latency_ms', 0)}` ms",
                f"- Agentic latency: `{record['agentic'].get('latency_ms', 0)}` ms",
                "",
            ]
        )
        if record["naive"].get("error"):
            lines.append(f"- Naive error: `{record['naive']['error']}`")
        if record["agentic"].get("error"):
            lines.append(f"- Agentic error: `{record['agentic']['error']}`")
        lines.append("")

    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text("\n".join(lines).rstrip() + "\n")
    write_json_report(payload, markdown_path)
    return markdown_path


def build_report_payload(dataset_name: str, records: list[dict], skipped_llm: bool) -> dict:
    return {
        "dataset_name": dataset_name,
        "skipped_llm": skipped_llm,
        "summary": summarize_records(records),
        "records": records,
    }
