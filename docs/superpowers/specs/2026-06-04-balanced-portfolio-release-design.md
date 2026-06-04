# Balanced Portfolio Release Design

Date: 2026-06-04

## Purpose

AGRAGraph is already a compact Agentic RAG portfolio project: a LangGraph state machine with query classification, retrieval, reranking, answer generation, hallucination checking, relevance checking, query refinement, and finalization.

The next release should make the project more convincing in two ways:

1. Improve the demo experience so a reviewer can understand the pipeline quickly.
2. Add a small reproducible benchmark so the README claims are backed by local evidence instead of only explanation.

The release should preserve the existing graph shape. The graph currently has 9 nodes, and this work will not add a 10th node.

## Scope

The release has three parts.

### Demo Polish

The Streamlit app gets a smoother first-run experience:

- A sample corpus can be loaded from the UI.
- A small set of sample questions is available in the query view.
- The pipeline trace is shown as a readable step sequence.
- Citations include short source previews, not only source names.
- Existing sidebar configuration remains in place.

The app continues to surface LLM/API failures instead of swallowing them.

### Benchmark And Eval Harness

A new eval layer compares naive RAG against the existing agentic graph:

- A small QA dataset lives in `evals/datasets/sample_qa.json`.
- A naive baseline lives under `evals/baselines/`.
- An eval runner loads the same sample corpus into `DocumentStore`, runs both systems, and writes result records.
- A report writer produces JSON and Markdown output.

The benchmark is intentionally small and local. It is meant to make the portfolio demo reproducible, not to claim academic benchmark validity.

### README Proof Layer

The README links to the benchmark output and explains how to reproduce it locally:

- How to run the Streamlit demo.
- How to load the sample corpus.
- How to run the eval command.
- What the benchmark does and does not prove.

## Architecture

The current graph remains the core runtime:

- `src/graph/state.py` remains the source of truth for state fields.
- `src/graph/nodes.py` remains the home of graph node functions.
- `src/graph/edges.py` is not changed unless a flow bug is discovered.
- `src/retrievers/` remains responsible for loading, storing, and retrieving documents.

New code sits around the graph instead of inside it:

- `demo/sample_data.py` exposes the sample corpus and sample questions.
- `evals/datasets/sample_qa.json` stores the eval questions and expected metadata.
- `evals/baselines/naive_rag.py` implements retrieve plus generate without self-correction.
- `evals/run_eval.py` orchestrates naive and agentic runs.
- `evals/report.py` writes Markdown and JSON reports.
- `docs/benchmark-results.md` stores the latest sample result for README linking.

## Data Flow

The common data flow is:

```text
sample corpus
  -> DocumentStore
  -> naive runner / agentic graph runner
  -> answer records
  -> metrics
  -> Markdown and JSON reports
```

Both runners use the same sample corpus and dataset. This keeps comparisons focused on the pipeline behavior rather than input differences.

## Eval Behavior

The eval command should be:

```bash
python3 -m evals.run_eval --dataset evals/datasets/sample_qa.json --out docs/benchmark-results.md
```

When an API key is available, the runner can produce generated answers and LLM-judge metrics.

When an API key is not available:

- Dataset loading still runs.
- Sample corpus loading still runs.
- Retrieval smoke checks still run.
- LLM-dependent metrics are marked as skipped.
- The report clearly sets `skipped_llm=true`.

One failed question should not stop the whole eval. The runner records the error for that question and continues.

## Metrics

The first version uses a small metric set:

- `answer_produced`: whether a non-empty answer exists.
- `citation_hit`: whether expected source IDs appear in citations or retrieved chunks.
- `retry_count`: how many refinement loops the graph used.
- `latency_ms`: wall-clock runtime for each question.
- `groundedness`: LLM judge result when available.
- `relevance`: LLM judge result when available.

The Markdown report includes:

- Dataset summary.
- Naive vs agentic comparison table.
- Retry, citation, and latency summary.
- A few example comparison records.
- Caveat that this is a small local benchmark.

## UI Behavior

The Streamlit UI keeps the existing two-tab model:

- Documents
- Query

Changes in the Documents tab:

- Add a `Load sample corpus` action.
- Show indexed document count after loading.

Changes in the Query tab:

- Add selectable sample questions.
- Keep free-form query input.
- Show answer, confidence, hallucination status, relevance status, retry count, citations, and pipeline trace.
- Show citation previews using document metadata and short content snippets.

## Error Handling

Runtime behavior remains explicit:

- Graph node LLM failures surface in Streamlit.
- Eval runner handles per-question failures by writing an `error` field.
- Eval runner marks unavailable LLM-dependent checks as skipped rather than pretending they passed.
- Tests reset the `DocumentStore` singleton before scenarios that need isolation.

## Tests

Add focused tests:

- `tests/test_sample_data.py`: sample corpus and sample questions are valid.
- `tests/test_eval_dataset.py`: QA dataset schema is valid.
- `tests/test_eval_metrics.py`: deterministic metric calculations are correct.

Keep existing tests:

- `tests/test_graph.py`
- `tests/test_retrievers.py`
- `tests/test_evaluators.py`

Streamlit browser end-to-end tests are out of scope for this release. A basic import/smoke check is sufficient.

## Out Of Scope

This release will not include:

- New graph nodes.
- External vector databases.
- Production authentication.
- Deployment automation.
- Large public benchmark datasets.
- LangSmith or any external observability vendor integration.
- Full browser E2E testing for Streamlit.

## Success Criteria

The release is successful when:

- A reviewer can load sample docs and run a sample query in the Streamlit UI.
- The pipeline trace is readable without reading the source code.
- `python3 -m evals.run_eval --dataset evals/datasets/sample_qa.json --out docs/benchmark-results.md` produces a report.
- The eval runner degrades clearly when no LLM key is configured.
- `python3 -m pytest tests/ -v` passes.
- The graph integrity check still lists the same 9 graph nodes.

