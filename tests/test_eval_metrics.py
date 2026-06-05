from evals.metrics import answer_produced, citation_hit, extract_source_ids, summarize_records


def test_answer_produced():
    assert answer_produced("Paris is the capital.")
    assert not answer_produced("")
    assert not answer_produced("   ")
    assert not answer_produced(None)


def test_extract_source_ids_from_citations_and_lists():
    values = [
        "[Source 1] from graph-flow",
        "retrieval-routing",
        {"source_id": "self-correction"},
        {"source": "document-store"},
    ]

    assert extract_source_ids(values) == [
        "graph-flow",
        "retrieval-routing",
        "self-correction",
        "document-store",
    ]


def test_citation_hit():
    assert citation_hit(["graph-flow"], ["[Source 1] from graph-flow"])
    assert citation_hit(["graph-flow"], ["graph-flow"])
    assert not citation_hit(["graph-flow"], ["document-store"])


def test_summarize_records_counts_answer_and_citation_hits():
    records = [
        {
            "naive": {"answer": "answer", "observed_source_ids": ["graph-flow"], "latency_ms": 10},
            "agentic": {"answer": "answer", "observed_source_ids": ["graph-flow"], "latency_ms": 20, "retry_count": 1},
            "expected_source_ids": ["graph-flow"],
        },
        {
            "naive": {"answer": "", "observed_source_ids": [], "latency_ms": 5},
            "agentic": {"answer": "answer", "observed_source_ids": ["document-store"], "latency_ms": 15, "retry_count": 0},
            "expected_source_ids": ["graph-flow"],
        },
    ]

    summary = summarize_records(records)

    assert summary["total_questions"] == 2
    assert summary["naive_answer_rate"] == 0.5
    assert summary["agentic_answer_rate"] == 1.0
    assert summary["naive_citation_hit_rate"] == 0.5
    assert summary["agentic_citation_hit_rate"] == 0.5
    assert summary["agentic_total_retries"] == 1
