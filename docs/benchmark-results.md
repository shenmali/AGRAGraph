# AGRAGraph Sample Benchmark Results

Dataset: `agra_sample_qa`
Questions: 5
LLM-dependent steps skipped: `True`

This is a small local portfolio benchmark. It is not an academic benchmark.

## Summary

| Metric | Naive RAG | Agentic Graph |
| --- | ---: | ---: |
| Answer rate | 0.000 | 0.000 |
| Citation hit rate | 1.000 | 0.000 |
| Agentic total retries | n/a | 0 |

## Records

### q1: How many nodes does the AGRAGraph graph use?

- Expected sources: `graph-flow`
- Naive observed sources: `retrieval-routing, graph-flow, benchmark-purpose`
- Agentic observed sources: ``
- Naive latency: `12` ms
- Agentic latency: `0` ms

- Agentic error: `LLM configuration unavailable; agentic graph skipped.`

### q2: Which retriever is used for factual queries?

- Expected sources: `retrieval-routing`
- Naive observed sources: `retrieval-routing, document-store, agra-overview, benchmark-purpose`
- Agentic observed sources: ``
- Naive latency: `0` ms
- Agentic latency: `0` ms

- Agentic error: `LLM configuration unavailable; agentic graph skipped.`

### q3: What happens when the generated answer is hallucinated?

- Expected sources: `self-correction`
- Naive observed sources: `benchmark-purpose, self-correction, agra-overview, retrieval-routing, document-store`
- Agentic observed sources: ``
- Naive latency: `0` ms
- Agentic latency: `0` ms

- Agentic error: `LLM configuration unavailable; agentic graph skipped.`

### q4: Does AGRAGraph require an external vector database?

- Expected sources: `document-store`
- Naive observed sources: `document-store, agra-overview`
- Agentic observed sources: ``
- Naive latency: `0` ms
- Agentic latency: `0` ms

- Agentic error: `LLM configuration unavailable; agentic graph skipped.`

### q5: Why does the project include a local benchmark?

- Expected sources: `benchmark-purpose`
- Naive observed sources: `document-store, benchmark-purpose, agra-overview`
- Agentic observed sources: ``
- Naive latency: `0` ms
- Agentic latency: `0` ms

- Agentic error: `LLM configuration unavailable; agentic graph skipped.`
