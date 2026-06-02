# AGRAGraph — Development Guidelines

> Agentic RAG pipeline with LangGraph. Self-correcting retrieval. 9-node state graph.

## Karpathy Principles

### 1. Think Before Coding
- State assumptions explicitly. If uncertain, ask.
- Before adding a new node/edge, ask: does this belong in the graph or is it a utility?
- Graph nodes are for state-machine steps. Everything else goes in `retrievers/` or `models/`.

### 2. Simplicity First
- The graph has 9 nodes. Don't add a 10th without a verified need.
- `nodes.py` functions return `dict` with keys from `AgentState`. No exceptions.
- Rerank uses a local cross-encoder (`ms-marco-MiniLM-L-6-v2`), NOT an LLM call. <50ms per chunk.
- DocumentStore is pure in-memory (numpy-backed). No external DB dependency.

### 3. Surgical Changes
- If you change a field in `AgentState` (state.py), check every node that reads/writes it.
- Don't touch the graph edges (`edges.py`) unless the pipeline flow actually changes.
- Streamlit UI wiring lives in `demo/app.py` only. Config lives in `src/config.py` only.

### 4. Goal-Driven Execution
- Verify: `python3 -m pytest tests/ -v` must pass after every change.
- Graph integrity check: `python3 -c "from src.graph.builder import graph; print(list(graph.nodes.keys()))"`
- Streamlit check: app must import without error before launching.

## Architecture

```
src/
├── graph/
│   ├── state.py       # AgentState TypedDict — ONE source of truth for all state keys
│   ├── nodes.py       # classify_query → retrieve → rerank → generate → check_* → refine → finalize
│   ├── edges.py       # Conditional routing: hallucination → refine, relevance → refine, classify → retriever
│   └── builder.py     # StateGraph assembly. Entry: classify_query. Exit: finalize (END)
├── retrievers/
│   ├── document_store.py  # Singleton, in-memory, numpy cosine similarity
│   ├── bm25_retriever.py  # rank_bm25 keyword search
│   ├── dense_retriever.py # sentence-transformers semantic search
│   └── loader.py          # PDF/MD/TXT extractor (PyPDF2)
├── models/
│   └── llm.py             # OpenRouter ↔ OpenAI ↔ Ollama client factory. get_llm() returns callable.
└── config.py               # Env-driven config. No defaults leaked to nodes.

demo/app.py  → Streamlit UI (sidebar config + document upload + query + results)
```

## Graph Flow

```
classify → [bm25|dense] → rerank → generate → hallucination? ──(yes, retry<max)──→ refine → rerank
                                                        │
                                                   (no) │
                                                        ▼
                                                  relevance? ──(no, retry<max)──→ refine → rerank
                                                        │
                                                   (yes)│
                                                        ▼
                                                     finalize → END
```

## Conventions

- **No LLM calls in retrievers or reranker.** Only in `classify_query`, `generate`, `check_hallucination`, `check_relevance`, `refine_query`.
- **Env vars only in config.py.** Nodes read from `config` object, never from `os.getenv()`.
- **DocumentStore singleton.** Use `DocumentStore.get_instance()`. Tests reset it via `store.documents = []`.
- **Error handling:** If `get_llm()` fails (no API key), the graph will fail at the first LLM-using node. That's expected. Streamlit shows the error. No try/except swallowing.

## Tests

- `test_graph.py` — Unit tests for edge routing logic (no LLM needed)
- `test_retrievers.py` — DocumentStore + BM25 (needs sentence-transformers, no LLM)
- `test_evaluators.py` — Document model only (no LLM needed)

Run: `python3 -m pytest tests/ -v`
