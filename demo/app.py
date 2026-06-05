import streamlit as st
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

st.set_page_config(page_title="Agentic RAG with LangGraph", page_icon="🧠", layout="wide")


def render_css():
    with open(os.path.join(os.path.dirname(__file__), "assets", "style.css")) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)


def render_sidebar():
    with st.sidebar:
        st.markdown("## Configuration")
        st.selectbox("LLM Provider", ["openrouter", "openai", "ollama"], key="provider")
        st.text_input("API Key", type="password", key="api_key")
        st.text_input("Model", key="model",
                      value=os.getenv("OPENROUTER_MODEL", "openai/gpt-4o"))
        st.markdown("---")
        st.slider("Top-K", 3, 20, 10, key="top_k")
        st.slider("Max Retries", 0, 5, 2, key="max_retries")
        st.markdown("---")
        st.caption("[GitHub](https://github.com/shenmali/AGRAGraph)")


def render_documents():
    from demo.sample_data import load_sample_corpus
    from src.retrievers.document_store import DocumentStore
    from src.retrievers.loader import extract_text
    import uuid

    store = DocumentStore.get_instance()
    st.metric("Documents indexed", store.count())

    if st.button("Load sample corpus"):
        added = load_sample_corpus()
        if added:
            st.success(f"Loaded {added} sample documents")
        else:
            st.info("Sample corpus already loaded")
        st.rerun()

    tab1, tab2 = st.tabs([" Paste Text", " Upload File"])

    with tab1:
        with st.form("add_text"):
            text = st.text_area("Paste content", height=120,
                               placeholder="Paste any text here...")
            source = st.text_input("Source", placeholder="e.g. Wikipedia - AI")
            if st.form_submit_button("Add Text") and text.strip():
                store.add_documents([text.strip()], [{"source": source or "manual"}], [str(uuid.uuid4())])
                st.success("Added")
                st.rerun()

    with tab2:
        uploaded = st.file_uploader("Upload file", type=["txt", "md", "pdf"],
                                    accept_multiple_files=True,
                                    help="Supported: TXT, MD, PDF")
        if uploaded:
            for file in uploaded:
                content = extract_text(file.read(), file.name)
                if content.strip():
                    if len(content) > 12000:
                        chunks = [content[i:i+6000] for i in range(0, len(content), 6000)]
                        for j, chunk in enumerate(chunks):
                            store.add_documents(
                                [chunk],
                                [{"source": f"{file.name} (part {j+1}/{len(chunks)})"}],
                                [str(uuid.uuid4())],
                            )
                        st.info(f"Split {file.name} into {len(chunks)} chunks")
                    else:
                        store.add_documents([content], [{"source": file.name}], [str(uuid.uuid4())])
                        st.success(f"Added {file.name} ({len(content)} chars)")
                else:
                    st.warning(f"Skipped {file.name}: empty after extraction")
            st.rerun()


def run_graph(query: str) -> dict:
    from src.config import config
    from src.graph.builder import graph

    config.top_k_initial = st.session_state.get("top_k", 10)
    config.max_retries = st.session_state.get("max_retries", 2)
    config.llm_provider = st.session_state.get("provider", "openrouter")

    provider = st.session_state.get("provider", "openrouter")
    if provider == "openrouter":
        config.openrouter_api_key = st.session_state.get("api_key", "")
        config.openrouter_model = st.session_state.get("model", "openai/gpt-4o")
    elif provider == "openai":
        config.openai_api_key = st.session_state.get("api_key", "")
        config.openai_model = st.session_state.get("model", "gpt-4o")
    elif provider == "ollama":
        config.ollama_model = st.session_state.get("model", "llama3.1:8b")

    state = {
        "query": query, "query_type": None, "retrieved_chunks": [],
        "reranked_chunks": [], "generated_answer": None,
        "hallucination_check": None, "relevance_check": None,
        "retry_count": 0, "max_retries": config.max_retries,
        "confidence": 0.0, "citations": [], "intermediate_results": [], "error": None,
    }
    return graph.invoke(state, {"recursion_limit": 25})


def render_results(result: dict):
    if not result:
        return
    st.divider()
    st.subheader("Answer")
    st.markdown(result.get("generated_answer", "*No answer generated.*"))

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Confidence", f"{result.get('confidence', 0):.0%}")
    c2.metric("Hallucination", result.get("hallucination_check", "N/A").capitalize())
    c3.metric("Relevance", result.get("relevance_check", "N/A").capitalize())
    c4.metric("Retries", result.get("retry_count", 0))

    chunks = result.get("reranked_chunks", [])
    if chunks:
        with st.expander("Sources", expanded=True):
            for index, doc in enumerate(chunks, start=1):
                source = doc.metadata.get("source", "unknown")
                preview = doc.content[:280] + ("..." if len(doc.content) > 280 else "")
                st.markdown(f"**[Source {index}] {source}**")
                st.caption(preview)

    with st.expander("Pipeline trace", expanded=True):
        for index, step in enumerate(result.get("intermediate_results", []), start=1):
            st.markdown(f"**{index}. `{step['node']}`** - {step['output']}")


def main():
    render_css()
    st.title("Agentic RAG — LangGraph")
    st.caption("Self-correcting multi-strategy retrieval. Bring your own LLM key.")
    render_sidebar()

    t1, t2 = st.tabs(["Documents", "Query"])
    with t1:
        render_documents()
    with t2:
        from demo.sample_data import get_sample_questions

        sample_question = st.selectbox(
            "Try a sample question",
            [""] + get_sample_questions(),
            format_func=lambda value: "Custom question" if value == "" else value,
        )
        query = st.text_area("Enter query", value=sample_question, height=80)
        c1, c2 = st.columns([1, 3])
        if c1.button("Run", type="primary"):
            with st.spinner("Running..."):
                result = run_graph(query.strip())
                st.session_state["result"] = result
        if c2.button("Clear"):
            st.session_state.pop("result", None)
            st.rerun()
        if "result" in st.session_state:
            render_results(st.session_state["result"])


if __name__ == "__main__":
    main()
