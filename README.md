# StudyMate-An-AI-Powered-PDF-Based-Q-A-System-for-Students
from __future__ import annotations

import io
from pathlib import Path
from typing import List

import streamlit as st

from studymate.config import load_config
from studymate.retriever import Retriever
from studymate.llm_client import GeminiClient
from studymate.pdf_utils import read_pdf_text
from studymate.chunking import chunk_pages


st.set_page_config(page_title="StudyMate", layout="wide")
config = load_config()

if "retriever" not in st.session_state:
    st.session_state.retriever = Retriever(config)
if "llm" not in st.session_state:
    try:
        st.session_state.llm = GeminiClient(config)
    except Exception as e:
        st.session_state.llm = None
        st.sidebar.error(f"LLM init failed: {e}")

st.title("StudyMate: PDF Q&A for Students")

with st.sidebar:
    st.header("Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", accept_multiple_files=True, type=["pdf"]
    )
    build_clicked = st.button("Index Uploaded PDFs")

    st.markdown("---")
    st.caption(
        f"Embedding model: `{config.embedding_model}` | Index dir: `{config.data_dir}`"
    )

if build_clicked and uploaded_files:
    temp_dir = Path(".studymate_tmp")
    temp_dir.mkdir(exist_ok=True)

    saved_paths: List[Path] = []
    for uf in uploaded_files:
        p = temp_dir / uf.name
        with open(p, "wb") as f:
            f.write(uf.getbuffer())
        saved_paths.append(p)

    with st.spinner("Building index..."):
        added = st.session_state.retriever.build_index_from_pdfs(saved_paths)
    st.success(f"Indexed {added} chunks from {len(saved_paths)} file(s).")

st.subheader("Ask a question")
query = st.text_input("Your question")
ask = st.button("Get Answer")

if ask and query.strip():
    retriever = st.session_state.retriever
    results = retriever.retrieve(query)
    if not results:
        st.warning("No results found. Please upload and index PDFs first.")
    else:
        contexts = [r.text for r in results]
        if st.session_state.llm is None:
            st.error("LLM not configured. Set GOOGLE_API_KEY in .env.")
        else:
            with st.spinner("Generating answer..."):
                answer = st.session_state.llm.answer(query, contexts)
            st.markdown("### Answer")
            st.write(answer)

        st.markdown("### Sources")
        for i, r in enumerate(results, start=1):
            st.markdown(
                f"{i}. Score: `{r.score:.3f}` — Document `{Path(r.source_path).name}` page {r.page_number}"
            )
            with st.expander("View chunk"):
                st.code(r.text)
