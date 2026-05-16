# app.py - complete fixed version

import streamlit as st
import os
from ingest import ingest_file
from rag_chain import get_qa_chain, LANGUAGES


# ── PAGE SETUP ────────────────────────────────────────────
st.set_page_config(
    page_title="🌐 Multilingual RAG",
    page_icon="🌐",
    layout="centered"
)

st.title("🌐 Multilingual Document Q&A")
st.caption("Upload documents in any language — ask questions in your language!")


# ── SIDEBAR ───────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    model_choice = st.selectbox(
        "Ollama Model",
        ["llama3.2", "qwen2.5"],
        help="qwen2.5 is better for Telugu & Hindi"
    )

    response_lang = st.selectbox(
        "🗣️ Reply Language",
        list(LANGUAGES.keys()),
        index=0
    )

    st.markdown("---")
    st.markdown("**Supported Input Languages:**")
    st.markdown("🇮🇳 Hindi | తెలుగు | 🇬🇧 English")
    st.markdown("🇪🇸 Spanish | 🇫🇷 French | + 45 more")


# ── FILE UPLOAD ───────────────────────────────────────────
st.subheader("📄 Upload Document")

uploaded = st.file_uploader(
    "Upload a PDF or TXT file (any language)",
    type=["pdf", "txt"]
)

if uploaded:
    os.makedirs("data", exist_ok=True)
    file_path = f"data/{uploaded.name}"

    with open(file_path, "wb") as f:
        f.write(uploaded.read())

    with st.spinner(f"📥 Processing '{uploaded.name}'..."):
        ingest_file(file_path)

    st.success(f"✅ '{uploaded.name}' is ready! Ask questions below.")
    st.info(f"🗣️ Answers will be in **{response_lang}**")


    # ── CHAT INTERFACE ────────────────────────────────────
    st.subheader("💬 Ask Your Question")
    st.caption("You can type your question in ANY language!")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous chat history
    for chat in st.session_state.chat_history:
        with st.chat_message("user"):
            st.write(chat["question"])
        with st.chat_message("assistant"):
            st.write(chat["answer"])
            with st.expander("📎 Source Chunks Used"):
                for i, chunk in enumerate(chat["sources"]):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(chunk)
                    st.markdown("---")

    # New question input
    question = st.chat_input("Type your question here... (any language)")

    if question:
        # Show user message
        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            with st.spinner("🤔 Thinking..."):

                # Get chain and retriever
                chain, retriever = get_qa_chain(
                    response_language=response_lang,
                    model=model_choice
                )

                # Get answer from chain
                answer = chain.invoke(question)

                # Get source chunks separately
                source_docs = retriever.invoke(question)

            # Display answer
            st.write(answer)

            # Display source chunks
            with st.expander("📎 Source Chunks Used"):
                for i, doc in enumerate(source_docs):
                    st.markdown(f"**Chunk {i+1}:**")
                    st.write(doc.page_content)
                    st.markdown("---")

        # Save to chat history
        st.session_state.chat_history.append({
            "question": question,
            "answer": answer,
            "sources": [doc.page_content for doc in source_docs]
        })