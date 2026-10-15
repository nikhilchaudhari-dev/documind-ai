
import hashlib
import tempfile
from pathlib import Path

import streamlit as st

from src.knowledge_base import ingest_pdf
from src.rag import answer_question
from src.embeddings import load_embedding_model
from src.vector_stores import load_knowledge_base


# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_NAME = "DocuMind AI"
DEFAULT_MODEL = "llama3.2:3b"
STORE_DIR = Path("data") / "streamlit_knowledge_base"
TOP_K = 5


# ============================================================
# SESSION STATE
# ============================================================

if "active_upload_id" not in st.session_state:
    st.session_state.active_upload_id = None

if "processed_filename" not in st.session_state:
    st.session_state.processed_filename = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


# ============================================================
# CACHED EMBEDDING MODEL
# ============================================================

@st.cache_resource
def get_embedding_model():
    """Load the embedding model once and reuse it."""
    return load_embedding_model()


# ============================================================
# RAG QUESTION ANSWERING
# ============================================================

def get_rag_answer(question: str, store_dir: Path) -> dict:
    """Retrieve relevant PDF chunks and generate an answer."""
    index, chunks = load_knowledge_base(str(store_dir))
    embedding_model = get_embedding_model()

    return answer_question(
        question=question,
        index=index,
        chunks=chunks,
        embedding_model=embedding_model,
        top_k=TOP_K,
        llm_model=DEFAULT_MODEL,
    )


# ============================================================
# SOURCE DISPLAY HELPERS
# ============================================================

def get_source_value(source, *keys):
    """Get a source field while supporting different metadata names."""
    if not isinstance(source, dict):
        return None

    for key in keys:
        value = source.get(key)
        if value is not None and value != "":
            return value

    metadata = source.get("metadata")
    if isinstance(metadata, dict):
        for key in keys:
            value = metadata.get(key)
            if value is not None and value != "":
                return value

    return None


def render_sources(sources):
    """Display retrieved source passages without rendering raw HTML."""
    if not sources:
        st.caption("No source passages were returned.")
        return

    with st.expander(f"📖 Sources used ({len(sources)})", expanded=False):
        for number, source in enumerate(sources, start=1):
            filename = get_source_value(
                source, "source", "filename", "file_name", "document"
            )

            page = get_source_value(
                source, "page", "page_number", "page_num"
            )

            passage = get_source_value(
                source, "text", "content", "chunk", "page_content"
            )

            if isinstance(source, str):
                filename = source

            if filename:
                heading = f"Source {number}: {Path(str(filename)).name}"
            else:
                heading = f"Source {number}"

            if page is not None:
                heading += f" · Page {page}"

            st.markdown(f"**{heading}**")

            if passage:
                st.write(str(passage))
            elif not isinstance(source, str):
                st.caption("Source details are available, but no passage text was returned.")

            if number < len(sources):
                st.divider()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:
    st.title("📚 DocuMind AI")
    st.caption("Your personal AI document assistant")

    st.divider()

    st.subheader("Upload a document")

    uploaded_file = st.file_uploader(
        "Choose a PDF file",
        type=["pdf"],
        accept_multiple_files=False,
        help="Upload one PDF to start asking questions about it.",
    )

    if uploaded_file is not None:
        pdf_bytes = uploaded_file.getvalue()
        upload_id = hashlib.sha256(pdf_bytes).hexdigest()

        # Process only when a new or different PDF is uploaded.
        if upload_id != st.session_state.active_upload_id:
            temp_pdf_path = None

            try:
                with st.spinner("Reading and indexing your PDF..."):
                    STORE_DIR.mkdir(parents=True, exist_ok=True)

                    with tempfile.NamedTemporaryFile(
                        mode="wb",
                        suffix=".pdf",
                        delete=False,
                    ) as temp_pdf:
                        temp_pdf.write(pdf_bytes)
                        temp_pdf_path = Path(temp_pdf.name)

                    ingest_pdf(
                        str(temp_pdf_path),
                        str(STORE_DIR),
                    )

                st.session_state.active_upload_id = upload_id
                st.session_state.processed_filename = uploaded_file.name
                st.session_state.chat_history = []

                st.success("PDF processed successfully.")

            except Exception as exc:
                st.error(f"Could not process the PDF: {exc}")

            finally:
                if temp_pdf_path is not None:
                    try:
                        temp_pdf_path.unlink(missing_ok=True)
                    except OSError:
                        pass

    st.divider()

    st.subheader("Active document")

    if st.session_state.processed_filename:
        st.success("Document ready")
        st.write(f"📄 {st.session_state.processed_filename}")
        st.caption("You can now ask questions about this PDF.")

        if st.button("Clear chat history", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()
    else:
        st.info("Upload a PDF to get started.")

    st.divider()

    st.caption("AI model")
    st.write(f"**{DEFAULT_MODEL}**")
    st.caption("Answers are generated using your local Ollama model.")


# ============================================================
# MAIN HEADER — NATIVE STREAMLIT, NO CUSTOM HTML
# ============================================================

st.markdown("##### AI-POWERED DOCUMENT INTELLIGENCE")
st.title("Chat with your documents.")

st.write(
    "Turn lengthy PDFs into clear answers. Ask questions, "
    "explore important information, and review the passages "
    "that support your answers."
)

st.divider()


# ============================================================
# EMPTY STATE
# ============================================================

if not st.session_state.processed_filename:
    st.subheader("Get started")

    st.write(
        "Upload a PDF using the sidebar. DocuMind AI will process "
        "the document so you can ask questions about its contents."
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        with st.container(border=True):
            st.subheader("📄 Read")
            st.write("Extract readable text from your PDF.")

    with col2:
        with st.container(border=True):
            st.subheader("🔎 Retrieve")
            st.write("Find relevant passages for your question.")

    with col3:
        with st.container(border=True):
            st.subheader("💬 Ask")
            st.write("Get answers with supporting source passages.")

    st.info("Start by uploading a PDF in the sidebar.")


# ============================================================
# CHAT HISTORY
# ============================================================

if st.session_state.processed_filename:
    st.subheader("Conversation")

    for message in st.session_state.chat_history:
        role = message.get("role", "assistant")
        content = message.get("content", "")

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant" and message.get("sources"):
                render_sources(message["sources"])


# ============================================================
# CHAT INPUT AND ANSWER GENERATION
# ============================================================

if st.session_state.processed_filename:
    question = st.chat_input(
        "Ask a question about your document..."
    )

    if question and question.strip():
        question = question.strip()

        # Display and save the user's question.
        st.session_state.chat_history.append(
            {
                "role": "user",
                "content": question,
            }
        )

        with st.chat_message("user"):
            st.markdown(question)

        # Generate the assistant's response.
        with st.chat_message("assistant"):
            try:
                with st.spinner("Searching your document and generating an answer..."):
                    result = get_rag_answer(question, STORE_DIR)

                answer = result.get(
                    "answer",
                    "I could not generate an answer for this question.",
                )

                sources = result.get("sources", []) or []

                if not isinstance(answer, str):
                    answer = str(answer)

                st.markdown(answer)

                if sources:
                    render_sources(sources)
                else:
                    st.caption("No source passages were returned.")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    }
                )

            except Exception as exc:
                error_message = (
                    "I couldn't generate an answer. Please check that "
                    "your document was processed correctly and that "
                    "Ollama is running with the configured model."
                )

                st.error(error_message)
                st.caption(f"Technical details: {exc}")

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": error_message,
                        "sources": [],
                    }
                )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "DocuMind AI · Retrieval-Augmented Generation (RAG) · "
    "Local document question answering"
)
