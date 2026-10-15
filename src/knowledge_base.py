
import json
from pathlib import Path

from src.document_loader import load_pdf
from src.text_splitter import chunk_pages
from src.embeddings import load_embedding_model, embed_texts
from src.vector_stores import (
    create_index,
    save_knowledge_base,
    load_knowledge_base,
    search_index,
)


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def ingest_pdf(pdf_path: str, store_dir: str) -> int:
    """Extract, chunk, embed, and persist a PDF."""

    pages = load_pdf(pdf_path)
    chunks = chunk_pages(pages)

    if not chunks:
        raise ValueError("The PDF produced no searchable chunks.")

    model = load_embedding_model()

    vectors = embed_texts(
        [chunk["text"] for chunk in chunks],
        model,
    )

    index = create_index(vectors)

    save_knowledge_base(index, chunks, store_dir)

    # Record which embedding model created this index.
    manifest = {
        "embedding_model": MODEL_NAME,
        "chunk_count": len(chunks),
        "source_file": Path(pdf_path).name,
    }

    manifest_path = Path(store_dir) / "manifest.json"
    temp_path = Path(store_dir) / "manifest.json.tmp"

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    temp_path.replace(manifest_path)

    return len(chunks)


def ask_knowledge_base(
    question: str,
    store_dir: str,
    top_k: int = 3,
) -> list[dict]:
    """Retrieve relevant passages from a saved knowledge base."""

    if not question.strip():
        raise ValueError("Question must not be empty.")

    store_path = Path(store_dir)
    manifest_path = store_path / "manifest.json"

    if not manifest_path.is_file():
        raise FileNotFoundError(
            "Knowledge base manifest not found. Ingest a PDF first."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("embedding_model") != MODEL_NAME:
        raise ValueError(
            "Embedding model does not match the saved knowledge base."
        )

    index, chunks = load_knowledge_base(store_dir)
    model = load_embedding_model()

    query_vector = embed_texts([question], model)[0]

    return search_index(
        index=index,
        chunks=chunks,
        query_vector=query_vector,
        top_k=top_k,
    )
