
import json
from pathlib import Path

import faiss
import numpy as np


def create_index(vectors: list[list[float]]):
    """Create a vector index from normalized embeddings."""

    if not vectors:
        raise ValueError("Cannot create an index without vectors.")

    matrix = np.asarray(vectors, dtype=np.float32)

    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Vectors must be a non-empty 2D matrix.")

    if not np.isfinite(matrix).all():
        raise ValueError("Vectors contain invalid numeric values.")

    norms = np.linalg.norm(matrix, axis=1)

    if not np.allclose(norms, 1.0, atol=1e-4):
        raise ValueError("Expected normalized embedding vectors.")

    index = faiss.IndexFlatIP(matrix.shape[1])
    index.add(matrix)

    return index


def save_knowledge_base(index, chunks: list[dict], directory: str):
    """Persist the FAISS index and its corresponding chunk metadata."""

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk counts do not match.")

    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)

    faiss.write_index(index, str(path / "index.faiss"))

    metadata_path = path / "chunks.json"
    temp_path = path / "chunks.json.tmp"

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(chunks, file, ensure_ascii=False, indent=2)

    temp_path.replace(metadata_path)


def load_knowledge_base(directory: str):
    """Load a saved index and its chunk metadata."""

    path = Path(directory)
    index_path = path / "index.faiss"
    metadata_path = path / "chunks.json"

    if not index_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(
            "Knowledge base not found. Ingest a PDF first."
        )

    index = faiss.read_index(str(index_path))

    with metadata_path.open("r", encoding="utf-8") as file:
        chunks = json.load(file)

    if index.ntotal != len(chunks):
        raise ValueError("Saved index and metadata are inconsistent.")

    return index, chunks


def search_index(
    index,
    chunks: list[dict],
    query_vector: list[float],
    top_k: int = 3,
) -> list[dict]:
    """Search a saved FAISS index using a normalized query vector."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not query_vector:
        raise ValueError("Query vector must not be empty.")

    if index.ntotal != len(chunks):
        raise ValueError("Index and chunk counts do not match.")

    if index.ntotal == 0:
        return []

    vector = np.asarray([query_vector], dtype=np.float32)

    if vector.shape[1] != index.d:
        raise ValueError("Query embedding dimension does not match index.")

    if not np.isfinite(vector).all():
        raise ValueError("Query vector contains invalid values.")

    norm = np.linalg.norm(vector)

    if not np.isclose(norm, 1.0, atol=1e-4):
        raise ValueError("Expected a normalized query vector.")

    scores, indices = index.search(
        vector,
        min(top_k, index.ntotal),
    )

    results = []

    for score, position in zip(scores[0], indices[0]):
        if position < 0:
            continue

        result = dict(chunks[int(position)])
        result["similarity_score"] = float(score)
        results.append(result)

    return results
