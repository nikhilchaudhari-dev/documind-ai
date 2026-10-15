
import numpy as np


def search_chunks(
    chunks: list[dict],
    query: str,
    model,
    top_k: int = 3,
) -> list[dict]:
    """Return document chunks ranked by semantic similarity."""

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    if not query.strip():
        raise ValueError("Query must not be empty.")

    if not chunks:
        return []

    chunk_texts = [chunk["text"] for chunk in chunks]

    # Chunks are embedded together for this prototype.
    chunk_vectors = model.encode(
        chunk_texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    # Embed the user's question.
    query_vector = model.encode(
        [query],
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )[0]

    # Normalized vectors: dot product equals cosine similarity.
    scores = np.asarray(chunk_vectors) @ np.asarray(query_vector)

    # Rank the chunks from most to least similar.
    ranked_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in ranked_indices:
        result = dict(chunks[int(index)])
        result["similarity_score"] = float(scores[index])
        results.append(result)

    return results
