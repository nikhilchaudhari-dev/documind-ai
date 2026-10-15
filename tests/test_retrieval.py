
import numpy as np
import pytest

from src.retrieval import search_chunks


class FakeEmbeddingModel:
    """Deterministic embeddings for testing retrieval logic."""

    def encode(
        self,
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    ):
        vectors = []

        for text in texts:
            if "overfitting" in text.lower():
                vectors.append([1.0, 0.0])
            else:
                vectors.append([0.0, 1.0])

        return np.array(vectors)


def test_retrieval_ranks_relevant_chunk_first():
    chunks = [
        {
            "text": "A recipe for vegetable soup.",
            "source": "notes.pdf",
            "page_number": 1,
        },
        {
            "text": "Overfitting occurs when a model memorizes training data.",
            "source": "notes.pdf",
            "page_number": 5,
        },
    ]

    results = search_chunks(
        chunks,
        "Explain overfitting",
        FakeEmbeddingModel(),
        top_k=1,
    )

    assert len(results) == 1
    assert "overfitting" in results[0]["text"].lower()
    assert results[0]["page_number"] == 5
    assert results[0]["similarity_score"] == pytest.approx(1.0)


def test_empty_chunks_return_empty_results():
    assert search_chunks(
        [],
        "Explain overfitting",
        FakeEmbeddingModel(),
    ) == []


def test_empty_query_is_rejected():
    with pytest.raises(ValueError):
        search_chunks([], "   ", FakeEmbeddingModel())


def test_invalid_top_k_is_rejected():
    with pytest.raises(ValueError):
        search_chunks([], "overfitting", FakeEmbeddingModel(), top_k=0)
