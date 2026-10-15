
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_embedding_model():
    """Load the local text embedding model."""
    return SentenceTransformer(MODEL_NAME)


def embed_texts(texts: list[str], model) -> list[list[float]]:
    """Convert text into normalized embedding vectors."""

    if not texts:
        return []

    if any(not text.strip() for text in texts):
        raise ValueError("Texts must not contain empty strings.")

    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=False,
    )

    return vectors.tolist()
