
from src.embeddings import load_embedding_model
from src.retrieval import search_chunks

chunks = [
    {
        "text": (
            "Overfitting happens when a model learns the training "
            "examples too closely and performs poorly on new data."
        ),
        "source": "ml_notes.pdf",
        "page_number": 5,
    },
    {
        "text": (
            "Gradient descent updates model parameters to reduce "
            "the loss function during training."
        ),
        "source": "ml_notes.pdf",
        "page_number": 8,
    },
    {
        "text": (
            "A database index can speed up queries by reducing "
            "the amount of data that must be searched."
        ),
        "source": "database_notes.pdf",
        "page_number": 3,
    },
]

model = load_embedding_model()

results = search_chunks(
    chunks=chunks,
    query="Why does my model fail on unseen examples?",
    model=model,
    top_k=2,
)

for result in results:
    print(f"Source: {result['source']}")
    print(f"Page: {result['page_number']}")
    print(f"Similarity: {result['similarity_score']:.4f}")
    print(f"Text: {result['text']}")
    print("-" * 50)
