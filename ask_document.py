
import argparse
import json
from pathlib import Path

from src.embeddings import load_embedding_model, embed_texts
from vector_stores import load_knowledge_base, search_index


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def main():
    parser = argparse.ArgumentParser(
        description="Search an existing document knowledge base."
    )
    parser.add_argument(
        "--store",
        default="data/knowledge_base",
        help="Directory containing the saved knowledge base",
    )
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be positive.")

    store_path = Path(args.store)
    manifest_path = store_path / "manifest.json"

    if not manifest_path.is_file():
        parser.error("Knowledge base not found. Run ingestion first.")

    try:
        with manifest_path.open("r", encoding="utf-8") as file:
            manifest = json.load(file)

        if manifest.get("embedding_model") != MODEL_NAME:
            parser.error("Embedding model does not match the saved index.")

        index, chunks = load_knowledge_base(args.store)

        if index.ntotal == 0:
            parser.error("Knowledge base contains no vectors.")

        model = load_embedding_model()

    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))

    print(f"Loaded {len(chunks)} chunks. Ask your questions.")
    print("Type 'exit' to stop.")

    while True:
        question = input("\nQuestion: ").strip()

        if question.lower() == "exit":
            break

        if not question:
            print("Please enter a question.")
            continue

        query_vector = embed_texts([question], model)[0]

        results = search_index(
            index=index,
            chunks=chunks,
            query_vector=query_vector,
            top_k=args.top_k,
        )

        for result in results:
            print(f"\nSource: {result['source']}")
            print(f"Page: {result['page_number']}")
            print(f"Similarity: {result['similarity_score']:.4f}")
            print(result["text"])
            print("-" * 50)


if __name__ == "__main__":
    main()
