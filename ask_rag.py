
import argparse
import json
from pathlib import Path

from src.embeddings import MODEL_NAME, load_embedding_model
from src.rag import answer_question
from src.vector_stores import load_knowledge_base


def main():
    parser = argparse.ArgumentParser(
        description="Ask questions about your PDF using local RAG."
    )
    parser.add_argument(
        "--store",
        default="data/knowledge_base",
        help="Directory containing the saved knowledge base.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of document passages to retrieve.",
    )
    parser.add_argument(
        "--model",
        default="llama3.2:3b",
        help="Ollama model to use for answer generation.",
    )
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero.")

    store_path = Path(args.store)
    manifest_path = store_path / "manifest.json"

    if not manifest_path.is_file():
        raise SystemExit(
            f"Manifest not found: {manifest_path}. "
            "Ingest a PDF before asking questions."
        )

    with manifest_path.open("r", encoding="utf-8") as file:
        manifest = json.load(file)

    if manifest.get("embedding_model") != MODEL_NAME:
        raise SystemExit(
            "The knowledge base embedding model does not match "
            "the model configured in src/embeddings.py."
        )

    index, chunks = load_knowledge_base(str(store_path))

    if index.ntotal == 0:
        raise SystemExit("The knowledge base contains no vectors.")

    print(f"Loaded {len(chunks)} document chunks.")
    print(f"Answer model: {args.model}")
    print("Ask a question, or type 'exit' to quit.\n")

    embedding_model = load_embedding_model()

    while True:
        try:
            question = input("Question: ").strip()

            if question.lower() in {"exit", "quit"}:
                break

            if not question:
                continue

            result = answer_question(
                question=question,
                index=index,
                chunks=chunks,
                embedding_model=embedding_model,
                top_k=args.top_k,
                llm_model=args.model,
            )

            print(f"\nAnswer: {result['answer']}")

            if result["sources"]:
                print("\nSources:")
                for source in result["sources"]:
                    print(
                        f"  {source['citation']} "
                        f"{source['source']} — "
                        f"Page {source['page_number']}"
                    )
                    print(f"  Passage: {source['text'][:400]}\n")

            print("-" * 60)

        except KeyboardInterrupt:
            print("\nExiting.")
            break
        except Exception as error:
            print(f"\nError: {error}")
            print(
                "Check that Ollama is running and the requested "
                "model is installed."
            )


if __name__ == "__main__":
    main()
