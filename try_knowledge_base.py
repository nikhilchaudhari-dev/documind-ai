
from src.knowledge_base import ingest_pdf, ask_knowledge_base

PDF_PATH = "sample.pdf"
STORE_DIR = "data/knowledge_base"

# Run ingestion once for this PDF.
count = ingest_pdf(PDF_PATH, STORE_DIR)
print(f"Indexed {count} chunks.")

# Ask questions without re-ingesting the PDF.
while True:
    question = input("\nAsk about your document (or type exit): ")

    if question.strip().lower() == "exit":
        break

    try:
        results = ask_knowledge_base(
            question,
            STORE_DIR,
            top_k=3,
        )

        for result in results:
            print(f"\nSource: {result['source']}")
            print(f"Page: {result['page_number']}")
            print(
                f"Similarity: {result['similarity_score']:.4f}"
            )
            print(result["text"])
            print("-" * 40)

    except (ValueError, FileNotFoundError) as error:
        print(f"Could not answer: {error}")
