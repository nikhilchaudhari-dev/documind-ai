
import argparse
from pathlib import Path

from src.knowledge_base import ingest_pdf


def main():
    parser = argparse.ArgumentParser(
        description="Build a searchable knowledge base from a PDF."
    )
    parser.add_argument("pdf", help="Path to the input PDF")
    parser.add_argument(
        "--store",
        default="data/knowledge_base",
        help="Directory for the saved knowledge base",
    )
    args = parser.parse_args()

    pdf_path = Path(args.pdf)

    if not pdf_path.is_file():
        parser.error(f"PDF file not found: {pdf_path}")

    try:
        count = ingest_pdf(str(pdf_path), args.store)
        print(f"Successfully indexed {count} chunks.")
        print(f"Knowledge base saved to: {args.store}")

    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
