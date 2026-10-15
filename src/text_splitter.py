
def chunk_pages(
    pages: list[dict],
    chunk_size: int = 1000,
    overlap: int = 150,
) -> list[dict]:
    """
    Split extracted PDF pages into smaller text chunks.

    Preserve source and page metadata for citations.
    """

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be >= 0 and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        text = page["text"].strip()

        if not text:
            continue

        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))

            # Prefer a natural boundary near the target end.
            if end < len(text):
                boundary = max(
                    text.rfind("\n", start, end),
                    text.rfind(" ", start, end),
                )

                # Avoid creating very small chunks.
                if boundary > start + chunk_size // 2:
                    end = boundary

            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append({
                    "text": chunk_text,
                    "source": page["source"],
                    "page_number": page["page_number"],
                    "chunk_id": len(chunks),
                })

            # Advance by the normal step, unless a natural
            # boundary moved the end earlier.
            start = max(start + step, end - overlap)

    return chunks
