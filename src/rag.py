
from src.embeddings import embed_texts
from src.generation import DEFAULT_LLM_MODEL, generate_answer
from src.vector_stores import search_index


def answer_question(
    question,
    index,
    chunks,
    embedding_model,
    top_k=5,
    llm_model=DEFAULT_LLM_MODEL,
    chat_fn=None,
):
    """Retrieve relevant passages and generate a grounded answer."""

    if not isinstance(question, str) or not question.strip():
        raise ValueError("Question must not be empty.")

    if top_k <= 0:
        raise ValueError("top_k must be positive.")

    query_vector = embed_texts([question], embedding_model)[0]

    retrieved_chunks = search_index(
        index=index,
        chunks=chunks,
        query_vector=query_vector,
        top_k=top_k,
    )

    return generate_answer(
        question=question,
        retrieved_chunks=retrieved_chunks,
        model_name=llm_model,
        chat_fn=chat_fn,
    )
