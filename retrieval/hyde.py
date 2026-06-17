"""
HyDE - Hypothetical Document Embeddings.

Instead of embedding the raw user query, ask the LLM to draft a short
hypothetical answer first, then embed that text.  Documents in the corpus
are written in a declarative, answer-like style; a hypothetical answer is
stylistically much closer to them than a terse question is, which typically
improves retrieval quality.

Reference: Gao et al. (2022) "Precise Zero-Shot Dense Retrieval without
Relevance Labels" (https://arxiv.org/abs/2212.10496).
"""
import openai

_HYDE_SYSTEM_PROMPT = (
    "You are a helpful assistant. Given a question, write a short, factual "
    "passage (2-4 sentences) that directly answers the question as if it "
    "were an excerpt from a relevant document. "
    "Be specific and factual. Do not say 'I don't know'."
)


def generate_hypothetical_document(
    query: str,
    client: "openai.OpenAI",
    model: str,
    max_tokens: int = 200,
) -> str:
    """Generate a hypothetical answer document for *query* using the LLM.

    The generated text is used in place of the raw user question when
    computing the embedding sent to the vector database.  This is the core
    of the HyDE technique.

    Args:
        query: The original user question.
        client: An initialised ``openai.OpenAI``-compatible client.
        model: LLM model name to use for hypothesis generation.
        max_tokens: Upper token limit for the hypothetical document.

    Returns:
        The hypothetical document text, or the original *query* as a
        graceful fallback if generation fails.
    """
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
            temperature=0.7,
            stream=False,
        )
        return response.choices[0].message.content.strip()
    except Exception as exc:
        # Graceful degradation: fall back to the original query so retrieval
        # still proceeds even if the LLM call fails.
        print(f"[HyDE] Failed to generate hypothetical document: {exc}")
        return query


def generate_hypothetical_document_ollama(
    query: str,
    model: str,
    max_tokens: int = 200,
) -> str:
    """Generate a hypothetical answer document for *query* using a local Ollama model.

    This is the Ollama-backed counterpart of
    :func:`generate_hypothetical_document`.  No API key or base URL is
    required — the Ollama daemon on localhost is used directly.

    Args:
        query: The original user question.
        model: Ollama model name (e.g. ``"deepseek-r1:1.5b"``).
        max_tokens: Soft upper token limit passed via Ollama options.

    Returns:
        The hypothetical document text, or the original *query* as a
        graceful fallback if generation fails.
    """
    try:
        import ollama  # local import so the rest of the module never breaks

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": _HYDE_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            options={"num_predict": max_tokens},
        )
        return response["message"]["content"].strip()
    except Exception as exc:
        print(f"[HyDE] Ollama generation failed: {exc}")
        return query
