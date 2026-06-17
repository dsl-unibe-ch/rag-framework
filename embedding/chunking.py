"""Text chunking strategies for the RAG framework.

This module groups a document's sentences into chunks that are later
embedded and stored in the vector database. Three strategies are provided:

* ``sentence`` -- the original rule-based approach. Sentences are grouped
  into fixed-size, overlapping windows. Fast and deterministic, but it
  ignores the meaning of the text and can split a single idea across two
  chunks.
* ``semantic`` -- an embedding-based approach. Each sentence is embedded
  and a new chunk is started whenever the meaning shifts (i.e. the
  similarity between consecutive sentences drops below a data-driven
  threshold). This keeps semantically related sentences together.
* ``token`` -- token-budget chunking. Sentences are greedily accumulated
  until the chunk would exceed a target token count (e.g. 512), then a
  new chunk is started. An optional token-level overlap carries context
  across boundaries. Produces predictably sized chunks regardless of
  sentence length variance.

Use :func:`create_chunks` as the single entry point; it dispatches to the
correct strategy based on ``method``.
"""

from typing import Callable, List, Sequence

import numpy as np

from embedding.utils import chunk_sentences

# A callable that turns a list of texts into a list of embedding vectors.
EmbedFn = Callable[[List[str]], Sequence[Sequence[float]]]

# Allowed chunking strategy names.
SENTENCE_METHOD = "sentence"
SEMANTIC_METHOD = "semantic"
TOKEN_METHOD = "token"
ALLOWED_METHODS = (SENTENCE_METHOD, SEMANTIC_METHOD, TOKEN_METHOD)


def create_chunks(
    method: str,
    sentences: List[str],
    *,
    chunk_size: int,
    overlap_size: int,
    embed_fn: EmbedFn = None,
    breakpoint_percentile: float = 95.0,
    buffer_size: int = 1,
    max_chunk_sentences: int = 0,
    token_chunk_size: int = 512,
    token_chunk_overlap: int = 50,
    token_encoding: str = "cl100k_base",
    return_indices: bool = False,
) -> List:
    """Group sentences into chunks using the requested strategy.

    Args:
        method: The chunking strategy to use. One of ``ALLOWED_METHODS``.
        sentences: The document's sentences, in reading order.
        chunk_size: Sentences per chunk (``sentence`` strategy only).
        overlap_size: Overlapping sentences between consecutive chunks
            (``sentence`` strategy only).
        embed_fn: A callable mapping a list of texts to their embedding
            vectors. Required for the ``semantic`` strategy.
        breakpoint_percentile: Percentile (0-100) of consecutive-sentence
            distances used as the split threshold (``semantic`` only). A
            higher value yields fewer, larger chunks.
        buffer_size: Number of neighbouring sentences combined with each
            sentence to give it context before embedding (``semantic``
            only).
        max_chunk_sentences: Optional hard cap on the number of sentences
            in a semantic chunk. ``0`` disables the cap.
        token_chunk_size: Target token budget per chunk (``token`` strategy
            only).  Sentences are greedily added until this budget would be
            exceeded. Defaults to ``512``.
        token_chunk_overlap: Number of tokens from the end of one chunk to
            repeat at the start of the next (``token`` strategy only).
            ``0`` disables overlap.  Defaults to ``50``.
        token_encoding: tiktoken encoding name used to count tokens
            (``token`` strategy only). Defaults to ``"cl100k_base"``.
        return_indices: When ``True``, return a list of
            ``(chunk_text, first_sentence_index)`` tuples instead of plain
            strings.  The index refers to the position of the first sentence
            in the chunk within the input ``sentences`` list.

    Returns:
        A list of chunk strings, or a list of ``(str, int)`` tuples when
        ``return_indices`` is ``True``.

    Raises:
        ValueError: If ``method`` is unknown, or if the ``semantic``
            strategy is selected without an ``embed_fn``.
    """
    if method == SENTENCE_METHOD:
        chunks = chunk_sentences(sentences, chunk_size, overlap_size)
        if return_indices:
            step = max(1, chunk_size - overlap_size)
            return [(text, k * step) for k, text in enumerate(chunks)]
        return chunks

    if method == SEMANTIC_METHOD:
        if embed_fn is None:
            raise ValueError(
                "The 'semantic' chunking method requires an 'embed_fn'."
            )
        return chunk_sentences_semantically(
            sentences,
            embed_fn,
            breakpoint_percentile=breakpoint_percentile,
            buffer_size=buffer_size,
            max_chunk_sentences=max_chunk_sentences,
            return_first_indices=return_indices,
        )

    if method == TOKEN_METHOD:
        return chunk_by_tokens(
            sentences,
            max_tokens=token_chunk_size,
            overlap_tokens=token_chunk_overlap,
            encoding_name=token_encoding,
            return_first_indices=return_indices,
        )

    raise ValueError(
        f"Unknown chunking method '{method}'. "
        f"Allowed values are: {list(ALLOWED_METHODS)}."
    )


def chunk_sentences_semantically(
    sentences: List[str],
    embed_fn: EmbedFn,
    breakpoint_percentile: float = 95.0,
    buffer_size: int = 1,
    max_chunk_sentences: int = 0,
    return_first_indices: bool = False,
) -> List:
    """Group sentences into semantically coherent chunks.

    The algorithm embeds every sentence (optionally with a small window of
    surrounding context), measures the cosine distance between consecutive
    sentence embeddings, and starts a new chunk wherever that distance
    exceeds a percentile-based threshold. Large jumps in distance signal a
    change of topic and therefore a natural chunk boundary.

    Args:
        sentences: The document's sentences, in reading order.
        embed_fn: A callable mapping a list of texts to their embedding
            vectors.
        breakpoint_percentile: Percentile (0-100) of consecutive-sentence
            distances used as the split threshold. Higher means fewer,
            larger chunks.
        buffer_size: Number of neighbouring sentences combined with each
            sentence to give it context before embedding. ``0`` embeds
            each sentence on its own.
        max_chunk_sentences: Optional hard cap on the number of sentences
            per chunk. ``0`` disables the cap.
        return_first_indices: When ``True``, return a list of
            ``(chunk_text, first_sentence_index)`` tuples.

    Returns:
        A list of chunk strings, or ``(str, int)`` tuples when
        ``return_first_indices`` is ``True``.
    """
    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if len(cleaned) <= 1:
        result = cleaned
        if return_first_indices:
            return [(text, 0) for text in result]
        return result

    combined = _combine_sentences(cleaned, buffer_size)
    embeddings = np.asarray(list(embed_fn(combined)), dtype=float)
    distances = _consecutive_cosine_distances(embeddings)

    if distances.size == 0:
        chunk_text = " ".join(cleaned)
        return [(chunk_text, 0)] if return_first_indices else [chunk_text]

    threshold = float(np.percentile(distances, breakpoint_percentile))
    # Index i in ``distances`` measures the gap between sentence i and i+1,
    # so a value above the threshold means we split *after* sentence i.
    split_after = {
        i for i, distance in enumerate(distances) if distance > threshold
    }

    # raw_chunks holds (chunk_text, first_sentence_index) pairs.
    raw_chunks: List = []
    current: List[str] = []
    current_start: int = 0
    last_index = len(cleaned) - 1

    for i, sentence in enumerate(cleaned):
        if not current:
            current_start = i
        current.append(sentence)
        reached_cap = (
            max_chunk_sentences > 0 and len(current) >= max_chunk_sentences
        )
        if i in split_after or i == last_index or reached_cap:
            raw_chunks.append((" ".join(current), current_start))
            current = []

    if current:
        raw_chunks.append((" ".join(current), current_start))

    if return_first_indices:
        return raw_chunks
    return [text for text, _ in raw_chunks]


def _combine_sentences(sentences: List[str], buffer_size: int) -> List[str]:
    """Combine each sentence with ``buffer_size`` neighbours on each side.

    Embedding a sentence together with a little surrounding context makes
    the resulting vectors less noisy and the topic boundaries cleaner.

    Args:
        sentences: The sentences to combine.
        buffer_size: Number of neighbours to include on each side. ``0``
            returns the sentences unchanged.

    Returns:
        A list, the same length as ``sentences``, of context-enriched
        strings.
    """
    if buffer_size <= 0:
        return list(sentences)

    n = len(sentences)
    combined: List[str] = []
    for i in range(n):
        start = max(0, i - buffer_size)
        end = min(n, i + buffer_size + 1)
        combined.append(" ".join(sentences[start:end]))
    return combined


def _consecutive_cosine_distances(embeddings: np.ndarray) -> np.ndarray:
    """Compute cosine distance between each pair of adjacent embeddings.

    Args:
        embeddings: A ``(n, d)`` array of sentence embeddings.

    Returns:
        A ``(n - 1,)`` array where element ``i`` is ``1 - cosine_similarity``
        between embedding ``i`` and ``i + 1``.
    """
    if embeddings.shape[0] < 2:
        return np.empty(0, dtype=float)

    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms[norms == 0] = 1e-10
    normalized = embeddings / norms

    similarities = np.sum(normalized[:-1] * normalized[1:], axis=1)
    return 1.0 - similarities


def chunk_by_tokens(
    sentences: List[str],
    max_tokens: int = 512,
    overlap_tokens: int = 50,
    encoding_name: str = "cl100k_base",
    return_first_indices: bool = False,
) -> List:
    """Group sentences into chunks that stay within a target token budget.

    Sentences are accumulated greedily until adding the next sentence would
    exceed ``max_tokens``.  An optional token overlap carries context from
    the tail of one chunk into the start of the next, preventing information
    loss at hard boundaries.  A sentence that is individually longer than
    ``max_tokens`` is always kept as its own chunk to avoid data loss.

    Token counts are computed with `tiktoken`_ using the given encoding.
    ``cl100k_base`` is the encoding for GPT-4 / GPT-3.5-turbo and is a
    reasonable approximation for most modern LLMs.

    .. _tiktoken: https://github.com/openai/tiktoken

    Args:
        sentences: The document's sentences, in reading order.
        max_tokens: Maximum number of tokens per chunk.  Defaults to ``512``.
        overlap_tokens: Number of tokens from the end of the previous chunk
            to repeat at the start of the next chunk.  ``0`` disables
            overlap.  Defaults to ``50``.
        encoding_name: tiktoken encoding name used for token counting.
            Defaults to ``"cl100k_base"``.
        return_first_indices: When ``True``, return a list of
            ``(chunk_text, first_sentence_index)`` tuples.

    Returns:
        A list of chunk strings, or ``(str, int)`` tuples when
        ``return_first_indices`` is ``True``.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding_name)

        def count_tokens(text: str) -> int:
            return len(enc.encode(text))
    except Exception:
        # Graceful fallback: estimate 1.33 tokens per whitespace-separated word.
        def count_tokens(text: str) -> int:  # type: ignore[misc]
            return max(1, int(len(text.split()) * 1.33))

    cleaned = [s.strip() for s in sentences if s and s.strip()]
    if not cleaned:
        return []

    token_counts = [count_tokens(s) for s in cleaned]
    n = len(cleaned)
    raw_chunks: List = []  # list of (chunk_text, first_sentence_index)
    i = 0

    while i < n:
        start_idx = i
        current_tokens = 0
        j = i

        while j < n:
            t = token_counts[j]
            # Always include at least one sentence even if it exceeds the budget,
            # so a single over-long sentence does not cause an infinite loop.
            if current_tokens + t > max_tokens and j > i:
                break
            current_tokens += t
            j += 1

        # sentences[i:j] form the current chunk.
        raw_chunks.append((" ".join(cleaned[i:j]), start_idx))

        # Determine the start of the next chunk considering token overlap.
        if overlap_tokens > 0 and j > i + 1:
            # Walk backward from j-1 accumulating tokens until the overlap
            # budget is exhausted.  Never step back to i itself so that each
            # iteration always makes forward progress.
            overlap_acc = 0
            next_start = j
            for k in range(j - 1, i, -1):
                if overlap_acc + token_counts[k] <= overlap_tokens:
                    overlap_acc += token_counts[k]
                    next_start = k
                else:
                    break
            i = next_start
        else:
            i = j

    if return_first_indices:
        return raw_chunks
    return [text for text, _ in raw_chunks]
