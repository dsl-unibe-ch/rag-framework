import chromadb
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from config.embedding_config import (
    use_hybrid_search as _cfg_use_hybrid,
    hybrid_rrf_k as _cfg_rrf_k,
    hybrid_candidates as _cfg_hybrid_candidates,
)

# Module-level cache so each unique model name is loaded only once across
# all requests in the same Django worker process.
_ST_MODEL_CACHE: dict[str, SentenceTransformer] = {}

# ---------------------------------------------------------------------------
# BM25 helpers for hybrid search
# ---------------------------------------------------------------------------
# Cache keyed by (db_path, collection_name) so each collection gets its own
# index.  The cache is invalidated whenever the document count changes.
_BM25_CACHE: dict[tuple, dict] = {}


def _tokenize(text: str) -> list[str]:
    """Lower-case whitespace tokenizer used for BM25 indexing and querying."""
    return text.lower().split()


def _get_bm25_index(collection, cache_key: tuple) -> "dict | None":
    """Return a cached BM25 index for *collection*, rebuilding when stale.

    The index is rebuilt whenever the collection's document count changes
    (i.e. after a re-index run).  Returns ``None`` if ``rank_bm25`` is not
    installed, so callers can fall back to pure vector search gracefully.

    Args:
        collection: The ChromaDB collection object.
        cache_key: A ``(db_path, collection_name)`` tuple used as the
            dictionary key.

    Returns:
        A dict with keys ``index`` (BM25Okapi), ``ids``, ``docs``,
        ``metas``, and ``count``; or ``None`` on failure.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("[Hybrid] rank-bm25 not installed; falling back to vector-only search.")
        return None

    current_count = collection.count()
    cached = _BM25_CACHE.get(cache_key)
    if cached is not None and cached["count"] == current_count:
        return cached

    try:
        result = collection.get(include=["documents", "metadatas"])
        ids   = result["ids"]   or []
        docs  = result["documents"]  or []
        metas = result["metadatas"] or []
    except Exception as exc:
        print(f"[Hybrid] Failed to fetch documents for BM25: {exc}")
        return None

    if not docs:
        return None

    tokenized = [_tokenize(d) for d in docs]
    index = BM25Okapi(tokenized)

    entry = {"count": current_count, "index": index, "ids": ids, "docs": docs, "metas": metas}
    _BM25_CACHE[cache_key] = entry
    return entry


def _rrf_fuse(
    vec_ids:   list,
    vec_docs:  list,
    vec_metas: list,
    vec_dists: list,
    bm25_ids:  list,
    bm25_docs: list,
    bm25_metas: list,
    top_n: int,
    k: int = 60,
) -> dict:
    """Merge vector and BM25 ranked lists with Reciprocal Rank Fusion.

    RRF score for document d:
        score(d) = sum_i  1 / (k + rank_i(d))
    where rank_i(d) is d's 0-based position in ranked list i.

    Args:
        vec_ids / vec_docs / vec_metas / vec_dists: Vector-search candidates
            in order (most similar first).
        bm25_ids / bm25_docs / bm25_metas: BM25 candidates in order
            (highest score first).
        top_n: Number of results to return.
        k: RRF smoothing constant (default 60).

    Returns:
        A ChromaDB-style result dict with keys ``ids``, ``documents``,
        ``metadatas``, ``distances`` — each a single-element list to match
        the ``collection.query()`` output format.
    """
    vec_rank  = {id_: rank for rank, id_ in enumerate(vec_ids)}
    bm25_rank = {id_: rank for rank, id_ in enumerate(bm25_ids)}

    # Build lookup tables from both candidate pools.
    id_to_doc:      dict = {}
    id_to_meta:     dict = {}
    id_to_vec_dist: dict = {}

    for id_, doc, meta, dist in zip(vec_ids, vec_docs, vec_metas, vec_dists):
        id_to_doc[id_]      = doc
        id_to_meta[id_]     = meta
        id_to_vec_dist[id_] = dist

    for id_, doc, meta in zip(bm25_ids, bm25_docs, bm25_metas):
        if id_ not in id_to_doc:
            id_to_doc[id_]  = doc
            id_to_meta[id_] = meta

    all_ids = set(vec_rank) | set(bm25_rank)
    rrf_scores = {
        id_: (1.0 / (k + vec_rank[id_])  if id_ in vec_rank  else 0.0) +
             (1.0 / (k + bm25_rank[id_]) if id_ in bm25_rank else 0.0)
        for id_ in all_ids
    }

    ranked = sorted(all_ids, key=lambda x: rrf_scores[x], reverse=True)[:top_n]

    return {
        "ids":       [ranked],
        "documents": [[id_to_doc.get(i, "")  for i in ranked]],
        "metadatas": [[id_to_meta.get(i, {}) for i in ranked]],
        # Use the vector distance when available; BM25-only hits get 0.0.
        "distances": [[id_to_vec_dist.get(i, 0.0) for i in ranked]],
    }

class ChromaRetriever:
    """
    A class for retrieving documents from a ChromaDB collection based on semantic similarity using embeddings.
    """
    def __init__(self, embedding_model: str, db_path: str, db_collection: str, n_results: int) -> None:
        self.embedding_model = embedding_model
        self.db_path = db_path
        self.db_collection = db_collection
        self.n_results = n_results
        if self.embedding_model not in _ST_MODEL_CACHE:
            _ST_MODEL_CACHE[self.embedding_model] = SentenceTransformer(
                self.embedding_model, trust_remote_code=True
            )
        self.model = _ST_MODEL_CACHE[self.embedding_model]
        self.client = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client.get_collection(name=self.db_collection)

    def retrieve(self, query: str, *, embed_text: str | None = None, use_hybrid: "bool | None" = None):
        """Embed the query and retrieve relevant documents from the collection.

        Args:
            query: The user's original question.
            embed_text: Optional text to embed instead of *query*.  Pass the
                HyDE-generated hypothetical document here when HyDE is active.
                Falls back to *query* when ``None``.
            use_hybrid: Override the ``use_hybrid_search`` config flag for this
                request.  ``None`` (default) uses the config value.
        """
        _use_hybrid = use_hybrid if use_hybrid is not None else _cfg_use_hybrid
        try:
            text_to_embed = embed_text if embed_text else query
            embedded_query = self.model.encode(text_to_embed).tolist()

            if _use_hybrid:
                candidates = max(self.n_results, self.n_results * _cfg_hybrid_candidates)
                safe_candidates = min(candidates, self.collection.count() or 1)
                vec_results = self.collection.query(
                    query_embeddings=[embedded_query],
                    n_results=safe_candidates,
                )
                bm25_data = _get_bm25_index(
                    self.collection, (self.db_path, self.db_collection)
                )
                if bm25_data is not None:
                    scores = bm25_data["index"].get_scores(_tokenize(query))
                    ranked_idx = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[:safe_candidates]
                    return _rrf_fuse(
                        vec_ids=vec_results["ids"][0],
                        vec_docs=vec_results["documents"][0],
                        vec_metas=vec_results["metadatas"][0],
                        vec_dists=vec_results["distances"][0],
                        bm25_ids=[bm25_data["ids"][i]   for i in ranked_idx],
                        bm25_docs=[bm25_data["docs"][i]  for i in ranked_idx],
                        bm25_metas=[bm25_data["metas"][i] for i in ranked_idx],
                        top_n=self.n_results,
                        k=_cfg_rrf_k,
                    )
                # rank_bm25 not available — fall through to plain vector search.

            results = self.collection.query(
                query_embeddings=[embedded_query],
                n_results=self.n_results,
            )
            return results
        except Exception as e:
            print(f"An error occurred during retrieval: {e}")
            return None
        

    def format_results_for_prompt(self, results):
        """Format retrieval results into a string for the LLM prompt.

        Includes provenance metadata (page, section, source URL, ingest date)
        when available.  Fields are omitted when not present in the stored
        metadata so older chunks without rich metadata still render correctly.

        Args:
            results: The dictionary returned by the retrieve method.

        Returns:
            A formatted string containing the retrieved data.
        """
        if not results:
            return "No relevant data found."

        formatted_data = ""
        for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted_data += f"Document {idx + 1}:\n"
            formatted_data += f"  File      : {metadata.get('file_name', 'N/A')}\n"
            formatted_data += f"  Chunk ID  : {metadata.get('chunk_id', 'N/A')}\n"
            if "page_number" in metadata:
                formatted_data += f"  Page      : {metadata['page_number']}\n"
            if "section_title" in metadata:
                formatted_data += f"  Section   : {metadata['section_title']}\n"
            if metadata.get("source_url"):
                formatted_data += f"  Source URL: {metadata['source_url']}\n"
            if "ingest_date" in metadata:
                formatted_data += f"  Indexed   : {metadata['ingest_date']}\n"
            formatted_data += f"  Content:\n{doc}\n"
            formatted_data += "-" * 80 + "\n"

        return formatted_data

    
class OpenAIChromaRetriever:
    """
    A class for retrieving documents from a ChromaDB collection based on semantic similarity using embeddings. Uses OpenAI API for embeddings.
    """
    def __init__(self, openai_client: OpenAI, embedding_model: str, db_path: str, db_collection: str, n_results: int) -> None:
        """
        Args:
            openai_client: The initialized OpenAI client object.
            embedding_model: The name of the embedding model to use (e.g., 'text-embedding-3-small').
            db_path: Path to ChromaDB.
            db_collection: Name of the collection.
            n_results: Number of results to return.
        """
        self.db_path = db_path
        self.db_collection = db_collection
        self.n_results = n_results
        
        # Dependency Injection
        self.client_openai = openai_client
        self.model_name = embedding_model
        
        # Initialize ChromaDB
        self.client_db = chromadb.PersistentClient(path=self.db_path)
        self.collection = self.client_db.get_collection(name=self.db_collection)

    def retrieve(self, query: str, *, embed_text: str | None = None, use_hybrid: "bool | None" = None):
        """Embed the query with the OpenAI API and retrieve relevant documents.

        Args:
            query: The user's original question.
            embed_text: Optional text to embed instead of *query*.  Pass the
                HyDE-generated hypothetical document here when HyDE is active.
                Falls back to *query* when ``None``.
            use_hybrid: Override the ``use_hybrid_search`` config flag for this
                request.  ``None`` (default) uses the config value.
        """
        _use_hybrid = use_hybrid if use_hybrid is not None else _cfg_use_hybrid
        try:
            text_to_embed = embed_text if embed_text else query
            clean_embed = text_to_embed.replace("\n", " ")

            response = self.client_openai.embeddings.create(
                model=self.model_name,
                input=[clean_embed],
            )
            embedded_query = response.data[0].embedding

            if _use_hybrid:
                candidates = max(self.n_results, self.n_results * _cfg_hybrid_candidates)
                safe_candidates = min(candidates, self.collection.count() or 1)
                vec_results = self.collection.query(
                    query_embeddings=[embedded_query],
                    n_results=safe_candidates,
                )
                bm25_data = _get_bm25_index(
                    self.collection, (self.db_path, self.db_collection)
                )
                if bm25_data is not None:
                    scores = bm25_data["index"].get_scores(_tokenize(query))
                    ranked_idx = sorted(
                        range(len(scores)), key=lambda i: scores[i], reverse=True
                    )[:safe_candidates]
                    return _rrf_fuse(
                        vec_ids=vec_results["ids"][0],
                        vec_docs=vec_results["documents"][0],
                        vec_metas=vec_results["metadatas"][0],
                        vec_dists=vec_results["distances"][0],
                        bm25_ids=[bm25_data["ids"][i]   for i in ranked_idx],
                        bm25_docs=[bm25_data["docs"][i]  for i in ranked_idx],
                        bm25_metas=[bm25_data["metas"][i] for i in ranked_idx],
                        top_n=self.n_results,
                        k=_cfg_rrf_k,
                    )
                # rank_bm25 not available — fall through to plain vector.

            results = self.collection.query(
                query_embeddings=[embedded_query],
                n_results=self.n_results,
            )
            return results
        except Exception as e:
            print(f"An error occurred during OpenAI retrieval: {e}")
            return None

    def format_results_for_prompt(self, results):
        """Format retrieval results into a string for the LLM prompt.

        Includes provenance metadata (page, section, source URL, ingest date)
        when available.  Fields are omitted when not present in the stored
        metadata so older chunks without rich metadata still render correctly.

        Args:
            results: The dictionary returned by the retrieve method.

        Returns:
            A formatted string containing the retrieved data.
        """
        if not results or not results['documents']:
            return "No relevant data found."
        if len(results['documents'][0]) == 0:
            return "No relevant data found."

        formatted_data = ""
        for idx, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
            formatted_data += f"Document {idx + 1}:\n"
            formatted_data += f"  File      : {metadata.get('file_name', 'N/A')}\n"
            formatted_data += f"  Chunk ID  : {metadata.get('chunk_id', 'N/A')}\n"
            if "page_number" in metadata:
                formatted_data += f"  Page      : {metadata['page_number']}\n"
            if "section_title" in metadata:
                formatted_data += f"  Section   : {metadata['section_title']}\n"
            if metadata.get("source_url"):
                formatted_data += f"  Source URL: {metadata['source_url']}\n"
            if "ingest_date" in metadata:
                formatted_data += f"  Indexed   : {metadata['ingest_date']}\n"
            formatted_data += f"  Content:\n{doc}\n"
            formatted_data += "-" * 80 + "\n"

        return formatted_data


    