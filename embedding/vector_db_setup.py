"""Vector database setup script with incremental indexing.

Run this script to build or update the ChromaDB vector database from the
raw documents in ``raw_db``.  On every run the script compares each source
file against a manifest to decide what action to take:

* **Skip** -- file content and chunking config are unchanged since the last
  run.  No embedding calls are made for this file.
* **Re-index** -- file content has changed, or the chunking configuration
  has been updated.  Old chunks are deleted and new ones are embedded and
  stored.
* **Index** -- new file that has never been seen before.
* **Remove** -- file was deleted from ``raw_db`` since the last run.  Its
  chunks are deleted from ChromaDB.

The manifest is stored at ``{db_directory}/{collection_name}_manifest.json``
alongside the ChromaDB data.
"""

import sys
import os
import chromadb
from datetime import datetime, timezone
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from config.embedding_config import (
    model_name,
    vector_db,
    raw_db,
    data_language,
    db_directory,
    chunk_size,
    overlap_size,
    collection_name,
    use_openai_embeddings,
    openai_embedding_model,
    openai_embedding_base_url,
    chunking_method,
    semantic_breakpoint_percentile,
    semantic_buffer_size,
    semantic_max_chunk_sentences,
    source_url as default_source_url,
)

from embedding.utils import (
    get_file_paths,
    read_document_sentences,
)
from embedding.chunking import create_chunks
from embedding.indexer import (
    get_manifest_path,
    load_manifest,
    save_manifest,
    compute_file_hash,
    compute_config_fingerprint,
    needs_reindex,
    get_stale_files,
    make_manifest_entry,
    load_sidecar_metadata,
)

# Load environment variables (for OPENAI_API_KEY)
load_dotenv(os.path.join(parent_dir, '.env'))

# Initialize ChromaDB client
client = chromadb.PersistentClient(path=db_directory)


def build_embedder(openai_client, embedding_model):
    """Build a function that embeds a list of texts into vectors.

    The returned callable hides whether embeddings come from an OpenAI
    compatible API or a local SentenceTransformer model, so the rest of
    the pipeline (semantic chunking and chunk storage) can stay agnostic
    of the backend.

    Args:
        openai_client: An initialized OpenAI client, or ``None`` when using
            a local model.
        embedding_model: A loaded SentenceTransformer, or ``None`` when
            using the OpenAI API.

    Returns:
        A callable mapping ``list[str]`` to ``list[list[float]]``.
    """
    def embed_texts(texts):
        if use_openai_embeddings:
            # It is good practice to replace newlines for embeddings.
            cleaned = [text.replace("\n", " ") for text in texts]
            response = openai_client.embeddings.create(
                model=openai_embedding_model,
                input=cleaned,
            )
            return [item.embedding for item in response.data]
        # Local SentenceTransformer call; convert to plain lists for ChromaDB.
        return embedding_model.encode(texts).tolist()

    return embed_texts


def delete_file_chunks(collection, chunk_ids: list[str], file_label: str) -> None:
    """Delete a file's chunks from ChromaDB, ignoring missing IDs.

    Args:
        collection: The ChromaDB collection object.
        chunk_ids: List of chunk IDs to delete.
        file_label: Human-readable label used in log output.
    """
    if not chunk_ids:
        return
    try:
        collection.delete(ids=chunk_ids)
    except Exception as e:
        print(f"  Warning: could not delete old chunks for '{file_label}': {e}")


def index_file(
    file_path: str,
    collection,
    embed_texts,
) -> list[str]:
    """Read, chunk, embed, and store a single document with rich metadata.

    Each stored chunk includes:
    - ``file_name``: basename of the source file (always present).
    - ``chunk_id``: sequential integer index within the file (always present).
    - ``ingest_date``: ISO-8601 UTC timestamp of this indexing run (always present).
    - ``source_url``: from a per-file ``.meta.json`` sidecar or the global
      config ``source_url`` setting; omitted when empty.
    - ``page_number``: 1-indexed page number (PDF only).
    - ``section_title``: heading under which the chunk appears (Markdown,
      HTML, DOCX only).

    Args:
        file_path: Absolute path to the source file.
        collection: The ChromaDB collection to write into.
        embed_texts: Callable that maps a list of strings to embeddings.

    Returns:
        The list of chunk IDs that were written, in order.  Returns an empty
        list if the file could not be processed.
    """
    # Load sentences with per-sentence page/section metadata.
    try:
        sentences_data = read_document_sentences(file_path, data_language)
    except ValueError as exc:
        print(f"  Unsupported file type: {exc}")
        return []
    except Exception as exc:
        print(f"  Error reading '{os.path.basename(file_path)}': {exc}")
        return []

    if not sentences_data:
        print(f"  No text extracted from '{os.path.basename(file_path)}'.")
        return []

    sentence_texts = [s["text"] for s in sentences_data]

    # Chunk with indices so we can look up the first sentence's metadata.
    chunk_pairs = create_chunks(
        chunking_method,
        sentence_texts,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        embed_fn=embed_texts,
        breakpoint_percentile=semantic_breakpoint_percentile,
        buffer_size=semantic_buffer_size,
        max_chunk_sentences=semantic_max_chunk_sentences,
        return_indices=True,
    )

    file_name = os.path.basename(file_path)
    ingest_date = datetime.now(timezone.utc).isoformat()

    # Per-file source_url: sidecar overrides global config.
    sidecar = load_sidecar_metadata(file_path)
    source_url = sidecar.get("source_url", default_source_url)

    written_ids: list[str] = []

    for i, (chunk_text, first_sent_idx) in enumerate(chunk_pairs):
        # Derive chunk-level metadata from the first sentence in this chunk.
        first_sent = sentences_data[
            min(first_sent_idx, len(sentences_data) - 1)
        ]

        metadata: dict = {
            "file_name": file_name,
            "chunk_id": i,
            "ingest_date": ingest_date,
        }
        # Only add optional fields when they carry a real value.
        if source_url:
            metadata["source_url"] = source_url
        if "page_number" in first_sent:
            metadata["page_number"] = first_sent["page_number"]
        if "section_title" in first_sent:
            metadata["section_title"] = first_sent["section_title"]

        try:
            embedding = embed_texts([chunk_text])[0]
        except Exception as exc:
            print(f"  Error embedding chunk {i} of '{file_name}': {exc}")
            continue

        chunk_id = f"{file_name}_chunk_{i}"
        collection.upsert(
            documents=[chunk_text],
            embeddings=[embedding],
            metadatas=[metadata],
            ids=[chunk_id],
        )
        written_ids.append(chunk_id)

    return written_ids


def main():
    print("\n--- Embedding and Storing Documents in ChromaDB ---")

    # ---------------------------------------------------------
    # 1. SETUP: Initialize embedding backend
    # ---------------------------------------------------------
    openai_client = None
    embedding_model = None

    if use_openai_embeddings:
        print(f"Using OpenAI Compatible API.")
        print(f"Model: {openai_embedding_model}")
        print(f"Base URL: {openai_embedding_base_url}")
        openai_client = OpenAI(
            base_url=openai_embedding_base_url,
            api_key=os.environ.get("OPENAI_API_KEY"),
        )
    else:
        print(f"Using Local SentenceTransformer.")
        print(f"Model: {model_name}")
        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

    embed_texts = build_embedder(openai_client, embedding_model)

    print(f"Chunking Method: {chunking_method}")
    if chunking_method == "semantic":
        print(f"  Breakpoint Percentile : {semantic_breakpoint_percentile}")
        print(f"  Buffer Size           : {semantic_buffer_size}")
        print(f"  Max Chunk Sentences   : {semantic_max_chunk_sentences}")
    else:
        print(f"  Chunk Size  : {chunk_size}")
        print(f"  Overlap Size: {overlap_size}")
    print(f"Raw Data Directory: {raw_db}")
    print(f"Vector DB Directory: {db_directory}")
    print(f"Vector DB Type: {vector_db}\n")

    # ---------------------------------------------------------
    # 2. Load manifest and compute config fingerprint
    # ---------------------------------------------------------
    manifest_path = get_manifest_path(db_directory, collection_name)
    manifest = load_manifest(manifest_path)

    config_fp = compute_config_fingerprint(
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        overlap_size=overlap_size,
        semantic_breakpoint_percentile=semantic_breakpoint_percentile,
        semantic_buffer_size=semantic_buffer_size,
        semantic_max_chunk_sentences=semantic_max_chunk_sentences,
    )

    # ---------------------------------------------------------
    # 3. Discover source files and open/create the collection
    # ---------------------------------------------------------
    file_paths = get_file_paths(raw_db, ["txt", "pdf", "docx", "md", "html", "htm", "csv"])
    print(f"Found {len(file_paths)} source files.")

    collection = client.get_or_create_collection(collection_name)

    # ---------------------------------------------------------
    # 4. Remove chunks for source files that no longer exist
    # ---------------------------------------------------------
    stale_paths = get_stale_files(file_paths, manifest)
    if stale_paths:
        print(f"\nRemoving {len(stale_paths)} deleted file(s) from the DB...")
        for stale_path in stale_paths:
            label = os.path.basename(stale_path)
            old_ids = manifest[stale_path].get("chunk_ids", [])
            delete_file_chunks(collection, old_ids, label)
            del manifest[stale_path]
            print(f"  Removed: {label} ({len(old_ids)} chunks deleted)")

    # ---------------------------------------------------------
    # 5. Index new / changed files; skip unchanged ones
    # ---------------------------------------------------------
    stats = {"skipped": 0, "new": 0, "updated": 0, "failed": 0}

    print()
    for file_path in tqdm(file_paths, desc="Processing documents"):
        file_hash = compute_file_hash(file_path)
        file_label = os.path.basename(file_path)

        if not needs_reindex(file_path, file_hash, config_fp, manifest):
            stats["skipped"] += 1
            continue

        is_update = file_path in manifest
        if is_update:
            # Delete the previously stored chunks before re-indexing.
            old_ids = manifest[file_path].get("chunk_ids", [])
            delete_file_chunks(collection, old_ids, file_label)

        new_ids = index_file(file_path, collection, embed_texts)

        if new_ids:
            manifest[file_path] = make_manifest_entry(file_hash, config_fp, new_ids)
            if is_update:
                stats["updated"] += 1
            else:
                stats["new"] += 1
        else:
            stats["failed"] += 1

    # ---------------------------------------------------------
    # 6. Persist the updated manifest
    # ---------------------------------------------------------
    save_manifest(manifest_path, manifest)

    # ---------------------------------------------------------
    # 7. Summary
    # ---------------------------------------------------------
    total_stored = sum(
        len(entry.get("chunk_ids", [])) for entry in manifest.values()
    )
    print("\n--- Indexing Complete ---")
    print(f"  New files indexed   : {stats['new']}")
    print(f"  Files re-indexed    : {stats['updated']}")
    print(f"  Files skipped       : {stats['skipped']} (unchanged)")
    print(f"  Files failed        : {stats['failed']}")
    print(f"  Total chunks in DB  : {total_stored}")
    print(f"  Manifest saved to   : {manifest_path}")


if __name__ == "__main__":
    main()
