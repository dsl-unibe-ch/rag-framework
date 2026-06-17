"""Manifest-based helpers for incremental vector-database indexing.

A *manifest* is a small JSON file stored next to the ChromaDB database that
records, for every indexed source file:

* the SHA-256 hash of its contents at the time it was last indexed,
* a fingerprint of the chunking configuration that was active,
* the list of ChromaDB chunk IDs that were written for it,
* the UTC timestamp of the last indexing run.

On each run :func:`needs_reindex` compares the current hash and config
fingerprint against the stored values.  A file is only re-processed when
its content has changed *or* the chunking configuration has changed.
Files that have been deleted from the source directory are also detected
and their chunks can be removed from the database before updating the
manifest.

Typical usage inside ``vector_db_setup.py``::

    from embedding.indexer import (
        get_manifest_path,
        load_manifest,
        save_manifest,
        compute_file_hash,
        compute_config_fingerprint,
        needs_reindex,
        get_stale_files,
        make_manifest_entry,
    )

    manifest_path = get_manifest_path(db_directory, collection_name)
    manifest = load_manifest(manifest_path)
    config_fp = compute_config_fingerprint(
        chunking_method=chunking_method,
        chunk_size=chunk_size,
        ...
    )

    for file_path in all_files:
        file_hash = compute_file_hash(file_path)
        if not needs_reindex(file_path, file_hash, config_fp, manifest):
            continue   # skip – nothing changed
        ...            # delete old chunks, embed new ones
        manifest[file_path] = make_manifest_entry(file_hash, config_fp, new_chunk_ids)

    for stale_path in get_stale_files(all_files, manifest):
        ...            # delete stale chunks from ChromaDB
        del manifest[stale_path]

    save_manifest(manifest_path, manifest)
"""

import hashlib
import json
import os
from datetime import datetime, timezone
from typing import Dict, List


# Bump this when the manifest schema changes in a backwards-incompatible way.
_MANIFEST_VERSION = 1


def get_manifest_path(db_directory: str, collection_name: str) -> str:
    """Return the path to the manifest JSON file for a collection.

    The manifest is stored inside ``db_directory`` so it stays co-located
    with the ChromaDB data it describes.

    Args:
        db_directory: Directory where the vector database is stored.
        collection_name: Name of the ChromaDB collection.

    Returns:
        Absolute path to the manifest JSON file.
    """
    return os.path.join(db_directory, f"{collection_name}_manifest.json")


def load_manifest(manifest_path: str) -> Dict[str, Dict]:
    """Load the manifest from disk, returning an empty dict if none exists.

    Args:
        manifest_path: Path to the manifest JSON file.

    Returns:
        A dict mapping absolute source-file paths to their index metadata
        dicts (keys: ``hash``, ``config_fingerprint``, ``chunk_ids``,
        ``indexed_at``).
    """
    if not os.path.exists(manifest_path):
        return {}

    with open(manifest_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)

    return data.get("files", {})


def save_manifest(manifest_path: str, files: Dict[str, Dict]) -> None:
    """Persist the manifest to disk atomically.

    The directory is created if it does not yet exist.

    Args:
        manifest_path: Path to the manifest JSON file.
        files: Dict mapping absolute source-file paths to their metadata.
    """
    os.makedirs(os.path.dirname(manifest_path), exist_ok=True)

    data = {
        "version": _MANIFEST_VERSION,
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "files": files,
    }

    # Write to a temp file then rename for atomicity.
    tmp_path = manifest_path + ".tmp"
    with open(tmp_path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp_path, manifest_path)


def compute_file_hash(file_path: str) -> str:
    """Compute the SHA-256 hash of a file's raw bytes.

    Reads the file in 64 KiB blocks so large files do not exhaust memory.

    Args:
        file_path: Path to the file.

    Returns:
        Hex-encoded SHA-256 digest string.
    """
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as fh:
        for block in iter(lambda: fh.read(65_536), b""):
            sha256.update(block)
    return sha256.hexdigest()


def compute_config_fingerprint(**chunking_params) -> str:
    """Compute a short fingerprint of the active chunking configuration.

    Any change to the keyword arguments (chunking method, sizes, percentile,
    etc.) produces a different fingerprint, which causes all files to be
    re-indexed with the new settings.

    Args:
        **chunking_params: All chunking config values that affect the shape
            of the final chunks, e.g.::

                compute_config_fingerprint(
                    chunking_method="semantic",
                    chunk_size=20,
                    overlap_size=5,
                    semantic_breakpoint_percentile=95,
                    semantic_buffer_size=1,
                    semantic_max_chunk_sentences=0,
                )

    Returns:
        An 8-character lowercase hex string.
    """
    canonical = json.dumps(chunking_params, sort_keys=True)
    return hashlib.md5(canonical.encode()).hexdigest()[:8]


def needs_reindex(
    file_path: str,
    file_hash: str,
    config_fingerprint: str,
    manifest: Dict[str, Dict],
) -> bool:
    """Return ``True`` if a file must be (re-)indexed.

    A file needs indexing when any of the following is true:

    * It has never been indexed (not present in the manifest).
    * Its content has changed (the stored hash differs from ``file_hash``).
    * The chunking configuration has changed (the stored fingerprint differs).

    Args:
        file_path: Absolute path to the source file.
        file_hash: Current SHA-256 hash of the file.
        config_fingerprint: Fingerprint of the current chunking config.
        manifest: The manifest dict returned by :func:`load_manifest`.

    Returns:
        ``True`` if the file should be indexed; ``False`` to skip it.
    """
    entry = manifest.get(file_path)
    if entry is None:
        return True
    if entry.get("hash") != file_hash:
        return True
    if entry.get("config_fingerprint") != config_fingerprint:
        return True
    return False


def get_stale_files(
    current_file_paths: List[str],
    manifest: Dict[str, Dict],
) -> List[str]:
    """Return source paths that are tracked in the manifest but no longer exist.

    These files have been deleted from ``raw_db`` since the last run.  Their
    chunks should be removed from ChromaDB and their entries deleted from the
    manifest.

    Args:
        current_file_paths: All files currently found in the source directory.
        manifest: The manifest dict returned by :func:`load_manifest`.

    Returns:
        List of absolute paths that are in the manifest but not on disk.
    """
    current_set = set(current_file_paths)
    return [path for path in manifest if path not in current_set]


def make_manifest_entry(
    file_hash: str,
    config_fingerprint: str,
    chunk_ids: List[str],
) -> Dict:
    """Build a manifest entry dict for a freshly indexed file.

    Args:
        file_hash: SHA-256 hash of the file content at index time.
        config_fingerprint: Active chunking config fingerprint.
        chunk_ids: Ordered list of ChromaDB chunk IDs written for this file.

    Returns:
        A dict ready to be stored in the manifest under the file's path key.
    """
    return {
        "hash": file_hash,
        "config_fingerprint": config_fingerprint,
        "chunk_ids": chunk_ids,
        "indexed_at": datetime.now(timezone.utc).isoformat(),
    }


def load_sidecar_metadata(file_path: str) -> Dict:
    """Load optional per-file metadata from a sidecar JSON file.

    If a file ``{file_path}.meta.json`` exists next to the source document,
    it is read and its contents returned.  This allows per-document metadata
    (e.g. ``source_url``) to override global config values without requiring
    changes to the config file.

    Example sidecar file ``report.pdf.meta.json``::

        {
            "source_url": "https://example.com/reports/report.pdf"
        }

    Args:
        file_path: Absolute path to the source document (not the sidecar).

    Returns:
        A dict with whatever keys the sidecar file contains, or an empty
        dict if the sidecar does not exist or cannot be parsed.
    """
    sidecar_path = file_path + ".meta.json"
    if not os.path.exists(sidecar_path):
        return {}
    try:
        with open(sidecar_path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if isinstance(data, dict):
            return data
    except Exception as exc:
        print(f"  Warning: could not read sidecar '{sidecar_path}': {exc}")
    return {}
