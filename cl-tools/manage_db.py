#!/usr/bin/env python3
"""Collection management CLI for the FRAG vector database.

Provides subcommands to list, inspect, and delete ChromaDB collections
without writing ad-hoc code.

Usage examples::

    python cl-tools/manage_db.py list
    python cl-tools/manage_db.py stats
    python cl-tools/manage_db.py stats --collection my_collection44
    python cl-tools/manage_db.py inspect --collection my_collection44
    python cl-tools/manage_db.py inspect --collection my_collection44 --file report.pdf
    python cl-tools/manage_db.py delete --collection my_collection44
    python cl-tools/manage_db.py delete --collection my_collection44 --yes
"""

import argparse
import json
import os
import sys

# ---------------------------------------------------------------------------
# Resolve repo root so the script can be run from any working directory.
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_SCRIPT_DIR)
sys.path.insert(0, _REPO_ROOT)

import chromadb

from config.embedding_config import (
    db_directory,
    collection_name as default_collection,
)
from embedding.indexer import get_manifest_path, load_manifest


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_client() -> chromadb.PersistentClient:
    """Return a ChromaDB persistent client for the configured DB directory."""
    return chromadb.PersistentClient(path=db_directory)


def _load_manifest_for(col_name: str) -> dict:
    """Load the manifest for *col_name*, returning an empty dict if absent."""
    path = get_manifest_path(db_directory, col_name)
    return load_manifest(path)


def _manifest_files(manifest: dict) -> dict:
    """Return only the file entries from a manifest.

    ``load_manifest()`` already strips the envelope (version, updated_at)
    and returns the flat ``{file_path: entry}`` dict, so this is a
    pass-through that makes the intent explicit.
    """
    return manifest


def _pluralise(n: int, singular: str, plural: str = "") -> str:
    if not plural:
        plural = singular + "s"
    return f"{n} {singular if n == 1 else plural}"


def _manifest_updated_at(col_name: str) -> str:
    """Return the updated_at timestamp from the raw manifest JSON, or em-dash."""
    path = get_manifest_path(db_directory, col_name)
    if not os.path.exists(path):
        return "—"
    try:
        with open(path) as f:
            raw = json.load(f)
        ts = raw.get("updated_at", "—")
        return ts[:19].replace("T", " ") if ts and ts != "—" else "—"
    except Exception:
        return "—"


def _col_exists(client, name: str) -> bool:
    return any(c.name == name for c in client.list_collections())


def _separator(char: str = "─", width: int = 60) -> str:
    return char * width


# ---------------------------------------------------------------------------
# Subcommand: list
# ---------------------------------------------------------------------------

def cmd_list(args) -> None:
    """List all ChromaDB collections stored in db_directory."""
    client = _make_client()
    collections = client.list_collections()

    if not collections:
        print("No collections found in:", db_directory)
        return

    print(f"\n{'Collection':35} {'Chunks':>8}  {'Files':>6}  {'Last indexed':>22}  Config FP")
    print(_separator("─", 90))

    for col in sorted(collections, key=lambda c: c.name):
        count = col.count()
        manifest = _load_manifest_for(col.name)
        files = _manifest_files(manifest)
        n_files = len(files)
        updated_at = _manifest_updated_at(col.name)
        # Derive config fingerprint — all file entries share the same one.
        fps = {v.get("config_fingerprint", "?") for v in files.values()}
        fp_str = next(iter(fps)) if len(fps) == 1 else f"{len(fps)} mixed"

        active_marker = " *" if col.name == default_collection else "  "
        print(
            f"{active_marker}{col.name:33} {count:>8}  {n_files:>6}  "
            f"{updated_at:>22}  {fp_str}"
        )

    print()
    print(f"  * = active collection in embedding_config.py")
    print(f"  DB directory: {db_directory}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: stats
# ---------------------------------------------------------------------------

def cmd_stats(args) -> None:
    """Print detailed statistics for one collection."""
    col_name = args.collection or default_collection
    client = _make_client()

    if not _col_exists(client, col_name):
        print(f"Error: collection '{col_name}' does not exist.")
        sys.exit(1)

    col = client.get_collection(col_name)
    manifest = _load_manifest_for(col_name)
    files = _manifest_files(manifest)

    total_chunks = col.count()
    n_files = len(files)

    # ── Gather per-file stats from the manifest ──────────────────────────
    chunk_counts = [len(v.get("chunk_ids", [])) for v in files.values()]
    avg_chunks = (sum(chunk_counts) / n_files) if n_files else 0
    min_chunks = min(chunk_counts, default=0)
    max_chunks = max(chunk_counts, default=0)

    # ── Config fingerprint ───────────────────────────────────────────────
    fps = {v.get("config_fingerprint", "?") for v in files.values()}
    fp_str = next(iter(fps)) if len(fps) == 1 else f"{len(fps)} distinct (re-index needed)"

    # ── Metadata fields present in actual stored chunks ──────────────────
    # Sample up to 200 chunks to discover which metadata fields are used.
    SAMPLE = 200
    sample_result = col.get(limit=SAMPLE, include=["metadatas"])
    meta_keys: set = set()
    for m in sample_result.get("metadatas") or []:
        meta_keys.update(m.keys())

    # ── Per-extension breakdown ──────────────────────────────────────────
    ext_counts: dict[str, int] = {}
    for file_path in files:
        ext = os.path.splitext(file_path)[1].lower() or "(no ext)"
        ext_counts[ext] = ext_counts.get(ext, 0) + 1

    # ── Output ───────────────────────────────────────────────────────────
    print()
    print(_separator("═"))
    print(f"  Collection: {col_name}")
    if col_name == default_collection:
        print(f"  (active collection in embedding_config.py)")
    print(_separator("═"))
    print(f"  DB directory    : {db_directory}")
    manifest_path = get_manifest_path(db_directory, col_name)
    has_manifest = os.path.exists(manifest_path)
    print(f"  Manifest        : {'present' if has_manifest else 'missing — run vector_db_setup.py'}")
    if has_manifest:
        print(f"  Last indexed    : {_manifest_updated_at(col_name)} UTC")
    print()
    print(f"  Total chunks    : {total_chunks}")
    print(f"  Indexed files   : {n_files}")
    if n_files:
        print(f"  Avg chunks/file : {avg_chunks:.1f}")
        print(f"  Min chunks/file : {min_chunks}")
        print(f"  Max chunks/file : {max_chunks}")
    print()
    print(f"  Config FP       : {fp_str}")
    print()
    if ext_counts:
        print("  File types:")
        for ext, cnt in sorted(ext_counts.items(), key=lambda x: -x[1]):
            print(f"    {ext:10} {_pluralise(cnt, 'file')}")
        print()
    if meta_keys:
        print(f"  Metadata fields : {', '.join(sorted(meta_keys))}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: inspect
# ---------------------------------------------------------------------------

def cmd_inspect(args) -> None:
    """Show per-file breakdown for a collection, optionally filtering by filename."""
    col_name = args.collection or default_collection
    file_filter = args.file.lower() if args.file else None

    client = _make_client()
    if not _col_exists(client, col_name):
        print(f"Error: collection '{col_name}' does not exist.")
        sys.exit(1)

    manifest = _load_manifest_for(col_name)
    files = _manifest_files(manifest)

    if not files:
        print(f"No manifest data found for '{col_name}'.")
        print("Run embedding/vector_db_setup.py to build the manifest.")
        return

    # Filter if requested
    if file_filter:
        files = {
            k: v for k, v in files.items()
            if file_filter in os.path.basename(k).lower()
        }
        if not files:
            print(f"No files matching '{args.file}' found in manifest.")
            return

    print()
    print(f"  Collection: {col_name}  ({len(files)} file(s) shown)")
    print(_separator())
    print(f"  {'File':40} {'Chunks':>6}  Indexed at")
    print(_separator())

    total = 0
    for file_path, entry in sorted(files.items(), key=lambda x: os.path.basename(x[0])):
        basename = os.path.basename(file_path)
        n = len(entry.get("chunk_ids", []))
        total += n
        indexed_at = entry.get("indexed_at", "—")[:19].replace("T", " ")
        print(f"  {basename:40} {n:>6}  {indexed_at}")

        # If a specific file was requested, also show chunk IDs
        if args.file:
            for chunk_id in entry.get("chunk_ids", []):
                print(f"    · {chunk_id}")

    print(_separator())
    print(f"  {'TOTAL':40} {total:>6}")
    print()


# ---------------------------------------------------------------------------
# Subcommand: delete
# ---------------------------------------------------------------------------

def cmd_delete(args) -> None:
    """Delete a collection and its manifest from the database."""
    col_name = args.collection or default_collection
    client = _make_client()

    if not _col_exists(client, col_name):
        print(f"Error: collection '{col_name}' does not exist.")
        sys.exit(1)

    col = client.get_collection(col_name)
    chunk_count = col.count()
    manifest = _load_manifest_for(col_name)
    n_files = len(_manifest_files(manifest))

    print()
    print(f"  You are about to delete collection: '{col_name}'")
    print(f"  This will permanently remove {_pluralise(chunk_count, 'chunk')} "
          f"from {_pluralise(n_files, 'indexed file')}.")

    if col_name == default_collection:
        print()
        print("  WARNING: This is the ACTIVE collection in embedding_config.py.")
        print("           The app will break until you re-index.")

    if not args.yes:
        print()
        confirm = input("  Type the collection name to confirm deletion: ").strip()
        if confirm != col_name:
            print("  Aborted — name did not match.")
            return

    # Delete the ChromaDB collection
    client.delete_collection(col_name)

    # Delete the manifest file if it exists
    manifest_path = get_manifest_path(db_directory, col_name)
    if os.path.exists(manifest_path):
        os.remove(manifest_path)
        print(f"  Manifest removed: {manifest_path}")

    print(f"  Collection '{col_name}' deleted ({_pluralise(chunk_count, 'chunk')} removed).")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build and return the argument parser."""
    parser = argparse.ArgumentParser(
        prog="manage_db",
        description="FRAG collection management — list, inspect, and delete ChromaDB collections.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
examples:
  python cl-tools/manage_db.py list
  python cl-tools/manage_db.py stats
  python cl-tools/manage_db.py stats --collection my_collection2
  python cl-tools/manage_db.py inspect
  python cl-tools/manage_db.py inspect --file report.pdf
  python cl-tools/manage_db.py inspect --collection old_col --file notes.md
  python cl-tools/manage_db.py delete --collection old_col
  python cl-tools/manage_db.py delete --collection old_col --yes
        """,
    )
    sub = parser.add_subparsers(dest="command", metavar="subcommand")
    sub.required = True

    # -- list ----------------------------------------------------------------
    sub.add_parser(
        "list",
        help="List all collections with chunk counts and index timestamps.",
    )

    # -- stats ---------------------------------------------------------------
    p_stats = sub.add_parser(
        "stats",
        help="Detailed statistics for a single collection.",
    )
    p_stats.add_argument(
        "--collection", "-c",
        metavar="NAME",
        default=None,
        help="Collection name (default: active collection from embedding_config.py).",
    )

    # -- inspect -------------------------------------------------------------
    p_inspect = sub.add_parser(
        "inspect",
        help="Per-file breakdown of chunks for a collection.",
    )
    p_inspect.add_argument(
        "--collection", "-c",
        metavar="NAME",
        default=None,
        help="Collection name (default: active collection from embedding_config.py).",
    )
    p_inspect.add_argument(
        "--file", "-f",
        metavar="FILENAME",
        default=None,
        help="Filter to files whose name contains this string (case-insensitive).",
    )

    # -- delete --------------------------------------------------------------
    p_delete = sub.add_parser(
        "delete",
        help="Permanently delete a collection and its manifest.",
    )
    p_delete.add_argument(
        "--collection", "-c",
        metavar="NAME",
        default=None,
        help="Collection name (default: active collection from embedding_config.py).",
    )
    p_delete.add_argument(
        "--yes", "-y",
        action="store_true",
        help="Skip interactive confirmation (use with care in scripts).",
    )

    return parser


def main() -> None:
    """Parse arguments and dispatch to the appropriate subcommand."""
    parser = build_parser()
    args = parser.parse_args()

    dispatch = {
        "list":    cmd_list,
        "stats":   cmd_stats,
        "inspect": cmd_inspect,
        "delete":  cmd_delete,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()
