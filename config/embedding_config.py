# this file containst default values related to embeddings and creating vectordb
import os

# Local SentenceTransformer model used when use_openai_embeddings = False.

#
#   "sentence-transformers/all-MiniLM-L6-v2"          -- small (90 MB), fast, English
#   "sentence-transformers/all-mpnet-base-v2"          -- larger, better quality, English
#   "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"  -- multilingual
#   "BAAI/bge-m3"                                      -- multilingual, state-of-the-art
#   "intfloat/multilingual-e5-large"                   -- multilingual, very strong
model_name = "BAAI/bge-m3"

# settings if using openai embeddings api or any openai compatible embedding api
use_openai_embeddings = False # set to True if you want to use openai embeddings api
openai_embedding_model = "embedding-model-name" # openai embedding model to use if use_openai_embeddings is set to True
openai_embedding_base_url = 'https://api.openai.com/v1' # openai base url. In case you are using a different base url for openai compatible api

vector_db = "chromaDB" # Allowed Values ['chromaDB', 'FAISS']. Only ChromaDB works now

collection_name = "my_collection" #name of the collection in the vector DB

# for windows make the path string raw string. example: raw_db = r"C:\path\to\data"
raw_db = "/path/to/data"  #root directory to where raw documents are stored

data_language = "english" #variable for the tokenizer. Supported language = ['czech', 'danish', 'dutch', 'english', 'estonian', 'finnish', 'french', 'german' ,'greek' ,'italian' ,'norwegian', 'polish' ,'portuguese', 'russian' ,'slovene','spanish', 'swedish', 'turkish']

db_directory = os.path.join(os.path.expanduser('~'), '.db')  #default. Change it to where you want to store the vector DB




# ---------------------------------------------------------------------------
# Chunking strategy
# ---------------------------------------------------------------------------
# How documents are split into chunks before embedding.
# Allowed Values:
#   "sentence" -> rule-based: fixed-size, overlapping windows of sentences
#                 (uses chunk_size and overlap_size above).
#   "semantic" -> embedding-based: starts a new chunk when the topic shifts.
#                 (uses the semantic_* settings below; chunk_size and
#                  overlap_size are ignored).
#   "token"    -> token-budget chunking: greedily accumulates sentences until
#                 the token limit (token_chunk_size) is reached, then starts a
#                 new chunk. Produces predictably sized chunks for any LLM
#                 context window (uses the token_* settings below).
chunking_method = "sentence"

# --- Semantic chunking settings (only used when chunking_method == "semantic") ---

# Percentile (0-100) of consecutive-sentence distances used as the split
# threshold. Higher -> fewer, larger chunks. Lower -> more, smaller chunks.
semantic_breakpoint_percentile = 95

# Number of neighbouring sentences combined with each sentence to give it
# context before embedding. 0 embeds each sentence on its own.
semantic_buffer_size = 1

# Optional hard cap on the number of sentences per semantic chunk, useful to
# avoid a single very large chunk. 0 disables the cap.
semantic_max_chunk_sentences = 0


# --- Token-aware chunking settings (only used when chunking_method == "token") ---

# Target number of tokens per chunk.  A lower value produces more focused
# chunks; a higher value provides more context per chunk.  512 is a safe
# default for most LLMs with a 4096-token context window.
token_chunk_size = 512

# Number of tokens of overlap carried over from the end of the previous chunk
# into the start of the next one.  Overlap prevents important context from
# disappearing at a hard boundary.  Set to 0 for no overlap.
token_chunk_overlap = 50

# tiktoken encoding used to count tokens.  "cl100k_base" is the encoding
# used by GPT-4, GPT-3.5-turbo, and many other modern models and is a
# reasonable universal approximation.  Change to match your LLM's tokeniser
# if you need exact counts (e.g. "p50k_base" for older GPT-3 models).
token_encoding = "cl100k_base"

# --- Sentence-based chunking settings (only used when chunking_method == "sentence") ---
chunk_size = 40   #number of sentences each chunk will contain in the vector db

overlap_size = 5 # must be less than the chunk_size. It indicates how many sentences overlaps when splitting chunks


# ---------------------------------------------------------------------------
# Embedding batch size
# ---------------------------------------------------------------------------
# Maximum number of texts sent to the embedding backend in a single call.
# For OpenAI-compatible APIs this maps directly to the number of items in
# one HTTP request; most providers cap this at 96–2048.  For local
# SentenceTransformer models, encode() handles its own internal batching
# but this controls how many chunks are collected before a single encode()
# call (set high, e.g. 512, for local models).
embedding_batch_size = 64

# ---------------------------------------------------------------------------
# Retrieval settings
# ---------------------------------------------------------------------------
# Default number of chunks returned from the vector database for each query.
# Can be overridden at query time via the web UI or CLI arguments.
n_results = 5


# ---------------------------------------------------------------------------
# Source metadata
# ---------------------------------------------------------------------------
# Optional URL indicating where the documents were sourced from.
# Used as a global default; individual files can override this by placing a
# sidecar file named "{filename}.meta.json" alongside them, e.g.:
#   {"source_url": "https://example.com/my-doc.pdf"}
# Leave as empty string if documents are local and have no upstream URL.
source_url = ""


# ---------------------------------------------------------------------------
# Hybrid search  (BM25 keyword  +  vector)  with Reciprocal Rank Fusion
# ---------------------------------------------------------------------------
# When True, a keyword-based BM25 index of all stored chunks is queried in
# parallel with the normal vector search.  The two ranked lists are merged
# with Reciprocal Rank Fusion (RRF) before the final results are returned.
#
# Benefits:
#   * Exact-match recall for names, IDs, and rare terms that dense embeddings
#     often miss.
#   * Robust across queries that are either keyword-like or semantic.
# Can be toggled per-request from the web UI or CLI --hybrid / --no-hybrid.
use_hybrid_search = False

# RRF constant k.  The standard default (60) works well in practice.
# Higher values reduce the penalty for lower-ranked results.
hybrid_rrf_k = 60

# Candidate multiplier.  Before fusing, each source retrieves
# n_results * hybrid_candidates candidates.  More candidates → better fusion
# quality at the cost of slightly more BM25 work.
hybrid_candidates = 3
