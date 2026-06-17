# this file containst default values related to embeddings and creating vectordb
import os

model_name = "Lajavaness/bilingual-embedding-large"  #choose any embedding model you prefer that runs in sentence-transformers

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

chunk_size = 20   #number of sentences each chunk will contain in the vector db

overlap_size = 5 # must be less than the chunk_size. It indicates how many sentences overlaps when splitting chunks


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
chunking_method = "semantic"

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
