import sys
import os
import chromadb
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
)

from embedding.utils import (
    get_file_paths,
    read_text_file,
    read_pdf_file,
    split_text_into_sentences,
)
from embedding.chunking import create_chunks

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


def main():
    print("\n--- Embedding and Storing Documents in ChromaDB ---")

    # ---------------------------------------------------------
    # 1. SETUP: Initialize the correct model based on config
    # ---------------------------------------------------------
    openai_client = None
    embedding_model = None

    if use_openai_embeddings:
        print(f"Using OpenAI Compatible API.")
        print(f"Model: {openai_embedding_model}")
        print(f"Base URL: {openai_embedding_base_url}")

        # Initialize OpenAI Client
        openai_client = OpenAI(
            base_url=openai_embedding_base_url,
            api_key=os.environ.get("OPENAI_API_KEY")  # Ensure this exists in your .env
        )
    else:
        print(f"Using Local SentenceTransformer.")
        print(f"Model: {model_name}")

        # Initialize Local Model
        embedding_model = SentenceTransformer(model_name, trust_remote_code=True)

    # Single embedding entry point reused for chunking and storage.
    embed_texts = build_embedder(openai_client, embedding_model)

    print(f"Chunking Method: {chunking_method}")
    if chunking_method == "semantic":
        print(f"Semantic Breakpoint Percentile: {semantic_breakpoint_percentile}")
        print(f"Semantic Buffer Size: {semantic_buffer_size}")
        print(f"Semantic Max Chunk Sentences: {semantic_max_chunk_sentences}")
    else:
        print(f"Chunk Size (sentences per chunk): {chunk_size}")
        print(f"Overlap Size (sentences): {overlap_size}")
    print(f"Raw Data Directory: {raw_db}")
    print(f"Vector Database Directory: {db_directory}\n")
    print(f"Vector Database is: {vector_db}\n")

    # Step 1: Load documents (txt and pdf)
    file_paths = get_file_paths(raw_db, ["txt", "pdf"])
    print(f"Found {len(file_paths)} files to process.\n")

    # Create or retrieve the collection in ChromaDB
    collection = client.get_or_create_collection(collection_name)

    total_chunks = 0

    for file_path in tqdm(file_paths, desc="Processing documents"):
        # Step 2: Read content based on file type
        if file_path.endswith('.txt'):
            text = read_text_file(file_path)
        elif file_path.endswith('.pdf'):
            text = read_pdf_file(file_path)
        else:
            print(f"Unsupported file type: {file_path}")
            continue

        # Step 3: Split text into sentences
        sentences = split_text_into_sentences(text, data_language)

        # Step 4: Chunk sentences using the configured strategy
        chunks = create_chunks(
            chunking_method,
            sentences,
            chunk_size=chunk_size,
            overlap_size=overlap_size,
            embed_fn=embed_texts,
            breakpoint_percentile=semantic_breakpoint_percentile,
            buffer_size=semantic_buffer_size,
            max_chunk_sentences=semantic_max_chunk_sentences,
        )

        # Use file name as the document ID and create metadata with chunk index
        file_name = os.path.basename(file_path)

        for i, chunk_text in enumerate(chunks):

            # ---------------------------------------------------------
            # 5. EMBED: Generate embedding based on selected method
            # ---------------------------------------------------------
            try:
                embedding = embed_texts([chunk_text])[0]
            except Exception as e:
                print(f"\nError embedding chunk {i} of {file_name}: {e}")
                continue

            # Create a unique ID for each chunk
            chunk_id = f"{file_name}_chunk_{i}"

            collection.add(
                documents=[chunk_text],
                embeddings=[embedding],
                metadatas=[{"file_name": file_name, "chunk_id": i}],
                ids=[chunk_id]
            )
            total_chunks += 1

    print("\n--- Embedding and Storage Complete ---")
    print(f"Stored {len(file_paths)} documents in ChromaDB.\n")
    print(f"Stored {total_chunks} Chunks in the DB")


if __name__ == "__main__":
    main()
