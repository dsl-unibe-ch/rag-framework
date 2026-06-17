import os
import sys
import chromadb
from sentence_transformers import SentenceTransformer


# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)


from config.embedding_config import model_name, db_directory, collection_name, n_results


# Initialize the ChromaDB persistent client
client = chromadb.PersistentClient(path=db_directory)

# Get the collection
collection = client.get_collection(name=collection_name)

# Define your query text
query_text = "I am looking for books about war"

# Initialize the embedding model (same as used during indexing)
embedding_model = SentenceTransformer(model_name)

# Embed the query text
query_embedding = embedding_model.encode(query_text)

# Perform the query
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=n_results
)

# Print out the results
print("\n--- Query Results ---\n")
for idx, (doc, metadata, distance) in enumerate(zip(results['documents'][0], results['metadatas'][0], results['distances'][0])):
    print(f"Result {idx + 1}:")
    print(f"Document ID: {metadata.get('chunk_id', 'N/A')}")
    print(f"File Name: {metadata.get('file_name', 'N/A')}")
    print(f"Distance: {distance}")
    print(f"Content:\n{doc}\n")
    print("-" * 80)
