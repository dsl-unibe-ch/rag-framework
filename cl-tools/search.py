"""Command-line vector database search tool.

Retrieves matching chunks from ChromaDB for each query and prints them to
stdout.  Both OpenAI-compatible APIs and local SentenceTransformer models
are supported for embeddings.

Optional HyDE (Hypothetical Document Embeddings) can be toggled via the
``--hyde`` / ``--no-hyde`` flags; the default is taken from
``config/llm_config.py :: use_hyde``.  When HyDE is active the configured
LLM (OpenAI-compatible or Ollama) generates a short hypothetical answer
which is embedded instead of the raw query, typically improving retrieval
quality.
"""
import argparse
import os
import sys

from dotenv import load_dotenv

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from retrieval.main import ChromaRetriever, OpenAIChromaRetriever
from retrieval.hyde import (
    generate_hypothetical_document,
    generate_hypothetical_document_ollama,
)
from config.embedding_config import (
    model_name,
    db_directory,
    collection_name,
    use_openai_embeddings,
    openai_embedding_model,
    openai_embedding_base_url,
)
from config.llm_config import (
    llm_model,
    openai_model,
    use_openai,
    openai_base_url,
    use_hyde as config_use_hyde,
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Semantic search against the RAG vector database."
    )
    parser.add_argument(
        "--number-results",
        type=int,
        required=True,
        help="Number of results to display for each query.",
    )
    hyde_group = parser.add_mutually_exclusive_group()
    hyde_group.add_argument(
        "--hyde",
        dest="use_hyde",
        action="store_true",
        default=None,
        help="Enable HyDE (generate a hypothetical answer before retrieval). "
             "Overrides use_hyde in llm_config.py.",
    )
    hyde_group.add_argument(
        "--no-hyde",
        dest="use_hyde",
        action="store_false",
        help="Disable HyDE regardless of llm_config.py setting.",
    )
    return parser


def main() -> None:
    """Run the interactive search loop."""
    args = _create_parser().parse_args()
    hyde_enabled = config_use_hyde if args.use_hyde is None else args.use_hyde

    if hyde_enabled:
        print("[HyDE enabled] A hypothetical answer will be generated before each retrieval.")

    # Build retriever (embedding backend)
    if use_openai_embeddings:
        load_dotenv(os.path.join(parent_dir, ".env"))
        from openai import OpenAI as _OpenAI
        embed_client = _OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=openai_embedding_base_url,
        )
        retriever = OpenAIChromaRetriever(
            openai_client=embed_client,
            embedding_model=openai_embedding_model,
            db_path=db_directory,
            db_collection=collection_name,
            n_results=args.number_results,
        )
    else:
        retriever = ChromaRetriever(
            embedding_model=model_name,
            db_path=db_directory,
            db_collection=collection_name,
            n_results=args.number_results,
        )

    while True:
        query = input("Type a query to search the DB. Type 'quit' to exit:  ").strip()
        if query.lower() == "quit":
            break

        # HyDE: generate hypothetical document, then embed that instead of the raw query.
        hyde_doc = None
        if hyde_enabled:
            if use_openai:
                load_dotenv(os.path.join(parent_dir, ".env"))
                from openai import OpenAI as _OpenAI
                hyde_client = _OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=openai_base_url,
                )
                hyde_doc = generate_hypothetical_document(query, hyde_client, openai_model)
            else:
                hyde_doc = generate_hypothetical_document_ollama(query, llm_model)

            print(f"\n[HyDE doc]: {hyde_doc}\n")

        results = retriever.retrieve(query, embed_text=hyde_doc)

        print("\n--- Query Results ---\n")
        for idx, (doc, metadata, distance) in enumerate(
            zip(results["documents"][0], results["metadatas"][0], results["distances"][0])
        ):
            print(f"Result {idx + 1}:")
            print(f"  Document ID : {metadata.get('chunk_id', 'N/A')}")
            print(f"  File Name   : {metadata.get('file_name', 'N/A')}")
            if metadata.get("page_number"):
                print(f"  Page        : {metadata['page_number']}")
            if metadata.get("section_title"):
                print(f"  Section     : {metadata['section_title']}")
            print(f"  Distance    : {distance}")
            print(f"  Content:\n{doc}\n")
            print("-" * 80)


if __name__ == "__main__":
    main()
