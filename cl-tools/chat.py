"""Command-line chat interface for the RAG framework.

Streams answers from the configured LLM, grounded in documents retrieved
from the ChromaDB vector store.  Both OpenAI-compatible APIs and local
Ollama models are supported for both the LLM and the embedding model.

Optional HyDE (Hypothetical Document Embeddings) can be toggled via the
``--hyde`` / ``--no-hyde`` flags; the default is taken from
``config/llm_config.py :: use_hyde``.
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
    n_results,
)
from llm.main import Responder, OpenAIResponder
from config.llm_config import (
    llm_model,
    prompt,
    openai_model,
    use_openai,
    openai_base_url,
    use_hyde as config_use_hyde,
)


def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stream RAG answers from the command line.")
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
    """Run the interactive chat loop."""
    args = _create_parser().parse_args()
    # None means the flag was not passed → fall back to config
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
            n_results=n_results,
        )
    else:
        retriever = ChromaRetriever(
            embedding_model=model_name,
            db_path=db_directory,
            db_collection=collection_name,
            n_results=n_results,
        )

    while True:
        user_query = input("Ask a question. Type quit to exit:  ").strip()
        if user_query.lower() == "quit":
            break

        print("Looking the DB for relevant information .......")

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
                hyde_doc = generate_hypothetical_document(user_query, hyde_client, openai_model)
            else:
                hyde_doc = generate_hypothetical_document_ollama(user_query, llm_model)

            print(f"\n[HyDE doc]: {hyde_doc}\n")

        search_results = retriever.retrieve(user_query, embed_text=hyde_doc)
        formatted_result = retriever.format_results_for_prompt(search_results)

        if use_openai:
            load_dotenv(os.path.join(parent_dir, ".env"))
            from openai import OpenAI as _OpenAI
            llm_client = _OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=openai_base_url,
            )
            responder = OpenAIResponder(
                data=formatted_result,
                model=openai_model,
                prompt_template=prompt,
                query=user_query,
                client=llm_client,
            )
        else:
            responder = Responder(
                data=formatted_result,
                model=llm_model,
                prompt_template=prompt,
                query=user_query,
            )

        responder.stream_response()


if __name__ == "__main__":
    main()
