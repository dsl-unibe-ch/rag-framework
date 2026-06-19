from django.shortcuts import render
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST
from django.http import StreamingHttpResponse, JsonResponse
from django.conf import settings

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
    n_results as default_n_results,
)
from llm.main import Responder, OpenAIResponder
from config.llm_config import (
    llm_model,
    prompt,
    use_openai,
    openai_model,
    record_data,
    openai_base_url,
    use_hyde as config_use_hyde,
)
from .models import ChatLog

from dotenv import load_dotenv
from openai import OpenAI
import os
import json


def home(request):
    footer_class = 'footer-absolute'
    return render(request, 'rag_app/home.html', {'footer_class': footer_class})


def search(request):
    """Render the vector-database search page.

    On GET: shows the empty search form.
    On POST: embeds the query (optionally via HyDE), retrieves matching
    chunks from ChromaDB, and renders the results.
    """
    submitted = False
    formatted_results = []
    hyde_doc = None
    hyde_enabled = config_use_hyde

    if request.method == "POST":
        query = request.POST["query"]
        n_results = int(request.POST["n_results"])
        submitted = True

        # HyDE: UI checkbox overrides the config default.
        ui_hyde = request.POST.get('use_hyde')
        hyde_enabled = (ui_hyde == '1') if ui_hyde is not None else config_use_hyde

        if use_openai_embeddings:
            load_dotenv(os.path.join(settings.BASE_DIR.parent, '.env'))
            openai_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=openai_embedding_base_url,
            )
            try:
                retriever = OpenAIChromaRetriever(
                    openai_client=openai_client,
                    embedding_model=openai_embedding_model,
                    db_path=db_directory,
                    db_collection=collection_name,
                    n_results=n_results,
                )
            except Exception as e:
                print(f"Error during retrieval: {e}")
        else:
            try:
                retriever = ChromaRetriever(
                    embedding_model=model_name,
                    db_path=db_directory,
                    db_collection=collection_name,
                    n_results=n_results,
                )
            except Exception as e:
                print(f"Error during retrieval: {e}")

        # Generate hypothetical document when HyDE is active.
        if hyde_enabled:
            if use_openai:
                load_dotenv(os.path.join(settings.BASE_DIR.parent, '.env'))
                hyde_client = OpenAI(
                    api_key=os.environ.get("OPENAI_API_KEY"),
                    base_url=openai_base_url,
                )
                hyde_doc = generate_hypothetical_document(query, hyde_client, openai_model)
            else:
                hyde_doc = generate_hypothetical_document_ollama(query, llm_model)

        raw_results = retriever.retrieve(query, embed_text=hyde_doc)

        if not raw_results:
            raw_results = {"documents": [[]], "metadatas": [[]], "distances": [[]]}

        documents = raw_results.get("documents", [[]])[0]
        metadatas = raw_results.get("metadatas", [[]])[0]
        distances = raw_results.get("distances", [[]])[0]

        for doc, metadata, distance in zip(documents, metadatas, distances):
            formatted_results.append({
                "content": doc,
                "file_name": metadata.get("file_name", "N/A"),
                "chunk_id": metadata.get("chunk_id", "N/A"),
                "page_number": metadata.get("page_number"),
                "section_title": metadata.get("section_title"),
                "source_url": metadata.get("source_url", ""),
                "ingest_date": metadata.get("ingest_date", ""),
                "distance": distance,
            })

        return render(
            request,
            "rag_app/search.html",
            {
                "data": formatted_results,
                "submitted": submitted,
                "query": query,
                "n_results": n_results,
                "footer_class": "footer-flex",
                "config_use_hyde": config_use_hyde,
                "hyde_enabled": hyde_enabled,
                "hyde_doc": hyde_doc,
            },
        )

    return render(
        request,
        "rag_app/search.html",
        {
            "data": formatted_results,
            "submitted": submitted,
            "n_results": default_n_results,
            "footer_class": "footer-absolute",
            "config_use_hyde": config_use_hyde,
            "hyde_enabled": config_use_hyde,
            "hyde_doc": None,
        },
    )


def chat_page(request):
    """Render the streaming chat page."""
    return render(request, 'rag_app/chat.html', {
        'footer_class': 'footer-absolute',
        'record_data': record_data,
        'default_n_results': default_n_results,
        'config_use_hyde': config_use_hyde,
    })


@csrf_exempt
@require_POST
def chat_stream(request):
    """Handle a streaming chat request.

    Reads the user query and optional overrides (n_results, use_hyde) from
    POST data.  When HyDE is active, the LLM first generates a hypothetical
    answer which is embedded instead of the raw question.  The final SSE
    payload includes both the streamed LLM answer and a ``<|DOCS_JSON|>``
    marker carrying retrieved-document metadata and the optional HyDE text.
    """
    user_query = request.POST.get('query', '').strip()
    if not user_query:
        return JsonResponse({"error": "No query provided"}, status=400)

    # Accept n_results from the UI; fall back to the config default.
    try:
        n_results = int(request.POST.get('n_results', default_n_results))
        n_results = max(1, n_results)
    except (ValueError, TypeError):
        n_results = default_n_results

    # HyDE: UI toggle overrides the config default.  The checkbox sends '1'
    # when enabled or '0' when explicitly disabled.  Absence of the key
    # means "use whatever the config says".
    ui_hyde = request.POST.get('use_hyde')
    if ui_hyde is not None:
        hyde_enabled = ui_hyde == '1'
    else:
        hyde_enabled = config_use_hyde

    # Generate hypothetical document before retrieval when HyDE is active.
    hyde_doc = None
    if hyde_enabled:
        if use_openai:
            load_dotenv(os.path.join(settings.BASE_DIR.parent, '.env'))
            hyde_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=openai_base_url,
            )
            hyde_doc = generate_hypothetical_document(user_query, hyde_client, openai_model)
        else:
            hyde_doc = generate_hypothetical_document_ollama(user_query, llm_model)

    # -- 1) Build retriever
    try:
        if use_openai_embeddings:
            load_dotenv(os.path.join(settings.BASE_DIR.parent, '.env'))
            openai_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=openai_embedding_base_url,
            )
            retriever = OpenAIChromaRetriever(
                openai_client=openai_client,
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
    except Exception as exc:
        err = f"Could not open collection '{collection_name}': {exc}"
        print(f"[chat_stream] {err}")
        return JsonResponse({"error": err}, status=500)

    # -- 2) Retrieve (using HyDE doc as embed text when available).
    # hyde_doc is None when HyDE is disabled OR when generation failed;
    # in both cases embed_text=None makes retrieve() use the raw query.
    if hyde_doc:
        print(f"[HyDE] Using hypothetical doc ({len(hyde_doc)} chars) for retrieval")
    search_results = retriever.retrieve(user_query, embed_text=hyde_doc)
    formatted_result = retriever.format_results_for_prompt(search_results)

    doc_list_for_frontend = []
    if search_results and 'documents' in search_results and 'metadatas' in search_results:
        for doc, metadata in zip(search_results['documents'][0], search_results['metadatas'][0]):
            doc_list_for_frontend.append({
                "file_name": metadata.get('file_name', 'N/A'),
                "chunk_id": metadata.get('chunk_id', 'N/A'),
                "page_number": metadata.get('page_number'),
                "section_title": metadata.get('section_title'),
                "source_url": metadata.get('source_url', ''),
                "ingest_date": metadata.get('ingest_date', ''),
                "content": doc,
            })

    # -- 3) Build LLM responder
    if use_openai:
        load_dotenv(os.path.join(settings.BASE_DIR.parent, '.env'))
        openai_client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=openai_base_url,
        )
        responder = OpenAIResponder(
            data=formatted_result,
            model=openai_model,
            prompt_template=prompt,
            query=user_query,
            client=openai_client,
        )
    else:
        responder = Responder(
            data=formatted_result,
            model=llm_model,
            prompt_template=prompt,
            query=user_query,
        )

    # -- 4) Streaming generator
    def stream_generator():
        full_response = ""
        for chunk in responder.stream_response_chunks():
            full_response += chunk
            yield chunk

        # Send retrieved docs + optional HyDE doc to the client in one JSON blob.
        meta_payload = {
            "docs": doc_list_for_frontend,
            "hyde_doc": hyde_doc,
        }
        yield f"<|DOCS_JSON|>{json.dumps(meta_payload)}"

        if record_data:
            ChatLog.objects.create(
                user_query=user_query,
                response=full_response,
            )

    return StreamingHttpResponse(stream_generator(), content_type='text/plain')
