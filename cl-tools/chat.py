import os
import sys
from dotenv import load_dotenv
from openai import OpenAI

# Add the parent directory to sys.path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

from retrieval.main import ChromaRetriever, OpenAIChromaRetriever
from config.embedding_config import model_name, db_directory, collection_name, use_openai_embeddings, openai_embedding_model, openai_embedding_base_url

from llm.main import Responder, OpenAIResponder
from config.llm_config import llm_model, prompt, openai_model, use_openai, openai_base_url


load_dotenv(os.path.join(parent_dir, '.env'))


openai_client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
    base_url=openai_base_url  
)

def main():
    while True:
        if use_openai_embeddings:
            load_dotenv(os.path.join(parent_dir, '.env'))
            openai_client = OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY"),
                base_url=openai_embedding_base_url
            )

            retriever = OpenAIChromaRetriever(
                openai_client=openai_client,
                embedding_model=openai_embedding_model,
                db_path=db_directory,
                db_collection=collection_name,
                n_results=5
                )
        else:
            retriever = ChromaRetriever(embedding_model=model_name, 
                                db_path=db_directory, 
                                db_collection=collection_name, 
                                n_results=5)
        
        user_query = str(input("Ask a question. Type quit to exit:  "))
        if user_query.lower() == "quit":
            break
        else:
            print("Looking the DB for relevant information .......")
            # get the data for the RAG and put it in str format
            search_results = retriever.retrieve(user_query)
            formated_result = retriever.format_results_for_prompt(search_results)

            if use_openai:
                responder = OpenAIResponder(data=formated_result, model=openai_model, 
                                            prompt_template=prompt, query= user_query,client=openai_client)
                responder.stream_response()
            else:
                responder = Responder(data=formated_result, model=llm_model, prompt_template=prompt, query=user_query)
                responder.stream_response()


if __name__ == "__main__":
    main()




