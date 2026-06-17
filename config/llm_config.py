
# config for using ollama
llm_model = 'deepseek-r1:1.5b' # select any model available on the ollama site https://ollama.com/search

#config for openai api
use_openai = False # set to True if using openai api and then select 'openai_model' variable. You need to add the openai api token in the .env file in the root dirextory

openai_base_url = 'https://api.openai.com/v1' # openai base url. In case you are using a different base url for openai compatible api

openai_model = 'gpt-4o' # if using openai api then select which model to use



# prompt template for the LLM
prompt = """
DOCUMENTS: \n
{data}
\n
\n
QUESTION:
{query}
\n
\n
INSTRUCTIONS:
Answer the users QUESTION using the DOCUMENTS text above.
Keep your answer ground in the facts of the DOCUMENT.
If the DOCUMENT doesn’t contain the facts to answer the QUESTION return NO Answer found
"""

# whether to record data from user interactions
record_data = False
# ---------------------------------------------------------------------------
# HyDE - Hypothetical Document Embeddings
# ---------------------------------------------------------------------------
# When True, the LLM is asked to draft a short hypothetical answer for the
# user's question *before* retrieval.  The hypothetical text is then embedded
# instead of the raw question, which typically improves retrieval quality
# because generated answers are stylistically closer to stored document
# chunks than a short question is.
#
# This setting acts as the global default; it can be toggled per-request from
# the web UI without modifying this file.
use_hyde = False
