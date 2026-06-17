## Framewrok for RAG (FRAG)

## ⚠️ Important Note

This project is still under active development and should be considered in a pre-release state. It has not been thoroughly tested yet, and some features may be incomplete or unstable. Use it at your own discretion and report any issues or bugs.

Contributions, feedback, and suggestions are welcome as we work toward a stable release!



# RAG Framework

This repository contains a Retrieval-Augmented Generation (RAG) framework for efficient information retrieval and natural language generation. The framework is designed for maximum flexibility, supporting:

*  Embeddings: Local generation (via SentenceTransformers) or API-based generation (OpenAI, GPUStack, etc.).

*  LLM Generation: Local models (via Ollama) or Cloud/API-based models (OpenAI, GPUStack, etc.).


## How to Get Started


**Prerequisites:**

*   Ollama server. Install from [https://ollama.com/](https://ollama.com/)
*   For using openai api or openai api compatable platform, get api key and store it in .env file in the root level of the directory. See .env.example 
*   python 3.12

**Steps:**

1.  **Create virtual environment:**

    ```bash
    python -m venv venv
    ```

2.  **Activate virtual environment:**

    ```bash
    source venv/bin/activate # For Linux/macOS
    venv\Scripts\activate # For Windows
    ```

3.  **Install packages:**

    ```bash
    pip install -r requirement.txt
    ```

4.  **Add a vector database:**

    *   Edit the file `config/embedding_config.py`. This file controls how your documents are processed and how embeddings are generated (either locally or via API).

    Here's a breakdown of the editable parameters in the file:

    *   `model_name`: This specifies the pre-trained model used for creating the embedding vectors. The example shows `"Lajavaness/bilingual-embedding-large"`, but you can choose a different model name depending on your needs.
    *   `vector_db`: This defines the type of vector database to use. Currently, only `'chromaDB'` is supported.
    *   `collection_name`: This specifies the name of the collection within the vector database where the embeddings will be stored. You can choose a name that suits your project.
    *   `raw_db`: This is the root directory where your raw documents are stored. Edit this path to point to your actual data location. For example: `raw_db = "/path/to/my/data"`
    *   `data_language`: This specifies the language of your data. The file provides a list of supported languages. Choose the one that matches your data.
    *   `db_directory`: This defines the location where the vector database will be stored. By default, it's set to the user's home directory under a `.db` folder. You can change this path to a different location if needed.
    *   `chunk_size`: This determines the number of sentences processed together when creating the vector database. You can adjust this value based on your data size and hardware capabilities.
    *   `overlap_size`: This determines the number of sentences overlaped between the chunk and the next chunk. This is useful to not lose semantic of chunks when splitting the text. The value must be lower than the chunk_size.
    *    `use_openai_embeddings`: This allows you to choose if choosing openai api for embedding or local sentence transformer model is set to True. If set to True, it will ignore `model_name` and uses `openai_embedding_model` instead
    *   `chunking_method`: This selects how documents are split into chunks before embedding. Allowed values are `"sentence"` and `"semantic"`:
        *   `"sentence"` (default): The original rule-based method. Sentences are grouped into fixed-size, overlapping windows using `chunk_size` and `overlap_size`. Fast and deterministic, but it can split a single idea across two chunks.
        *   `"semantic"`: An embedding-based method. Every sentence is embedded and a new chunk is started whenever the topic shifts (the similarity between consecutive sentences drops below a data-driven threshold). This keeps semantically related sentences together and tends to produce more coherent chunks. When this method is selected, `chunk_size` and `overlap_size` are ignored and the `semantic_*` settings below are used instead. It reuses your configured embedding backend (local SentenceTransformer or OpenAI compatible API), so embedding the document costs extra calls at indexing time.
    *   `semantic_breakpoint_percentile`: (semantic only) Percentile (0-100) of the consecutive-sentence distances used as the split threshold. A higher value (e.g. `95`) yields fewer, larger chunks; a lower value yields more, smaller chunks.
    *   `semantic_buffer_size`: (semantic only) Number of neighbouring sentences combined with each sentence to give it context before embedding. `0` embeds each sentence on its own; `1` is a good default.
    *   `semantic_max_chunk_sentences`: (semantic only) Optional hard cap on the number of sentences per chunk, useful to avoid a single very large chunk. `0` disables the cap.

    Example `embedding_config.py` (Remember to adapt these values to your specific setup):

    ```python
    import os

    model_name = "Lajavaness/bilingual-embedding-large"  

    # settings if using openai embeddings api or any openai compatible embedding api
    use_openai_embeddings = False # set to True if you want to use openai embeddings api
    openai_embedding_model = "embedding-model-name" # openai embedding model to use if use_openai_embeddings is set to True
    openai_embedding_base_url = 'https://api.openai.com/v1' # or openai compaible api base url 

    vector_db = "chromaDB"

    collection_name = "my_rag_collection"

    raw_db = "/path/to/my/data"  # Replace with the actual path

    data_language = "english"

    db_directory = os.path.join(os.path.expanduser('~'), '.my_rag_db')

    chunk_size = 20

    overlap_size = 5

    # Chunking strategy: "sentence" (rule-based) or "semantic" (embedding-based)
    chunking_method = "sentence"

    # Semantic chunking settings (only used when chunking_method == "semantic")
    semantic_breakpoint_percentile = 95
    semantic_buffer_size = 1
    semantic_max_chunk_sentences = 0
    ```

5.  **Create vector database:**

    *   After setting the data paths in `embedding_config.py`, run the following command to create the vector database:

    ```bash
    python embedding/vector_db_setup.py
    ```

    *   This will create a Chroma vector database using the configurations you provided.

6.  **LLM configurations (ollama or openai):**

    *   The file `config/llm_config.py` allows you to configure the large language model (LLM) used for text generation. You can specify the LLM and potentially edit the prompts used for generating text. This config file also allows you to choose between running ollama or openai api models. You can also choose to record chat log of users with the record_data variable. Here is an example file:
    ```python
    llm_model = 'deepseek-r1:1.5b' # select any model available on the ollama site https://ollama.com/search

    use_openai = False # set to True if using openai api and then select 'openai_model' variable

    openai_model = 'gpt-4o' # if using openai api then select which model to use

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

    record_data = False # set to true to record chat log
    ```

7.  **Run the system:**

    *   Once the vector database is created, you can run the chat and search functionalities using either the Django web app or the command-line tools.

    *   To run the Django app:

    ```bash
    python django-server/manage.py runserver
    ```

    *   This will start the Django development server, allowing you to access the web interface for chat and search (usually at `http://127.0.0.1:8000/` in your web browser). Check Django deployment for deployment

    *   To use the command-line tools:

    *   The functionalities are likely defined in the `cl-tools` directory (chat.py and search.py). You can refer to those files to understand how to use the command-line interface for chat and search.

