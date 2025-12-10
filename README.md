# Hierarchical Late-Chunking RAG

This project implements an advanced Retrieval-Augmented Generation (RAG) pipeline using a "Hierarchical Late-Chunking" strategy. It's designed to provide more accurate and context-aware answers from large documents by retrieving information from both coarse and fine-grained document segments.

The pipeline is built using `langgraph` to create a stateful, multi-step retrieval process, with `ChromaDB` as the vector store and `Jina` for generating embeddings.

## Table of Contents
- [Core Concepts](#core-concepts)
- [How It Works](#how-it-works)
  - [Ingestion](#ingestion)
  - [Retrieval](#retrieval)
- [Project Structure](#project-structure)
- [Setup and Installation](#setup-and-installation)
- [How to Run](#how-to-run)
- [Key Components](#key-components)

## Core Concepts

Standard RAG often struggles with finding the right "chunk" of text. If chunks are too small, they lack context. If they're too large, they contain too much noise. This project addresses that with two key ideas:

1.  **Hierarchical Retrieval:** Instead of a single flat list of chunks, we create a hierarchy:
    *   **Sections:** Large, coarse-grained parts of the document. These are good for understanding the high-level topic of a query.
    *   **Chunks:** Smaller, fine-grained pieces of text nested within sections. These are good for providing specific, targeted information for an answer.

2.  **Late Chunking / Fusion-in-Decoder:** This is a technique for creating more contextually aware chunk embeddings. Instead of just embedding the small chunk text on its own, this implementation attempts to create embeddings from the document's tokens directly. As a fallback, it uses a "global-fusion" method, where the embedding for a chunk is a weighted average of its own vector and the vector of the entire document. This helps the chunk "remember" its place and context within the larger document.

## How It Works

The system is split into two main phases: Ingestion and Retrieval.

### Ingestion

The `ingest_from_file` method orchestrates the entire process:

1.  **Load Document:** It uses the `docling` library to load and extract raw text from a file (e.g., PDF, TXT, DOCX).
2.  **Create Sections:** The document is split into large, overlapping sections (e.g., 2000 tokens).
3.  **Summarize & Embed Sections:** A language model (currently `DummyLLM`) summarizes each section. The *summary* is then embedded using the Jina embedding model. This creates a compact, high-level vector for each major part of the document.
4.  **Store Sections:** The section summaries, their embeddings, and metadata are stored in a dedicated `ChromaDB` collection (`rag_sections`).
5.  **Create Chunks:** The document is also split into smaller, overlapping chunks (e.g., 480 tokens).
6.  **Embed Chunks (Late-Chunking):**
    *   **Ideal Path:** It attempts to generate embeddings for every token in the document and then pools these token-vectors together to form a vector for each chunk.
    *   **Fallback Path:** If token-level embeddings aren't available, it generates a vector for the entire document and a separate vector for each chunk. It then "fuses" these two vectors (defaulting to 80% chunk vector, 20% document vector) to create the final chunk embedding.
7.  **Store Chunks:** The chunk text, their fused embeddings, and metadata (including which section they belong to) are stored in a second `ChromaDB` collection (`rag_chunks`).

### Retrieval

When a query is made via the `run` method, a `langgraph` state machine executes the following steps:

1.  **Query Expansion:** The initial query is expanded into a set of sub-queries to broaden the search.
2.  **Section Retrieval:** The query is used to search the `rag_sections` collection. The top 3 most relevant **sections** are retrieved. This narrows down the search space to the most relevant parts of the document.
3.  **Chunk Retrieval:** The query is then used to search for chunks *only within the sections that were previously retrieved*. This two-step process is much more efficient and accurate than searching all chunks at once.
4.  **Generate Answer:** The text from the top-retrieved chunks is compiled into a context. This context and the original query are passed to the LLM, which generates the final answer.

## Project Structure

```
/
├───.env                  # For API keys (JINA_API_KEY)
├───hierarchical_late_chunking.py # Main script with example usage
├───components/
│   ├───hierarchy_late_chunk.py # Core class for the RAG pipeline
│   ├───chroma_db.py          # Wrapper for ChromaDB vector store
│   ├───data_structures.py    # Defines data classes like RetrievalDoc and GraphState
│   ├───dummy_llm.py          # A placeholder LLM for summarization and answering
│   ├───embedding_interface.py # Abstract base class for embedding models
│   ├───vector_db_interface.py # Abstract base class for vector databases
│   └───embeddings_llm/
│       └───jina_embedding_model.py # Jina embedding model implementation
└───tests/
    └───...
```

## Setup and Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd hierarchical_late_chunking
    ```

2.  **Install dependencies:**
    This project is optimized for `uv` (a fast Python package installer), but `pip` also works.
    
    Using `uv` (Recommended):
    ```bash
    pip install uv
    uv pip install -r requirements.txt
    ```
    
    Or, if you know the dependencies:
    ```bash
    uv pip install chromadb langgraph docling python-dotenv google-generativeai
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root and add your keys.

    ```env
    # Google Gemini API Key
    GOOGLE_API_KEY="your_google_api_key_here"

    # --- Database Configuration ---
    
    # OPTION 1: Local Storage (Default)
    # Leave all CHROMA_* variables below empty or commented out.
    # Data will be stored locally in ./chroma_store directory.

    # OPTION 2: Chroma Cloud (Managed Service)
    # Uncomment and fill these if using Chroma Cloud
    # CHROMA_TOKEN="your_cloud_api_key"
    # CHROMA_TENANT="default_tenant"
    # CHROMA_CLOUD_DATABASE="default_database"

    # OPTION 3: Remote Server (Self-Hosted)
    # Uncomment and fill these if running your own Chroma server (e.g., Docker)
    # CHROMA_HOST="localhost" 
    # CHROMA_PORT=8000
    # CHROMA_SSL=False
    # CHROMA_TOKEN="" # Optional: specific token for self-hosted auth
    ```

## How to Run

### CLI Mode (Testing/Ingestion)

The main script `hierarchical_late_chunking.py` allows you to ingest documents and run queries in the terminal.

1.  **Place a document:** 
    The script looks for `tests/components/test_files/test_pdf.pdf` by default. If not found, it creates `dummy.txt`. You can modify the script to point to your own file.

2.  **Execute the script:**
    ```bash
    python hierarchical_late_chunking.py
    ```
    
    The script will:
    - Check if the document exists in the DB.
    - Ask if you want to re-ingest.
    - Start an interactive Q&A loop.

### Running the Frontend (Chat UI)

You can launch an interactive chat interface using `chainlit`.

```bash
chainlit run app.py
```

**Note**: The app runs on port 8000 by default.

**Expected Output:**
The UI will initialize the pipeline and allow you to upload PDFs or ask questions about pre-ingested data.


### Running with Docker

You can also run the application (Chainlit frontend) using Docker.

1.  **Run the container:**
    ```bash
    docker run -p 8000:8000 --env-file .env -v $(pwd)/my_rag_data:/app/chroma_store jokerssd/hierarchical-rag
    ```
    - my_rag_data: This is the directory where the chromadb data will be stored.
        - Will be created at the directory this command is ran at if it doesn't exist.

    Ensure your `.env` file is properly configured with `GOOGLE_API_KEY`.

2.  **Viewing Logs**
    To view the application's logs while it's running in Docker:
    ```bash
    docker logs -f <container_id_or_name>
    ```
    - Use `docker ps` to find the container's ID or name.
    - The `-f` flag "follows" the log output in real-time.

    For more details on persistence and cloud deployment, see [DEPLOY_DOCKER.md](DEPLOY_DOCKER.md).

## Key Components

*   **`HierarchyLateChunk`:** The main class in `components/hierarchy_late_chunk.py`. It orchestrates the entire ingestion and retrieval workflow.
*   **`GeminiEmbeddingModel`:** Implements the `EmbeddingInterface` to provide embeddings using Google's Gemini API.
*   **`ChromaDb`:** A wrapper around the `chromadb` client to handle creating collections and adding/querying documents.
*   **`GeminiLLM`:** The primary Large Language Model used for summarization, query expansion, and generating final answers.
*   **`langgraph`:** The framework used to define the retrieval process as a graph of connected nodes. Each node in the graph (`expand`, `sec_retrieve`, `chunk_retrieve`, `answer`) performs one step of the process.
*   **`docling`:** A utility library used in the ingestion step to robustly extract text content from various file formats like PDFs.


## Docker Deployment

```