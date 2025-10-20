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
    The project uses `uv` for package management.
    ```bash
    uv pip install -r requirements.txt 
    ```
    Or, if you know the dependencies:
    ```bash
    uv pip install jina chromadb langgraph docling python-dotenv google-generativeai
    ```

3.  **Set up environment variables:**
    Create a `.env` file in the project root and add your API keys:
    ```
    JINA_API_KEY="your_jina_api_key_here"
    GOOGLE_API_KEY="your_google_api_key_here"
    ```

## How to Run

The main script `hierarchical_late_chunking.py` provides a complete example.

1.  **Place a document:** Add a PDF file named `lamrim.pdf` to the root directory. If the script doesn't find it, it will automatically create and use a dummy text file (`dummy.txt`) for demonstration.

2.  **Execute the script:**
    ```bash
    python hierarchical_late_chunking.py
    ```

**Expected Output:**

The script will:
1.  Print a message indicating it is loading and ingesting the document.
2.  Show the results of the ingestion, including the number of sections and chunks created.
3.  Run a hardcoded example query: "How does Newton’s second law apply to circular motion?"
4.  Print the final answer generated by the pipeline.

```
--- Loading document from: dummy.txt ---
--- Processing text for document ID: dummy.txt ---
Token-level embeddings not available. Using global-fusion fallback.

Ingestion complete: {'doc_id': 'dummy.txt', 'num_sections': 1, 'num_chunks': 5}

--- Running Query ---
Question: How does Newton’s second law apply to circular motion?

Final Answer:
[An answer generated by the DummyLLM will appear here]
```

## Key Components

*   **`HierarchyLateChunk`:** The main class in `components/hierarchy_late_chunk.py`. It orchestrates the entire ingestion and retrieval workflow.
*   **`JinaEmbeddingModel`:** Implements the `EmbeddingInterface` to provide document, text, and (theoretically) token embeddings using the Jina AI API.
*   **`ChromaDb`:** A simple wrapper around the `chromadb` client to handle creating collections and adding/querying documents.
*   **`DummyLLM`:** A placeholder that simulates the behavior of a real Large Language Model for summarizing, expanding queries, and generating final answers. This allows the RAG pipeline to be tested without a real LLM dependency.
*   **`langgraph`:** The framework used to define the retrieval process as a graph of connected nodes. Each node in the graph (`expand`, `sec_retrieve`, `chunk_retrieve`, `answer`) performs one step of the process.
*   **`docling`:** A utility library used in the ingestion step to robustly extract text content from various file formats like PDFs.
