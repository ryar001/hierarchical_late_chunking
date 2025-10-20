# Project Plan: Hierarchical Late Chunking RAG

This document outlines the architecture and workflow of the Hierarchical Late Chunking Retrieval-Augmented Generation (RAG) system.

## 1. Project Overview

The project implements a sophisticated RAG pipeline that uses a two-tiered retrieval strategy: first identifying relevant large sections of a document and then pinpointing specific chunks within those sections. This "hierarchical" approach aims to improve retrieval accuracy by first getting the "big picture" and then focusing on the details.

The system is designed with a modular, interface-driven architecture, allowing for easy replacement of components like the LLM, embedding model, or vector database.

## 2. Core Components

- **Main Entrypoint (`hierarchical_late_chunking.py`):** Initializes and orchestrates the entire pipeline. It handles dependency injection (LLM, embeddings, DB) and runs the ingestion and retrieval processes.
- **Pipeline (`components/hierarchy_late_chunk.py`):** The central component containing the core logic for both data ingestion and the retrieval graph.
- **Interfaces (`components/*_interface.py`):** Python `Protocol`s define the contracts for the LLM, Embedding Model, and Vector Database, ensuring loose coupling.
- **Data Structures (`components/data_structures.py`):** Defines the key data models, notably `RetrievalDoc` for search results and `GraphState` for managing the flow in the retrieval graph.
- **Implementations:**
    - **LLM:** `GeminiLLM` (`components/llm/gemini_llm.py`) using the Google Gemini API.
    - **Embedding Model:** `JinaEmbeddingModel` (`components/embeddings_llm/jina_embedding_model.py`) using Jina AI embeddings.
    - **Vector Database:** `ChromaDb` (`components/chroma_db.py`) using the local persistent `chromadb` library.

## 3. Data Flow and Architecture Diagram

The process is divided into two main phases: **Ingestion** and **Retrieval**.

```mermaid
graph TD
    subgraph Ingestion Phase
        A[File Document (.pdf, .txt, etc.)] --> B{docling: DocumentConverter};
        B --> C[Raw Text];
        C --> D{Split into Sections};
        C --> E{Split into Chunks};

        D -- "For each section" --> F[LLM: Summarize Section];
        F --> G[Embedding Model: Embed Summary];
        G --> H[(ChromaDB: sections_collection)];

        E -- "For each chunk" --> I{Embedding Model: Create Chunk Embedding};
        I --> J[(ChromaDB: chunks_collection)];
    end

    subgraph Retrieval Phase
        K[User Query] --> L{LangGraph: Retrieval Pipeline};
        L --> M[Node 1: Expand Query];
        M --> N[Node 2: Section Retrieval];
        N -- "Query Embedding" --> H;
        H -- "Top N Section Summaries" --> N;
        N -- "Retrieved Section IDs" --> O[Node 3: Chunk Retrieval];
        O -- "Query Embedding + Section Filter" --> J;
        J -- "Top K Chunks per Section" --> O;
        O -- "Final Context (Top Chunks)" --> P[Node 4: Generate Answer];
        P --> Q[Final Answer];
    end

    %% Styling
    style A fill:#f9f,stroke:#333,stroke-width:2px
    style Q fill:#bbf,stroke:#333,stroke-width:2px
    style H fill:#ff9,stroke:#333,stroke-width:2px
    style J fill:#ff9,stroke:#333,stroke-width:2px
```

### 3.1. Ingestion Flow Explained

1.  **File Input:** The process starts with a file (e.g., `lamrim.pdf`).
2.  **Text Extraction:** `docling` library reads the file and extracts its raw text content.
3.  **Hierarchical Splitting:** The text is split into two levels:
    - **Sections:** Large, coarse-grained segments (~2000 tokens).
    - **Chunks:** Smaller, fine-grained segments (~480 tokens) with overlap.
4.  **Section Processing:**
    - Each section is summarized by the LLM (`GeminiLLM`).
    - The resulting *summary* is embedded.
    - The summary, its embedding, and metadata (like `doc_id`, `section_index`) are stored in the `sections_collection` in ChromaDB. The document stored is the summary, not the full section text.
5.  **Chunk Processing:**
    - Each chunk is embedded. The system attempts to use token-level embeddings for "true" late-chunking. If unavailable, it falls back to a "global-fusion" method where the chunk's embedding is combined with the entire document's embedding.
    - The chunk text, its embedding, and metadata (like `doc_id`, `chunk_index`, and its parent `section_id`) are stored in the `chunks_collection` in ChromaDB.

### 3.2. Retrieval Flow Explained

The retrieval process is a `langgraph` state machine.

1.  **Query Expansion:** The initial user query is expanded into a set of related queries by the LLM to broaden the search.
2.  **Section Retrieval:** The system searches the `sections_collection` for the most relevant section *summaries*. This quickly narrows down the search space to the most promising parts of the document.
3.  **Chunk Retrieval:** Using the `section_id`s from the previous step, the system performs a filtered search on the `chunks_collection`. This retrieves the most relevant chunks *only from within the already identified relevant sections*.
4.  **Answer Generation:** The text from the top-ranked chunks is compiled into a context. The LLM uses this context to generate a final, concise answer to the user's original query.

## 4. Data Storage

-   **Vector Storage:** `ChromaDB` (persistent local storage in `./chroma_store`).
    -   **`sections_collection`**:
        -   **IDs:** `{doc_id}_sec_{i}`
        -   **Documents:** Section summaries (generated by LLM).
        -   **Embeddings:** Vectors of the section summaries.
        -   **Metadata:** `{"type": "section", "doc_id": str, "section_index": int}`
    -   **`chunks_collection`**:
        -   **IDs:** `{doc_id}_ch_{i}`
        -   **Documents:** The raw text of the chunk.
        -   **Embeddings:** Vectors of the chunk text (potentially fused with global vector).
        -   **Metadata:** `{"type": "chunk", "doc_id": str, "chunk_index": int, "section_id": str}`
-   **File Storage:** The original source documents (e.g., `.pdf`, `.txt`) are not stored by the system after ingestion.
