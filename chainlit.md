# Welcome to Hierarchical Late Chunking RAG! 🚀

This application demonstrates a **Hierarchical Late Chunking** Retrieval-Augmented Generation (RAG) system.

## Features:
- **Hierarchical Indexing**: Documents are split into large sections and smaller chunks for better context.
- **Late Chunking / Fusion**: Uses advanced embedding techniques (Token-level or Global Fusion).
- **LangGraph Pipeline**: Orchestrates query expansion, section retrieval, and chunk retrieval.

## How to use:
## Start
```
docker run -it --name hierarchical-rag -p 8000:8000 -v $(pwd):/app hierarchical-rag
```
1. **Upload a PDF**: Click the attachment icon to upload a document (e.g., `lamrim.pdf`). Wait for ingestion to complete.
2. **Ask Questions**: Type your query in the chat. The system will retrieve relevant context and generate an answer.
