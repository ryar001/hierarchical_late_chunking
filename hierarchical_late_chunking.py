from __future__ import annotations
from typing import List
import os

# LangGraph
from langgraph.graph import StateGraph, END

# Vector DB (PyPI: chromadb)
import chromadb
from chromadb.api.models.Collection import Collection

# Import components
from components.llm_interface import LLMInterface
from components.embedding_interface import EmbeddingInterface
from components.vector_db_interface import VectorDbInterface
from components.jina_embedding_model import JinaEmbeddingModel
from components.chroma_db import ChromaDb
from components.data_structures import RetrievalDoc, GraphState
from components.hierarchy_late_chunk import HierarchyLateChunk
from components.dummy_llm import DummyLLM
from components.utils import mean_pool, fuse_vectors, sliding_chunks, _which_section, _pack_results


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Dependencies (inject)
    # NOTE: You will need to install jina to run this: `pip install jina`
    try:
        # Docling for file loading (PyPI: docling-core)
        from docling.loader import DoclingLoader
        emb = JinaEmbeddingModel(model_name="jina-embeddings-v2-base-en")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages: `pip install jina chromadb langgraph docling-core`")
        exit()
        
    vdb = ChromaDb(persist_directory="./chroma_store")
    llm = DummyLLM()

    pipeline = HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)

    # --- NEW: Ingest from a file ---
    
    # 1. Create a dummy file to test the ingestion
    dummy_file_path = "physics_chapter.txt"
    print(f"Creating a dummy file for testing: {dummy_file_path}")
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write(
            "Chapter 4: Dynamics and Circular Motion\n"
            "Newton’s second law states that force equals mass times acceleration (F=ma). "
            "In uniform circular motion, acceleration is v^2/r toward the center. Therefore, the net force "
            "required is mv^2/r. This follows from combining the kinematics of circular motion with Newton’s laws. "
            "Examples include satellites orbiting Earth, cars taking turns, and pendulums at small angles. "
            * 40  # repeat to make it long enough for multiple sections
        )

    # 2. Ingest the document directly from the file path
    # This single call now handles loading, text extraction, and processing.
    # It would work the same for a PDF: pipeline.ingest_from_file("my_report.pdf")
    info = pipeline.ingest_from_file(dummy_file_path)
    print("\nIngestion complete:", info)

    # 3. Ask a question
    q = "How does Newton’s second law apply to circular motion?"
    print("\n--- Running Query ---")
    print(f"Question: {q}")
    answer = pipeline.run(q)
    print("\nFinal Answer:\n", answer)
