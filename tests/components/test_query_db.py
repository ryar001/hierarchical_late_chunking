
import os
import pytest
from dotenv import load_dotenv
from components.embeddings_llm.gemini_embedding_model import GeminiEmbeddingModel
from components.db.chroma_db import ChromaDb
from components.hierarchy_late_chunk import HierarchyLateChunk
from components.llm.gemini_llm import GeminiLLM

# Load environment variables
load_dotenv()

@pytest.fixture(scope="module")
def pipeline():
    """Initializes the RAG pipeline for testing."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    assert google_api_key, "GOOGLE_API_KEY must be set in environment variables."

    emb = GeminiEmbeddingModel(api_key=google_api_key)
    llm = GeminiLLM(api_key=google_api_key)

    # ChromaDB Config
    chroma_host = os.environ.get("CHROMA_HOST")
    chroma_port = int(os.environ.get("CHROMA_PORT", 8000))
    chroma_token = os.environ.get("CHROMA_TOKEN")
    chroma_ssl = os.environ.get("CHROMA_SSL", "False").lower() == "true"

    chroma_headers = None
    if chroma_token:
        chroma_headers = {"X-Chroma-Token": chroma_token}

    try:
        if chroma_host:
             vdb = ChromaDb(
                persist_directory="./chroma_store",
                host=chroma_host,
                port=chroma_port,
                ssl=chroma_ssl,
                headers=chroma_headers
            )
             vdb.client.heartbeat()
             print("Connected to remote ChromaDB")
        else:
             raise Exception("No host provided, forcing local")
    except Exception as e:
        print(f"Using local ChromaDB: {e}")
        vdb = ChromaDb(
            persist_directory="./chroma_store",
            host=None,
            port=8000,
            ssl=False,
            headers=None
        )

    return HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)

def test_query_flow(pipeline):
    """
    Tests the full query flow:
    1. Checks if 'test_pdf.pdf' is ingested.
    2. Ingests it if missing.
    3. Runs a sample query.
    4. Verifies an answer is returned.
    """
    
    # Path to the PDF
    pdf_path = os.path.abspath("uploads/test_pdf.pdf")
    doc_id = "test_pdf.pdf"

    # 1. Check if document exists
    print(f"\nChecking existence of {doc_id}...")
    try:
        # Use a dummy embedding to check existence
        dummy_emb = pipeline.embedding_model.embed_text("test")
        existing_docs = pipeline.vectordb.query_by_embedding(
            collection=pipeline.sections_collection,
            query_embedding=dummy_emb,
            n_results=1,
            where={"doc_id": doc_id}
        )
        # Check if we got any IDs back
        is_ingested = False
        if existing_docs and "ids" in existing_docs and existing_docs["ids"]:
             if existing_docs["ids"][0]: # Check if the list of ids is not empty
                 is_ingested = True
    except Exception as e:
        print(f"Error checking document existence: {e}")
        is_ingested = False

    # 2. Ingest if missing
    if not is_ingested:
        print(f"Document {doc_id} not found. Ingesting from {pdf_path}...")
        assert os.path.exists(pdf_path), f"Test file not found at {pdf_path}"
        
        pipeline.ingest_from_file(pdf_path, doc_id=doc_id)
        print("Ingestion complete.")
    else:
        print(f"Document {doc_id} already exists. Skipping ingestion.")

    # 3. Run Query
    query = "What is the main topic of this document?" # Generic query
    print(f"Running query: {query}")
    
    result_state = pipeline.run(query)
    final_answer = result_state.get("final_answer", "")
    
    print(f"Answer: {final_answer}")

    # 4. Verify Answer
    assert final_answer, "Pipeline returned an empty answer."
    assert "error" not in final_answer.lower(), "Pipeline returned an error in the answer."
