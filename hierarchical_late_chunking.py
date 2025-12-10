
import os
from dotenv import load_dotenv
# Import components
# Import components
from components.embeddings_llm.gemini_embedding_model import GeminiEmbeddingModel
from components.db.chroma_db import ChromaDb

from components.hierarchy_late_chunk import HierarchyLateChunk
from components.llm.gemini_llm import GeminiLLM

load_dotenv()
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Dependencies (inject)
    try:
        emb = GeminiEmbeddingModel(api_key=GOOGLE_API_KEY)
        llm = GeminiLLM(api_key=GOOGLE_API_KEY)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        print("\nPlease ensure all required packages are installed and API keys are set. You can install packages using:\n  uv pip install jina chromadb langgraph docling google-generativeai")
        exit()
    breakpoint()
    # ChromaDB Config
    CHROMA_HOST = os.environ.get("CHROMA_HOST")
    CHROMA_PORT = int(os.environ.get("CHROMA_PORT", 8000))
    CHROMA_TOKEN = os.environ.get("CHROMA_TOKEN")
    CHROMA_SSL = os.environ.get("CHROMA_SSL", "False").lower() == "true"
    
    chroma_headers = None
    if CHROMA_TOKEN:
        chroma_headers = {"X-Chroma-Token": CHROMA_TOKEN}

    try:
        vdb = ChromaDb(
            persist_directory="./chroma_store",
            host=CHROMA_HOST,
            port=CHROMA_PORT,
            ssl=CHROMA_SSL,
            headers=chroma_headers
        )
        # Test connection
        vdb.client.heartbeat()
    except Exception as e:
        print(f"Could not connect to remote ChromaDB ({e}). Falling back to local mode.")
        vdb = ChromaDb(
            persist_directory="./chroma_store",
            host=None,
            port=8000,
            ssl=False,
            headers=None
        )

    pipeline = HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)

    # --- Ingest from a file ---
    # Use test_pdf.pdf if it exists, otherwise create and use a dummy text file.
    dummy_file_path = "dummy.txt"
    file_to_ingest = os.path.join(os.path.dirname(__file__), "tests/components/test_files/test_pdf.pdf") # Adjust path to root
    if not os.path.exists(file_to_ingest):
        print(f"'{file_to_ingest}' not found. Creating a dummy file for testing: {dummy_file_path}")
        file_to_ingest = dummy_file_path
        with open(file_to_ingest, "w", encoding="utf-8") as f:
            f.write(
                "Chapter 4: Dynamics and Circular Motion\n"
                "Newton’s second law states that force equals mass times acceleration (F=ma). "
                "In uniform circular motion, acceleration is v^2/r toward the center. Therefore, the net force "
                "required is mv^2/r. This follows from combining the kinematics of circular motion with Newton’s laws. "
                "Examples include satellites orbiting Earth, cars taking turns, and pendulums at small angles. "
                * 40  # Repeat to make it long enough for multiple sections
            )

    # 2. Check if the document is already stored and ask the user for action.
    doc_id = os.path.basename(file_to_ingest)
    
    try:
        # Use a dummy embedding to check for existence based on metadata
        # We must use query_by_embedding to avoid default embedding that leads to dimension mismatch
        dummy_emb = pipeline.embedding_model.embed_text("test") 
        existing_docs = vdb.query_by_embedding(
            collection=pipeline.sections_collection,
            query_embedding=dummy_emb,
            n_results=1,
            where={"doc_id": doc_id}
        )
        # The query returns a list of lists, check if the inner list is non-empty
        doc_exists = bool(existing_docs.get("ids", [[]])[0])
    except Exception:
        # This can happen if the collection doesn't exist yet.
        # print(f"Check failed, assuming missing: {e}")
        doc_exists = False

    should_ingest = True
    if doc_exists:
        while True:
            response = input(f"Document '{doc_id}' may already be stored. Re-ingest? (y/n): ").lower().strip()
            if response in ['y', 'n']:
                if response == 'n':
                    should_ingest = False
                break
            else:
                print("Invalid input. Please enter 'y' or 'n'.")

    if should_ingest:
        print(f"--- Ingesting document: {doc_id} ---")
        info = pipeline.ingest_from_file(file_to_ingest)
        print("\nIngestion complete:", info)
    else:
        print(f"\nSkipping ingestion. Using existing data for document '{doc_id}'.")

    # 3. Ask questions in a loop
    while True:
        print("\n\nEnter a question to ask the document (or type 'quit' to exit):")
        q = input("> ")
        if q.lower().strip() in ["quit", "exit"]:
            break
        if not q.strip():
            continue

        print("\n--- Running Query ---")
        print(f"Question: {q}")
        result_state = pipeline.run(q)
        answer = result_state.get("final_answer", "")
        print("\nFinal Answer:\n", answer)
        
        # Feedback Loop
        try:
            score_input = input("\nRate this answer (1-5) or press Enter to skip: ").strip()
            if score_input.isdigit():
                score = int(score_input)
                if 1 <= score <= 5:
                    pipeline.submit_feedback(
                        query=q,
                        chunk_ids=result_state.get("used_chunk_ids", []),
                        score=score
                    )
        except Exception as e:
            print(f"Error submitting feedback: {e}")