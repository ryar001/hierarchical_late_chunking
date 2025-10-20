
import os
from dotenv import load_dotenv
# Import components
from components.embeddings_llm.jina_embedding_model import JinaEmbeddingModel
from components.chroma_db import ChromaDb
from components.data_structures import RetrievalDoc, GraphState
from components.hierarchy_late_chunk import HierarchyLateChunk
from components.llm.gemini_llm import GeminiLLM
from components.utils import mean_pool, fuse_vectors, sliding_chunks, _which_section, _pack_results

load_dotenv()
JINA_API_KEY = os.environ.get("JINA_API_KEY")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")

# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Dependencies (inject)
    try:
        emb = JinaEmbeddingModel(model_name="jina-embeddings-v2-base-en", api_key=JINA_API_KEY)
        llm = GeminiLLM(api_key=GOOGLE_API_KEY)
    except (ImportError, ValueError) as e:
        print(f"Error: {e}")
        print("\nPlease ensure all required packages are installed and API keys are set. You can install packages using:\n  uv pip install jina chromadb langgraph docling google-generativeai")
        exit()

    vdb = ChromaDb(persist_directory="./chroma_store")

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
        # Use a dummy query to check for existence based on metadata
        existing_docs = vdb.query_by_text(
            collection=pipeline.sections_collection,
            query_text="*",
            n_results=1,
            where={"doc_id": doc_id}
        )
        # The query returns a list of lists, check if the inner list is non-empty
        doc_exists = bool(existing_docs.get("ids", [[]])[0])
    except Exception:
        # This can happen if the collection doesn't exist yet.
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
        answer = pipeline.run(q)
        print("\nFinal Answer:\n", answer)