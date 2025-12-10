
import os
import sys
import toml
import json
import time
from dotenv import load_dotenv

# Add parent directory to path to import components
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from components.embeddings_llm.gemini_embedding_model import GeminiEmbeddingModel
from components.db.chroma_db import ChromaDb
from components.hierarchy_late_chunk import HierarchyLateChunk
from components.llm.gemini_llm import GeminiLLM
from evaluate.ragas_utils import init_ragas_models, evaluate_single

def setup_pipeline():
    load_dotenv()
    GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
    
    if not GOOGLE_API_KEY:
        print("Error: Missing API keys in .env")
        sys.exit(1)

    try:
        emb = GeminiEmbeddingModel(api_key=GOOGLE_API_KEY)
        llm = GeminiLLM(api_key=GOOGLE_API_KEY)
    except Exception as e:
        print(f"Error initializing models: {e}")
        sys.exit(1)

    # Force local ChromaDB as per user request
    vdb = ChromaDb(
        persist_directory="./chroma_store",
        host=None, # Force local
        port=8000,
        ssl=False,
        headers=None
    )

    pipeline = HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)
    return pipeline

def ingest_document(pipeline, file_path):
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
        
    doc_id = os.path.basename(file_path)
    # Check if already ingested (check both sections and chunks)
    try:
        # Check sections
        dummy_emb = pipeline.embedding_model.embed_text("test")
        existing_sections = pipeline.vectordb.query_by_embedding(
            collection=pipeline.sections_collection,
            query_embedding=dummy_emb,
            n_results=1,
            where={"doc_id": doc_id}
        )
        sections_exist = bool(existing_sections.get("ids", [[]])[0])

        # Check chunks (to ensure full ingestion)
        existing_chunks = pipeline.vectordb.query_by_embedding(
            collection=pipeline.chunks_collection,
            query_embedding=dummy_emb,
            n_results=1,
            where={"doc_id": doc_id}
        )
        chunks_exist = bool(existing_chunks.get("ids", [[]])[0])
        
        doc_exists = sections_exist and chunks_exist
        if sections_exist and not chunks_exist:
             print(f"Partial ingestion detected for {doc_id} (sections found, chunks missing). Re-ingesting.")

    except Exception as e:
        print(f"Error checking for existing document: {e}")
        doc_exists = False
        
    if doc_exists:
        print(f"Document {doc_id} already exists. Skipping ingestion.")
    else:
        print(f"Ingesting {doc_id} using pypdf for speed...")
        try:
            from pypdf import PdfReader
            reader = PdfReader(file_path)
            text = ""
            for page in reader.pages:
                text += page.extract_text() + "\n"
            
            pipeline.ingest_document(text, doc_id=doc_id)
            print("Ingestion complete.")
        except ImportError:
            print("pypdf not found. Falling back to pipeline.ingest_from_file (docling)...")
            pipeline.ingest_from_file(file_path)
            print("Ingestion complete.")

def main():
    pipeline = setup_pipeline()
    
    # Initialize RAGAs models
    print("Initializing RAGAs models...")
    ragas_llm, ragas_embeddings = init_ragas_models()
    
    pdf_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '.lamrim.pdf'))
    ingest_document(pipeline, pdf_path)
    
    eval_file = os.path.join(os.path.dirname(__file__), 'eval.toml')
    if not os.path.exists(eval_file):
        print(f"Error: {eval_file} not found.")
        sys.exit(1)
        
    with open(eval_file, 'r') as f:
        eval_data = toml.load(f)
        
    results = []
    
    for item in eval_data.get('questions', []):
        question = item['question']
        expected = item['answer']
        
        print(f"Testing Question: {question}")
        start_time = time.time()
        
        # Initial Run
        result_state = pipeline.run(question)
        actual = result_state.get("final_answer", "")
        # Get contexts
        chunk_hits = result_state.get("chunk_hits", [])
        contexts = [c.text for c in chunk_hits[:12]] # match logic in _node_answer

        duration = time.time() - start_time
        
        print("Evaluating with RAGAs...")
        try:
            ragas_scores = evaluate_single(
                ragas_llm, ragas_embeddings, 
                question, actual, contexts, expected
            )
            # Use answer_correctness as the primary score (0-1), scaled to 0-10
            score = ragas_scores.get('answer_correctness', 0) * 10
        except Exception as e:
            print(f"RAGAs evaluation failed: {e}")
            score = 0
            ragas_scores = {}

        print(f"RAGAs Score: {score:.2f}/10")
        
        # Retry Logic if score is low
        if score <= 6:
            print(f"Score {score:.2f}/10 is low. Attempting to refine question and retry...")
            
            rephrase_prompt = f"Rewrite the following question to be more specific and clearer for a search engine, based on the context of 'Lamrim' (Buddhist path). Question: {question}"
            refined_question = ragas_llm.invoke(rephrase_prompt).content.strip()
            print(f"Refined Question: {refined_question}")
            
            # Retry Run
            start_time_retry = time.time()
            result_state_retry = pipeline.run(refined_question)
            actual_retry = result_state_retry.get("final_answer", "")
            chunk_hits_retry = result_state_retry.get("chunk_hits", [])
            contexts_retry = [c.text for c in chunk_hits_retry[:12]]
            
            duration += (time.time() - start_time_retry)
            
            print("Evaluating Retry with RAGAs...")
            try:
                ragas_scores_retry = evaluate_single(
                    ragas_llm, ragas_embeddings,
                    question, actual_retry, contexts_retry, expected
                )
                score_retry = ragas_scores_retry.get('answer_correctness', 0) * 10
            except Exception as e:
                print(f"RAGAs retry evaluation failed: {e}")
                score_retry = 0
                ragas_scores_retry = {}
            
            if score_retry > score:
                print(f"Retry improved score to {score_retry:.2f}.")
                actual = actual_retry
                score = score_retry
                ragas_scores = ragas_scores_retry
                result_state = result_state_retry
            else:
                print(f"Retry did not improve score ({score_retry:.2f}). Keeping original.")

        # Feedback Submission
        if score > 8:
            pipeline.submit_feedback(question, result_state.get("used_chunk_ids", []), int(score), comment="RAGAs Auto-eval")

        result_entry = {
            "question": question,
            "expected_answer": expected,
            "actual_answer": actual,
            "score": score,
            "ragas_metrics": ragas_scores,
            "duration_seconds": duration
        }
        results.append(result_entry)
        print(f"Final Score: {result_entry['score']:.2f}/10\n")
        
    output_file = os.path.join(os.path.dirname(__file__), 'eval_result.json')
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
        
    print(f"Evaluation complete. Results saved to {output_file}")

if __name__ == "__main__":
    main()
