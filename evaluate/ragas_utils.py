
import os
from datasets import Dataset
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.embeddings import JinaEmbeddings
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_recall, 
    answer_correctness
)

def init_ragas_models():
    """Initialize LLM and Embedding models for RAGAs."""
    google_api_key = os.environ.get("GOOGLE_API_KEY")
    jina_api_key = os.environ.get("JINA_API_KEY")

    if not google_api_key:
        raise ValueError("GOOGLE_API_KEY not found in environment")
        
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-flash", 
        google_api_key=google_api_key,
        temperature=0
    )

    embeddings = None
    if jina_api_key:
         embeddings = JinaEmbeddings(
            jina_api_key=jina_api_key,
            model_name="jina-embeddings-v2-base-en"
        )
    return llm, embeddings

def evaluate_single(llm, embeddings, question, answer, contexts, ground_truth):
    """
    Run RAGAs evaluation on a single sample.
    Returns a dictionary of scores.
    """
    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [contexts],         # contexts must be a list of strings
        "ground_truth": [ground_truth]  # ground_truth must be a string
    }
    dataset = Dataset.from_dict(data)

    metrics = [faithfulness, answer_relevancy, context_recall, answer_correctness]

    results = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=llm,
        embeddings=embeddings
    )
    
    # Results is a dictionary-like object, get the first (and only) row
    return results[0] # Ragas results are usually indexable
