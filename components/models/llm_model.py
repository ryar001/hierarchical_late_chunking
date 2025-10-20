from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

# GeminiLLM Models
class GenerateOutput(BaseModel):
    content: str = Field(..., description="The generated content from the LLM.")

class SummarizeOutput(BaseModel):
    summary: str = Field(..., description="The summarized text.")

class ExpandQueryOutput(BaseModel):
    expanded_queries: List[str] = Field(..., description="A list of expanded or related search queries.")

class AnswerOutput(BaseModel):
    answer: str = Field(..., description="The answer to the question based on the provided context.")

# JinaLLM Models

class EmbeddingData(BaseModel):
    embedding: List[float] = Field(..., description="The embedding vector for the input.")
    index: int = Field(..., description="The index of the input.")
    object: str = Field(..., description="The object type, e.g., 'embedding'.")

class EmbeddingUsage(BaseModel):
    prompt_tokens: int = Field(..., description="Number of tokens in the prompt.")
    total_tokens: int = Field(..., description="Total number of tokens processed.")

class EmbedOutput(BaseModel):
    data: List[EmbeddingData] = Field(..., description="A list of embedding data objects.")
    model: str = Field(..., description="The name of the model used for embedding.")
    object: str = Field(..., description="The object type, e.g., 'list'.")
    usage: EmbeddingUsage = Field(..., description="Token usage statistics.")

class RerankResult(BaseModel):
    document: Dict[str, Any] = Field(..., description="The original document.")
    index: int = Field(..., description="The original index of the document.")
    relevance_score: float = Field(..., description="The relevance score of the document to the query.")

class RerankUsage(BaseModel):
    total_tokens: int = Field(..., description="Total number of tokens processed.")

class RerankOutput(BaseModel):
    model: str = Field(..., description="The name of the reranker model used.")
    results: List[RerankResult] = Field(..., description="A list of reranked documents.")
    usage: RerankUsage = Field(..., description="Token usage statistics.")

class ReadOutput(BaseModel):
    content: str = Field(..., description="The parsed content from the URL.")
    url: str = Field(..., description="The URL from which the content was read.")
    # Add other fields if the 'read' method returns more structured data.

class SearchResult(BaseModel):
    url: str = Field(..., description="The URL of the search result.")
    title: str = Field(..., description="The title of the search result.")
    description: str = Field(..., description="A brief description of the search result.")
    # Add other fields as necessary.

class SearchOutput(BaseModel):
    results: List[SearchResult] = Field(..., description="A list of search results.")
    # Add other fields if the 'search' method returns more structured data.

class DeepSearchOutput(BaseModel):
    # Define the structure based on the actual API response for deepsearch
    # This is a placeholder as the exact structure is not detailed.
    response: Dict[str, Any] = Field(..., description="The response from the deepsearch operation.")

class Segment(BaseModel):
    text: str = Field(..., description="The text of the segment.")
    # Add other segment-related fields if available.

class SegmentOutput(BaseModel):
    segments: List[Segment] = Field(..., description="A list of text segments.")
    # Add other fields if the 'segment' method returns more structured data.

class ClassificationResult(BaseModel):
    label: str = Field(..., description="The predicted label for the input.")
    score: float = Field(..., description="The confidence score for the prediction.")

class ClassifyOutput(BaseModel):
    results: List[ClassificationResult] = Field(..., description="A list of classification results for the inputs.")
    # Add other fields if the 'classify' method returns more structured data.
