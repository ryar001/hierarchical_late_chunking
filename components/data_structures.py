from typing import Any, Dict, List, Optional, TypedDict
from dataclasses import dataclass

@dataclass
class RetrievalDoc:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None
    score: Optional[float] = None


class GraphState(TypedDict):
    query: str
    sub_queries: List[str]
    hypothetical_answer: Optional[str]
    section_hits: List[RetrievalDoc]
    chunk_hits: List[RetrievalDoc]
    final_answer: str
    used_chunk_ids: List[str]
    query_embedding: Optional[List[float]]
    hyde_embedding: Optional[List[float]]
    expanded_embeddings: Optional[List[List[float]]]
    doc_ids: Optional[List[str]]
