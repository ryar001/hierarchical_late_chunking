from typing import Any, Dict, List, Optional
from dataclasses import dataclass

@dataclass
class RetrievalDoc:
    id: str
    text: str
    metadata: Dict[str, Any]
    embedding: Optional[List[float]] = None


from typing import TypedDict

class GraphState(TypedDict, total=False):
    query: str
    sub_queries: List[str]
    section_hits: List[RetrievalDoc]
    chunk_hits: List[RetrievalDoc]
    final_answer: str
