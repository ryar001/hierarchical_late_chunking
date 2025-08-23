from typing import List, Optional, Protocol

class EmbeddingInterface(Protocol):
    """
    Embedding contract.
    Late chunking needs token-level vectors; if unavailable, implement a fallback.
    """
    def embed_text(self, text: str) -> List[float]: ...
    def embed_documents(self, docs: List[str]) -> List[List[float]]: ...
    def embed_tokens(self, text: str) -> Optional[List[List[float]]]: ...
