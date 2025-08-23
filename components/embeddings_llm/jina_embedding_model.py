from typing import List, Optional, Callable
from components.embedding_interface import EmbeddingInterface

try:
    from jina import JinaEmbeddings
except Exception:
    JinaEmbeddings = None

class JinaEmbeddingModel(EmbeddingInterface):
    """
    Jina text embedding wrapper.
    - If token-level embeddings are available in your Jina client, implement embed_tokens.
    - Otherwise, embed_tokens returns None and the pipeline will use a global-fusion fallback.
    """
    def __init__(self, model_name: str = "jina-embeddings-v2-base-en",
                 token_embed_fn: Optional[Callable[[str], List[List[float]]]] = None):
        if JinaEmbeddings is None:
            raise ImportError("jina package not found. `pip install jina`.")
        self.model = JinaEmbeddings(model_name=model_name)
        self._token_embed_fn = token_embed_fn

    def embed_text(self, text: str) -> List[float]:
        return self.model.embed(text)

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return [self.model.embed(d) for d in docs]

    def embed_tokens(self, text: str) -> Optional[List[List[float]]]:
        if self._token_embed_fn:
            return self._token_embed_fn(text)
        return None
