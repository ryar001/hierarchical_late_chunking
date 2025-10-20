from components.llm.models_const import Models
from typing import Callable, List, Optional
class BaseEmbeddingClass:
    """
    Jina text embedding wrapper using async API calls.
    - If token-level embeddings are available, implement embed_tokens.
    - Otherwise, embed_tokens returns None and the pipeline will use a global-fusion fallback.
    """

    def __init__(
        self,
        model_name: Models.EmbeddingModels = None,
        token_embed_fn: Optional[Callable[[str], List[List[float]]]] = None,
        api_key: Optional[str] = None,
        *args, **kwargs
        ):
        self._token_embed_fn = token_embed_fn

if __name__ == "__main__":
    print("BaseEmbeddingClass")
    br = BaseEmbeddingClass()