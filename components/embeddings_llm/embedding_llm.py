from typing import Callable, List, Optional
from components.embeddings_llm.jina_embedding_model import JinaEmbeddingModel
from components.embedding_interface import EmbeddingInterface

def get_embedding_model(model_name: str, token_embed_fn: Optional[Callable[[str], List[List[float]]]] = None) -> EmbeddingInterface:
    """
    Routes to the correct embedding model based on the model_name.

    Args:
        model_name: The name of the embedding model to use.
        token_embed_fn: An optional function for token-level embeddings.

    Returns:
        An instance of the specified EmbeddingInterface.

    Raises:
        ValueError: If an unsupported model_name is provided.
    """
    if model_name == "jina-embeddings-v2-base-en":
        return JinaEmbeddingModel(model_name=model_name, token_embed_fn=token_embed_fn)
    else:
        raise ValueError(f"Unsupported embedding model: {model_name}")
