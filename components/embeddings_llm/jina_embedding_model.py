from typing import Callable, List, Optional

from components.embedding_interface import EmbeddingInterface
from components.embeddings_llm.embedding_const import JinaModels
from components.llm.jina_llm import JinaLLM
from components.models.llm_model import EmbedOutput

class JinaEmbeddingModel(JinaLLM, EmbeddingInterface):
    """
    Jina text embedding wrapper using sync API calls.

    This class provides methods to embed text and documents using Jina AI's embedding models.
    It inherits from JinaLLM for API interaction and implements EmbeddingInterface.
    """

    def __init__(
        self,
        model_name: JinaModels.EmbeddingModels = JinaModels.EmbeddingModels.JINA_EMBEDDINGS_V4,
        token_embed_fn: Optional[Callable[[str], List[List[float]]]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the JinaEmbeddingModel.

        Args:
            model_name (JinaModels): The name of the Jina embedding model to use.
                                      Defaults to JinaModels.JINA_EMBEDDINGS_V4.
            token_embed_fn (Optional[Callable[[str], List[List[float]]]]): An optional
                                                                          function for token-level embeddings.
                                                                          Defaults to None.
            api_key (Optional[str]): Your Jina AI API key. If not provided, it will be
                                     fetched from the JINA_API_KEY environment variable.
        """
        super().__init__(api_key=api_key)
        self.model_name: JinaModels = model_name
        self._token_embed_fn: Optional[Callable[[str], List[List[float]]]] = token_embed_fn

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string of text.

        Args:
            text (str): The input text string to embed.

        Returns:
            List[float]: A list of floats representing the embedding of the text.
                         Returns an empty list if the input text is empty or embedding fails.
        """
        if not text:
            return []

        result: EmbedOutput = self.embed(model_name=self.model_name, input=[text])
        if result and result.data:
            return result.data[0].embedding
        return []

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.

        Args:
            docs (List[str]): A list of strings, where each string is a document to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
                               Returns an empty list if the input document list is empty or embedding fails.
        """
        if not docs:
            return []

        result: EmbedOutput = self.embed(model_name=self.model_name, input=docs)
        if result and result.data:
            return [item.embedding for item in result.data]
        return []

    def embed_tokens(self, text: str) -> Optional[List[List[float]]]:
        """
        Embeds tokens if a tokenization function is provided.

        Note: This implementation does not directly support token-level embeddings
        from the Jina API, as the API returns document-level embeddings.
        This method relies on the provided `_token_embed_fn` for tokenization
        and then would need to be adapted if token embeddings are required.

        Args:
            text (str): The input text string for which to generate token embeddings.

        Returns:
            Optional[List[List[float]]]: A list of token embeddings (each a list of floats)
                                         if `_token_embed_fn` is provided and successful,
                                         otherwise None.
        """
        if self._token_embed_fn:
            # This is a placeholder for where you would implement token-level embeddings.
            # The current Jina API doesn't directly support returning embeddings per token for a document.
            # You might need to call embed_text for each token, which is inefficient.
            return self._token_embed_fn(text)
        return None