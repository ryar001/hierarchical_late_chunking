
import os
from typing import Callable, List, Optional
import google.generativeai as genai
from components.embedding_interface import EmbeddingInterface
from components.llm.models_const import GeminiModels

class GeminiEmbeddingModel(EmbeddingInterface):
    """
    Gemini embedding wrapper.
    """

    def __init__(
        self,
        model_name: str = GeminiModels.EmbeddingModels.GEMINI_EMBEDDING_001,
        token_embed_fn: Optional[Callable[[str], List[List[float]]]] = None,
        api_key: Optional[str] = None,
    ) -> None:
        """
        Initializes the GeminiEmbeddingModel.

        Args:
            model_name (str): The name of the Gemini embedding model to use.
            token_embed_fn (Optional[Callable[[str], List[List[float]]]]): An optional
                                                                          function for token-level embeddings.
            api_key (Optional[str]): Your Google API key. If not provided, it will be
                                     fetched from the GOOGLE_API_KEY environment variable.
        """
        self.model_name = model_name
        self._token_embed_fn: Optional[Callable[[str], List[List[float]]]] = token_embed_fn
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        
        if self.api_key:
            self.api_key = self.api_key.strip().strip("'").strip('"')

        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY not provided as argument or environment variable.")

        genai.configure(api_key=self.api_key)

    def embed_text(self, text: str) -> List[float]:
        """
        Embeds a single string of text.

        Args:
            text (str): The input text string to embed.

        Returns:
            List[float]: A list of floats representing the embedding of the text.
        """
        if not text:
            return []

        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type="retrieval_document"
            )
            # The result is a dict with 'embedding' key which is a list of floats
            if 'embedding' in result:
                return result['embedding']
            return []
        except Exception as e:
            print(f"Error embedding text with Gemini: {e}")
            return []

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        """
        Embeds a list of documents.

        Args:
            docs (List[str]): A list of strings, where each string is a document to embed.

        Returns:
            List[List[float]]: A list of embeddings, where each embedding is a list of floats.
        """
        if not docs:
            return []

        try:
            result = genai.embed_content(
                model=self.model_name,
                content=docs,
                task_type="retrieval_document"
            )
            # When content is a list, result['embedding'] is a list of embeddings
            if 'embedding' in result:
                return result['embedding']
            return []
        except Exception as e:
            print(f"Error embedding documents with Gemini: {e}")
            return []

    def embed_tokens(self, text: str) -> Optional[List[List[float]]]:
        """
        Embeds tokens if a tokenization function is provided.
        """
        if self._token_embed_fn:
            return self._token_embed_fn(text)
        return None
