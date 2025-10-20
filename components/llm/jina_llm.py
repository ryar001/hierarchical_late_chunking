import os
from typing import Any, Dict, List, Optional

import requests
from components.llm.models_const import JinaModels, EMBEDDINGS, RERANK, READ, SEARCH, DEEPSEARCH, SEGMENT, CLASSIFY
from components.models.llm_model import (
    EmbedOutput, RerankOutput, ReadOutput, SearchOutput, DeepSearchOutput, SegmentOutput, ClassifyOutput
)

class JinaLLM:
    """
    Jina AI API wrapper for various services including embeddings, reranking, reading, etc.
    Get your Jina AI API key for free: https://jina.ai/?sui=apikey
    """

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or os.getenv("JINA_API_KEY")
        if not self.api_key:
            raise ValueError("JINA_API_KEY not provided as argument or environment variable.")

        self.base_headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }

        self.api_urls = JinaModels.llm_urls

    def _post_request(self, url: str, data: dict, custom_headers: Optional[Dict[str, str]] = None) -> dict:
        """Helper to make sync post requests to Jina API."""
        headers = self.base_headers.copy()
        if custom_headers:
            headers.update(custom_headers)

        try:
            with requests.post(url, headers=headers, json=data, stream=(headers.get("Accept") == "text/event-stream")) as response:
                response.raise_for_status()
                if headers.get("Accept") == "text/event-stream":
                    # Streaming response
                    result = []
                    for line in response.iter_lines():
                        if line:
                            result.append(line.decode('utf-8'))
                    return {"stream_data": result}
                return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Requests client error: {e}")
            return {}
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            return {}

    def embed(self, model_name: str, input: List[Any], **kwargs) -> EmbedOutput:
        """Generate embeddings for given inputs."""
        data = {"model": model_name, "input": input, **kwargs}
        result = self._post_request(self.api_urls[EMBEDDINGS], data)
        return EmbedOutput.model_validate(result)

    def rerank(self, model: JinaModels.RerankerModels, query: str, documents: List[Any], **kwargs) -> RerankOutput:
        """Rerank documents based on a query."""
        data = {"model": model.value, "query": query, "documents": documents, **kwargs}
        result = self._post_request(self.api_urls[RERANK], data)
        return RerankOutput.model_validate(result)

    def read(self, url: str, custom_headers: Optional[Dict[str, str]] = None, **kwargs) -> ReadOutput:
        """Retrieve and parse content from a URL."""
        data = {"url": url, **kwargs}
        result = self._post_request(self.api_urls[READ], data, custom_headers)
        return ReadOutput.model_validate(result)

    def search(self, query: str, custom_headers: Optional[Dict[str, str]] = None, **kwargs) -> SearchOutput:
        """Search the web for information."""
        data = {"q": query, **kwargs}
        result = self._post_request(self.api_urls[SEARCH], data, custom_headers)
        return SearchOutput.model_validate(result)

    def deepsearch(self, model: JinaModels.DeepsearchModels, messages: List[Dict[str, Any]], **kwargs) -> DeepSearchOutput:
        """Perform a comprehensive investigation on a topic."""
        data = {"model": model.value, "messages": messages, **kwargs}
        result = self._post_request(self.api_urls[DEEPSEARCH], data)
        return DeepSearchOutput.model_validate(result)

    def segment(self, content: str, **kwargs) -> SegmentOutput:
        """Segment text into chunks or tokens."""
        data = {"content": content, **kwargs}
        result = self._post_request(self.api_urls[SEGMENT], data)
        return SegmentOutput.model_validate(result)

    def classify(self, model: JinaModels.ClassifierModels, input: List[Any], labels: List[str], **kwargs) -> ClassifyOutput:
        """Classify text or images into categories."""
        data = {"model": model.value, "input": input, "labels": labels, **kwargs}
        result = self._post_request(self.api_urls[CLASSIFY], data)
        return ClassifyOutput.model_validate(result)
