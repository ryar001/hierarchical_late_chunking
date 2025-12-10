from enum import StrEnum
from abc import ABC, abstractmethod
from typing import Dict

EMBEDDINGS = "embeddings"
RERANK = "rerank"
READ = "read"
SEARCH = "search"
DEEPSEARCH = "deepsearch"
SEGMENT = "segment"
CLASSIFY = "classify"


class Models(ABC):
    llm_urls: Dict[str, str]
    @abstractmethod
    class EmbeddingModels(StrEnum): ...
    @abstractmethod
    class RerankerModels(StrEnum): ...
    @abstractmethod
    class DeepsearchModels(StrEnum): ...
    @abstractmethod
    class ClassifierModels(StrEnum): ...
    @abstractmethod
    class FastModels(StrEnum): ...

class GeminiModels(Models):
    class EmbeddingModels(StrEnum):
        GEMINI_EMBEDDING_001 = "gemini-embedding-001"

    class RerankerModels(StrEnum):
        GEMINI_EMBEDDING_001 = "gemini-embedding-001"

    class DeepsearchModels(StrEnum):
        GEMINI_3_PRO_PREVIEW = "gemini-3-pro-preview"
        GEMINI_2_DOT_5_PRO_PREVIEW = "gemini-2.5-pro-preview"
        GEMINI_2_DOT_5_FLASH = "gemini-2.5-flash"

    class FastModels(StrEnum):
        GEMINI_2_DOT_5_FLASH = "gemini-2.5-flash"
        GEMINI_2_DOT_5_FLASH_LITE = "gemini-2.5-flash-lite"

    class ClassifierModels(StrEnum):
        NONE = "none"


class JinaModels(Models):
    llm_urls = {
        EMBEDDINGS: "https://api.jina.ai/v1/embeddings",
        RERANK: "https://api.jina.ai/v1/rerank",
        READ: "https://r.jina.ai/",
        SEARCH: "https://s.jina.ai/",
        DEEPSEARCH: "https://deepsearch.jina.ai/v1/chat/completions",
        SEGMENT: "https://segment.jina.ai/",
        CLASSIFY: "https://api.jina.ai/v1/classify",
    }
    class EmbeddingModels(StrEnum):
        JINA_EMBEDDINGS_V2 = "jina-embeddings-v2"
        JINA_EMBEDDINGS_V3 = "jina-embeddings-v3"
        JINA_EMBEDDINGS_V4 = "jina-embeddings-v4"
        JINA_EMBEDDINGS_V2_BASE_EN = "jina-embeddings-v2-base-en"

    class RerankerModels(StrEnum):
        JINA_RERANKER_V1 = "jina-reranker-v1"
        JINA_RERANKER_V2_BASE_MULTILINGUAL = "jina-reranker-v2-base-multilingual"

    class DeepsearchModels(StrEnum):
        JINA_DEEPSEARCH_V1 = "jina-deepsearch-v1"

    class ClassifierModels(StrEnum):
        JINA_EMBEDDINGS_V3 = "jina-embeddings-v3"
        JINA_CLIP_V1 = "jina-clip-v1"
        JINA_CLIP_V2 = "jina-clip-v2"

if __name__ == "__main__":
    print(JinaModels.EmbeddingModels.JINA_EMBEDDINGS_V2.value)
    print(JinaModels.RerankerModels.JINA_RERANKER_V2_BASE_MULTILINGUAL.value)
    print(JinaModels.DeepsearchModels.JINA_DEEPSEARCH_V1.value)
    print(JinaModels.ClassifierModels.JINA_EMBEDDINGS_V3.value)
    print(JinaModels.ClassifierModels.JINA_CLIP_V1.value)
    print(JinaModels.ClassifierModels.JINA_CLIP_V2.value)