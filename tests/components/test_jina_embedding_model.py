import pytest
import os
import dotenv
# add the project root to the path
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from components.embeddings_llm.jina_embedding_model import JinaEmbeddingModel
from components.embeddings_llm.embedding_const import JinaModels

dotenv.load_dotenv()

@pytest.mark.asyncio
async def test_jina_embedding_model_live():
    """
    Live test for JinaEmbeddingModel using a JINA_API_KEY from environment variables.
    This test embeds a small text snippet to verify API connectivity and functionality.
    """
    if "JINA_API_KEY" not in os.environ:
        pytest.skip("JINA_API_KEY not set in environment variables. Skipping live test.")

    # Placeholder for text content from lamrim.pdf
    # Please provide a small text snippet from lamrim.pdf for this test.
    text_to_embed = """
    The path to enlightenment is a gradual process, not a sudden leap. It requires diligent study,
    contemplation, and meditation on the teachings.
    """

    model = JinaEmbeddingModel(model_name=JinaModels.EmbeddingModels.JINA_EMBEDDINGS_V4,
                               api_key=os.environ["JINA_API_KEY"])
    embedding = await model.embed_text(text_to_embed)

    assert isinstance(embedding, list)
    assert len(embedding) > 0
    # Jina embeddings v4 has a dimension of 2048 for v4
    assert len(embedding) == 2048

@pytest.mark.asyncio
async def test_jina_embedding_model_embed_documents_live():
    """
    Live test for JinaEmbeddingModel's embed_documents method.
    """
    if "JINA_API_KEY" not in os.environ:
        pytest.skip("JINA_API_KEY not set in environment variables. Skipping live test.")

    docs_to_embed = [
        "The path to enlightenment is a gradual process.",
        "It requires diligent study, contemplation, and meditation on the teachings."
    ]

    model = JinaEmbeddingModel(model_name=JinaModels.EmbeddingModels.JINA_EMBEDDINGS_V4)
    embeddings = await model.embed_documents(docs_to_embed)

    assert isinstance(embeddings, list)
    assert len(embeddings) == len(docs_to_embed)
    for embedding in embeddings:
        assert isinstance(embedding, list)
        assert len(embedding) == 768
