import pytest
from unittest.mock import MagicMock, patch
import tempfile
import shutil
import os
from components.chroma_db import ChromaDb

@pytest.fixture
def temp_chroma_dir():
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir

@pytest.fixture
def mock_chroma_client():
    with patch('chromadb.PersistentClient') as mock_client:
        yield mock_client

@pytest.fixture
def mock_collection():
    return MagicMock()

@pytest.fixture
def chroma_db_instance(mock_chroma_client, temp_chroma_dir):
    # Configure the mock client to return a mock collection when get_or_create_collection is called
    mock_chroma_client.return_value.get_or_create_collection.return_value = MagicMock()
    return ChromaDb(persist_directory=temp_chroma_dir)

def test_chroma_db_init(mock_chroma_client, temp_chroma_dir):
    db = ChromaDb(persist_directory=temp_chroma_dir)
    mock_chroma_client.assert_called_once_with(path=temp_chroma_dir)
    assert db.client == mock_chroma_client.return_value

def test_get_or_create(chroma_db_instance, mock_chroma_client, mock_collection):
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
    collection_name = "test_collection"
    result = chroma_db_instance.get_or_create(collection_name)
    mock_chroma_client.return_value.get_or_create_collection.assert_called_once_with(name=collection_name)
    assert result == mock_collection

def test_add(chroma_db_instance, mock_chroma_client, mock_collection):
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
    collection_name = "test_collection"
    ids = ["id1", "id2"]
    documents = ["doc1", "doc2"]
    embeddings = [[1.0, 2.0], [3.0, 4.0]]
    metadatas = [{"source": "a"}, {"source": "b"}]

    chroma_db_instance.add(collection_name, ids, documents, embeddings, metadatas)

    mock_chroma_client.return_value.get_or_create_collection.assert_called_once_with(name=collection_name)
    mock_collection.add.assert_called_once_with(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

def test_query_by_embedding(chroma_db_instance, mock_chroma_client, mock_collection):
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
    collection_name = "test_collection"
    query_embedding = [1.0, 2.0]
    n_results = 5
    where_clause = {"source": "test"}
    expected_result = {"ids": [["id1"]], "documents": [["doc1"]]}

    mock_collection.query.return_value = expected_result

    result = chroma_db_instance.query_by_embedding(collection_name, query_embedding, n_results, where=where_clause)

    mock_chroma_client.return_value.get_or_create_collection.assert_called_once_with(name=collection_name)
    mock_collection.query.assert_called_once_with(query_embeddings=[query_embedding], n_results=n_results, where=where_clause)
    assert result == expected_result

def test_query_by_text(chroma_db_instance, mock_chroma_client, mock_collection):
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
    collection_name = "test_collection"
    query_text = "test query"
    n_results = 3
    where_clause = {"source": "another_test"}
    expected_result = {"ids": [["id3"]], "documents": [["doc3"]]}

    mock_collection.query.return_value = expected_result

    result = chroma_db_instance.query_by_text(collection_name, query_text, n_results, where=where_clause)

    mock_chroma_client.return_value.get_or_create_collection.assert_called_once_with(name=collection_name)
    mock_collection.query.assert_called_once_with(query_texts=[query_text], n_results=n_results, where=where_clause)
    assert result == expected_result

def test_delete(chroma_db_instance, mock_chroma_client, mock_collection):
    mock_chroma_client.return_value.get_or_create_collection.return_value = mock_collection
    collection_name = "test_collection"
    ids_to_delete = ["id1", "id2"]

    chroma_db_instance.delete(collection_name, ids_to_delete)

    mock_chroma_client.return_value.get_or_create_collection.assert_called_once_with(name=collection_name)
    mock_collection.delete.assert_called_once_with(ids=ids_to_delete)

