from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from components.vector_db_interface import VectorDbInterface

class ChromaDb(VectorDbInterface):
    def __init__(self, persist_directory: str = "./chroma_store"):
        self.client = chromadb.PersistentClient(path=persist_directory)

    def get_or_create(self, name: str) -> Collection:
        return self.client.get_or_create_collection(name=name)

    def add(self, collection: str, ids: List[str], documents: List[str],
            embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        coll = self.get_or_create(collection)
        coll.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query_by_embedding(self, collection: str, query_embedding: List[float], n_results: int,
                           where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        coll = self.get_or_create(collection)
        return coll.query(query_embeddings=[query_embedding], n_results=n_results, where=where)

    def query_by_text(self, collection: str, query_text: str, n_results: int,
                      where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        coll = self.get_or_create(collection)
        return coll.query(query_texts=[query_text], n_results=n_results, where=where)

    def delete(self, collection: str, ids: List[str]) -> None:
        coll = self.get_or_create(collection)
        coll.delete(ids=ids)
