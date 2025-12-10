from typing import Any, Dict, List, Optional
import chromadb
from chromadb.api.models.Collection import Collection
from components.db.vector_db_interface import VectorDbInterface
from components.data_structures import RetrievalDoc

class ChromaDb(VectorDbInterface):
    def __init__(self, persist_directory: str = "./chroma_store", 
                 host: Optional[str] = None, 
                 port: int = 8000, 
                 ssl: bool = False, 
                 headers: Optional[Dict[str, str]] = None,
                 api_key: Optional[str] = None,
                 tenant: Optional[str] = None,
                 database: Optional[str] = None):
        if api_key and tenant and database:
            self.client = chromadb.CloudClient(
                api_key=api_key,
                tenant=tenant,
                database=database
            )
        elif host:
            if not host.startswith("http://") and not host.startswith("https://"):
                protocol = "https://" if ssl else "http://"
                host = f"{protocol}{host}"
            self.client = chromadb.HttpClient(host=host, port=port, ssl=ssl, headers=headers)
        else:
            self.client = chromadb.PersistentClient(path=persist_directory)
   

    def get_or_create(self, name: str) -> Collection:
        return self.client.get_or_create_collection(name=name)

    def add(self, collection: str, ids: List[str], documents: List[str],
            embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None:
        coll = self.get_or_create(collection)
        coll.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    def query_by_embedding(self, collection: str, query_embedding: List[float], n_results: int,
                           where: Optional[Dict[str, Any]] = None) -> List[RetrievalDoc]:
        coll = self.get_or_create(collection)
        res = coll.query(query_embeddings=[query_embedding], n_results=n_results, where=where)
        return self._pack_results(res)

    def query_by_text(self, collection: str, query_text: str, n_results: int,
                      where: Optional[Dict[str, Any]] = None) -> List[RetrievalDoc]:
        coll = self.get_or_create(collection)
        res = coll.query(query_texts=[query_text], n_results=n_results, where=where)
        return self._pack_results(res)

    def _pack_results(self, results: Dict[str, Any]) -> List[RetrievalDoc]:
        """Convert ChromaDB results to standardized RetrievalDoc objects."""
        from components.data_structures import RetrievalDoc
        ids = results.get("ids", [[]])
        docs = results.get("documents", [[]])
        metas = results.get("metadatas", [[]])
        embs = results.get("embeddings", [[]])
        out: List[RetrievalDoc] = []
        if not ids or not ids[0]:
            return out
        for i in range(len(ids[0])):
            metadata = metas[0][i] if metas and metas[0] and len(metas[0]) > i else {}
            # distances = results.get("distances", [[]])
            # score = distances[0][i] if distances and distances[0] else 0.0
            embedding = embs[0][i] if embs and embs[0] and len(embs[0]) > i else None
            out.append(RetrievalDoc(id=ids[0][i], text=docs[0][i], metadata=metadata, embedding=embedding))
        return out

    def delete(self, collection: str, ids: List[str]) -> None:
        coll = self.get_or_create(collection)
        coll.delete(ids=ids)

    def delete_collection(self, name: str) -> None:
        """Deletes a collection from the database."""
        self.client.delete_collection(name=name)
