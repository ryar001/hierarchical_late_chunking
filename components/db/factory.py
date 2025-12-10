from components.db.vector_db_interface import VectorDbInterface
from components.db.chroma_db import ChromaDb

# To use Firestore, user would implement: from components.firestore_db import FirestoreDb
# For now, we provide the logic to switch.

class ComponentFactory:
    @staticmethod
    def get_vector_db(db_type: str = "chroma", **kwargs) -> VectorDbInterface:
        """
        Factory method to get the vector database instance.
        
        Args:
            db_type: 'chroma' or 'firestore' (future)
        """
        if db_type.lower() == "chroma":
            return ChromaDb(
                persist_directory=kwargs.get("persist_directory", "./chroma_store"),
                host=kwargs.get("host"),
                port=kwargs.get("port", 8000),
                ssl=kwargs.get("ssl", False),
                headers=kwargs.get("headers"),
                api_key=kwargs.get("api_key"),
                tenant=kwargs.get("tenant"),
                database=kwargs.get("database")
            )
        elif db_type.lower() == "firestore":
            # Placeholder for Firestore implementation
            # return FirestoreDb(**kwargs)
            raise NotImplementedError("Firestore implementation not yet loaded. Please implement components/firestore_db.py")
        else:
            raise ValueError(f"Unknown db_type: {db_type}")

    @staticmethod
    def create_pipeline(llm, emb, db_type: str = "chroma", **kwargs):
        """Helper to create the full hierarchy pipeline."""
        vdb = ComponentFactory.get_vector_db(db_type, **kwargs)
        from components.hierarchy_late_chunk import HierarchyLateChunk
        return HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)
