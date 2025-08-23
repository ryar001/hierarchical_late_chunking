from __future__ import annotations
from typing import List, Dict, Any, Optional, Protocol, TypedDict, Callable
from dataclasses import dataclass
import uuid
import math
import os

# Docling for file loading (PyPI: docling-core)
from docling import DoclingLoader

# LangGraph
from langgraph.graph import StateGraph, END

# Vector DB (PyPI: chromadb)
import chromadb
from chromadb.api.models.Collection import Collection


# =========================
# Protocols (Dependency Injection)
# =========================

class LLMInterface(Protocol):
    """LLM contract for DI."""
    def generate(self, prompt: str) -> str: ...
    def summarize(self, text: str, max_tokens: int = 256) -> str: ...
    def expand_query(self, query: str, max_suggestions: int = 3) -> List[str]: ...
    def answer(self, question: str, context: str) -> str: ...


class EmbeddingInterface(Protocol):
    """
    Embedding contract.
    Late chunking needs token-level vectors; if unavailable, implement a fallback.
    """
    def embed_text(self, text: str) -> List[float]: ...
    def embed_documents(self, docs: List[str]) -> List[List[float]]: ...
    def embed_tokens(self, text: str) -> Optional[List[List[float]]]: ...


class VectorDbInterface(Protocol):
    """Vector DB contract."""
    def get_or_create(self, name: str) -> Collection: ...
    def add(self, collection: str, ids: List[str], documents: List[str],
            embeddings: List[List[float]], metadatas: List[Dict[str, Any]]) -> None: ...
    def query_by_embedding(self, collection: str, query_embedding: List[float], n_results: int,
                           where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...
    def query_by_text(self, collection: str, query_text: str, n_results: int,
                      where: Optional[Dict[str, Any]] = None) -> Dict[str, Any]: ...


# =========================
# Embeddings (Jina adapter)
# =========================

# If you have jina installed, import your client. We keep this minimal to avoid version pinning issues.
try:
    from jina import JinaEmbeddings  # pip install jina
except Exception:  # pragma: no cover
    JinaEmbeddings = None  # type: ignore


class JinaEmbeddingModel(EmbeddingInterface):
    """
    Jina text embedding wrapper.
    - If token-level embeddings are available in your Jina client, implement embed_tokens.
    - Otherwise, embed_tokens returns None and the pipeline will use a global-fusion fallback.
    """
    def __init__(self, model_name: str = "jina-embeddings-v2-base-en",
                 token_embed_fn: Optional[Callable[[str], List[List[float]]]] = None):
        if JinaEmbeddings is None:
            raise ImportError("jina package not found. `pip install jina`.")
        # JinaEmbeddings API shape may vary by version. Adjust as needed.
        self.model = JinaEmbeddings(model_name=model_name)
        self._token_embed_fn = token_embed_fn  # optional: supply a token-level function

    def embed_text(self, text: str) -> List[float]:
        # Many clients expose `.embed(text)` that returns a single vector
        return self.model.embed(text)  # type: ignore[attr-defined]

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        # Some clients provide batch API; else loop
        return [self.model.embed(d) for d in docs]  # type: ignore[attr-defined]

    def embed_tokens(self, text: str) -> Optional[List[List[float]]]:
        # If you can expose last-hidden-states (token vectors), pass a function in token_embed_fn
        if self._token_embed_fn:
            return self._token_embed_fn(text)
        return None  # Fallback handled by pipeline


# =========================
# Chroma (PyPI) wrapper
# =========================

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
        # Note: this uses server-side embedding if configured; we generally prefer query_by_embedding
        coll = self.get_or_create(collection)
        return coll.query(query_texts=[query_text], n_results=n_results, where=where)


# =========================
# Data structures
# =========================

class RetrievalDoc(TypedDict):
    id: str
    text: str
    metadata: Dict[str, Any]


class GraphState(TypedDict, total=False):
    query: str
    sub_queries: List[str]
    section_hits: List[RetrievalDoc]
    chunk_hits: List[RetrievalDoc]
    final_answer: str


# =========================
# Utility functions
# =========================

def mean_pool(vectors: List[List[float]]) -> List[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    n = len(vectors)
    return [sum(vec[i] for vec in vectors) / n for i in range(dim)]


def fuse_vectors(primary: List[float], global_vec: List[float], alpha: float = 0.8) -> List[float]:
    """
    Late-chunking fallback fusion: chunk_vec' = alpha*chunk_vec + (1-alpha)*global_doc_vec
    """
    if not primary:
        return global_vec
    if not global_vec:
        return primary
    d = min(len(primary), len(global_vec))
    return [alpha * primary[i] + (1.0 - alpha) * global_vec[i] for i in range(d)]


def sliding_chunks(tokens: List[str], chunk_size: int, overlap: int) -> List[Dict[str, Any]]:
    spans: List[Dict[str, Any]] = []
    i = 0
    step = max(1, chunk_size - overlap)
    while i < len(tokens):
        j = min(len(tokens), i + chunk_size)
        spans.append({"start": i, "end": j, "text": " ".join(tokens[i:j])})
        if j == len(tokens):
            break
        i += step
    return spans


# =========================
# Main Pipeline
# =========================

@dataclass
class HierarchyLateChunk:
    llm: LLMInterface
    embedding_model: EmbeddingInterface
    vectordb: VectorDbInterface
    sections_collection: str = "rag_sections"
    chunks_collection: str = "rag_chunks"
    default_chunk_size: int = 480
    default_overlap: int = 64
    default_section_tokens: int = 2000

    # -------------------
    # Ingestion
    # -------------------
    def ingest_from_file(self, file_path: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Loads a document from a file path (PDF, DOCX, TXT, etc.) using docling,
        extracts the text, and then processes it using the core ingestion logic.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found at: {file_path}")
            
        print(f"--- Loading document from: {file_path} ---")
        # 1. Use DoclingLoader to handle any file type
        loader = DoclingLoader.from_file(file_path)
        doc = loader.load()

        # 2. Extract the full text content
        doc_text = doc.get_text()
        
        # 3. Pass the extracted text to the original ingestion method
        #    Use the filename as a default doc_id if not provided
        doc_id = doc_id or os.path.basename(file_path)
        return self.ingest_document(doc_text, doc_id=doc_id)

    def ingest_document(self, doc_text: str, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Core ingestion logic for late-chunking.
        - Processes a raw string of text.
        - Try token-level embeddings -> pool into chunk vectors.
        - Else fallback: global doc vector + per-chunk vectors -> fused chunk vectors.
        - Additionally create section summaries + section vectors.
        """
        print(f"--- Processing text for document ID: {doc_id} ---")
        doc_id = doc_id or str(uuid.uuid4())
        tokens = doc_text.split()

        # 1) Attempt true late chunking (token vectors)
        token_vectors = self.embedding_model.embed_tokens(doc_text)

        # 2) Sectioning (coarse units)
        section_spans = sliding_chunks(tokens, self.default_section_tokens, overlap=128)
        sections: List[str] = [span["text"] for span in section_spans]

        # Section summaries (LLM)
        section_summaries: List[str] = [
            self.llm.summarize(text=sec, max_tokens=256) for sec in sections
        ]

        # Section embeddings (use summary text for compactness)
        section_embs: List[List[float]] = self.embedding_model.embed_documents(section_summaries)

        # Store sections
        section_ids = [f"{doc_id}_sec_{i}" for i in range(len(sections))]
        section_metas = [
            {"type": "section", "doc_id": doc_id, "section_index": i} for i in range(len(sections))
        ]
        self.vectordb.add(self.sections_collection, section_ids, section_summaries, section_embs, section_metas)

        # 3) Chunking (fine units)
        chunk_spans = sliding_chunks(tokens, self.default_chunk_size, self.default_overlap)
        chunk_texts: List[str] = [span["text"] for span in chunk_spans]

        # Build chunk embeddings
        if token_vectors is not None:
            # True late chunking: pool token vectors aligned with chunk spans
            chunk_vecs: List[List[float]] = []
            for span in chunk_spans:
                start, end = span["start"], span["end"]
                pooled = mean_pool(token_vectors[start:end]) if end > start else []
                chunk_vecs.append(pooled)
        else:
            # Fallback: global-fusion
            print("Token-level embeddings not available. Using global-fusion fallback.")
            global_vec = self.embedding_model.embed_text(doc_text)
            raw_chunk_vecs = self.embedding_model.embed_documents(chunk_texts)
            chunk_vecs = [fuse_vectors(rv, global_vec, alpha=0.8) for rv in raw_chunk_vecs]

        # Store chunks
        chunk_ids = [f"{doc_id}_ch_{i}" for i in range(len(chunk_texts))]
        chunk_metas: List[Dict[str, Any]] = []
        for i, span in enumerate(chunk_spans):
            sec_idx = _which_section(span["start"], section_spans)
            chunk_metas.append({
                "type": "chunk",
                "doc_id": doc_id,
                "chunk_index": i,
                "section_id": section_ids[sec_idx] if sec_idx is not None else None
            })

        self.vectordb.add(self.chunks_collection, chunk_ids, chunk_texts, chunk_vecs, chunk_metas)

        return {
            "doc_id": doc_id,
            "num_sections": len(sections),
            "num_chunks": len(chunk_texts),
        }

    # -------------------
    # Retrieval helpers
    # -------------------
    def _section_retrieval(self, query: str, top_n: int = 3) -> List[RetrievalDoc]:
        q_emb = self.embedding_model.embed_text(query)
        res = self.vectordb.query_by_embedding(self.sections_collection, q_emb, n_results=top_n,
                                               where={"type": "section"})
        return _pack_results(res)

    def _chunk_retrieval_from_sections(self, query: str, section_ids: List[str], k_per_section: int = 4) -> List[RetrievalDoc]:
        q_emb = self.embedding_model.embed_text(query)
        hits: List[RetrievalDoc] = []
        for sid in section_ids:
            res = self.vectordb.query_by_embedding(self.chunks_collection, q_emb, n_results=k_per_section,
                                                   where={"type": "chunk", "section_id": sid})
            hits.extend(_pack_results(res))
        
        seen = set()
        uniq: List[RetrievalDoc] = []
        for h in hits:
            if h["id"] not in seen:
                uniq.append(h)
                seen.add(h["id"])
        return uniq

    # -------------------
    # LangGraph Nodes
    # -------------------
    def _node_query_expansion(self, state: GraphState) -> GraphState:
        q = state["query"]
        expansions = self.llm.expand_query(q, max_suggestions=3)
        state["sub_queries"] = [q] + expansions
        return state

    def _node_section_retrieval(self, state: GraphState) -> GraphState:
        q = state["query"]
        state["section_hits"] = self._section_retrieval(q, top_n=3)
        return state

    def _node_chunk_retrieval(self, state: GraphState) -> GraphState:
        q = state["query"]
        sec_ids = [h["metadata"].get("section_id", h["id"]) for h in state.get("section_hits", [])]
        sec_ids = [sid for sid in sec_ids if sid is not None]
        chunks = self._chunk_retrieval_from_sections(q, sec_ids, k_per_section=4)
        state["chunk_hits"] = chunks
        return state

    def _node_answer(self, state: GraphState) -> GraphState:
        q = state["query"]
        chunks = state.get("chunk_hits", [])
        top_chunks = chunks[:6]
        context = "\n\n".join([c["text"] for c in top_chunks])
        final = self.llm.answer(q, context)
        state["final_answer"] = final
        return state

    # -------------------
    # Public: compile graph and run
    # -------------------
    def build_graph(self) -> Any:
        graph = StateGraph(GraphState)
        graph.add_node("expand", self._node_query_expansion)
        graph.add_node("sec_retrieve", self._node_section_retrieval)
        graph.add_node("chunk_retrieve", self._node_chunk_retrieval)
        graph.add_node("answer", self._node_answer)

        graph.set_entry_point("expand")
        graph.add_edge("expand", "sec_retrieve")
        graph.add_edge("sec_retrieve", "chunk_retrieve")
        graph.add_edge("chunk_retrieve", "answer")
        graph.add_edge("answer", END)
        return graph.compile()

    def run(self, query: str) -> str:
        app = self.build_graph()
        out: GraphState = app.invoke({"query": query})
        return out.get("final_answer", "")


# =========================
# Helpers
# =========================

def _which_section(start_token: int, section_spans: List[Dict[str, Any]]) -> Optional[int]:
    for i, sec in enumerate(section_spans):
        if sec["start"] <= start_token < sec["end"]:
            return i
    return None


def _pack_results(results: Dict[str, Any]) -> List[RetrievalDoc]:
    ids = results.get("ids", [[]])
    docs = results.get("documents", [[]])
    metas = results.get("metadatas", [[]])
    out: List[RetrievalDoc] = []
    if not ids or not ids[0]:
        return out
    for i in range(len(ids[0])):
        out.append({"id": ids[0][i], "text": docs[0][i], "metadata": metas[0][i]})
    return out


# =========================
# Example LLM for wiring
# =========================

class DummyLLM(LLMInterface):
    def generate(self, prompt: str) -> str:
        return f"[GEN]: {prompt[:200]}..."

    def summarize(self, text: str, max_tokens: int = 256) -> str:
        return (text[:max_tokens] + "...") if len(text) > max_tokens else text

    def expand_query(self, query: str, max_suggestions: int = 3) -> List[str]:
        base = [
            f"Key definitions related to: {query}",
            f"Worked example for: {query}",
            f"Edge cases / exceptions for: {query}",
        ]
        return base[:max_suggestions]

    def answer(self, question: str, context: str) -> str:
        return f"Q: {question}\n\nA (using {min(200, len(context))} chars of context):\n{context[:200]}..."


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Dependencies (inject)
    # NOTE: You will need to install jina to run this: `pip install jina`
    try:
        emb = JinaEmbeddingModel(model_name="jina-embeddings-v2-base-en")
    except ImportError as e:
        print(f"Error: {e}")
        print("Please install the required packages: `pip install jina chromadb langgraph docling-core`")
        exit()
        
    vdb = ChromaDb(persist_directory="./chroma_store")
    llm = DummyLLM()

    pipeline = HierarchyLateChunk(llm=llm, embedding_model=emb, vectordb=vdb)

    # --- NEW: Ingest from a file ---
    
    # 1. Create a dummy file to test the ingestion
    dummy_file_path = "physics_chapter.txt"
    print(f"Creating a dummy file for testing: {dummy_file_path}")
    with open(dummy_file_path, "w", encoding="utf-8") as f:
        f.write(
            "Chapter 4: Dynamics and Circular Motion\n"
            "Newton’s second law states that force equals mass times acceleration (F=ma). "
            "In uniform circular motion, acceleration is v^2/r toward the center. Therefore, the net force "
            "required is mv^2/r. This follows from combining the kinematics of circular motion with Newton’s laws. "
            "Examples include satellites orbiting Earth, cars taking turns, and pendulums at small angles. "
            * 40  # repeat to make it long enough for multiple sections
        )

    # 2. Ingest the document directly from the file path
    # This single call now handles loading, text extraction, and processing.
    # It would work the same for a PDF: pipeline.ingest_from_file("my_report.pdf")
    info = pipeline.ingest_from_file(dummy_file_path)
    print("\nIngestion complete:", info)

    # 3. Ask a question
    q = "How does Newton’s second law apply to circular motion?"
    print("\n--- Running Query ---")
    print(f"Question: {q}")
    answer = pipeline.run(q)
    print("\nFinal Answer:\n", answer)


