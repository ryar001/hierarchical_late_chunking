from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid
import os

from docling.document_converter import DocumentConverter
from langgraph.graph import StateGraph, END

from components.embeddings_llm.llm_interface import LLMInterface
from components.embedding_interface import EmbeddingInterface
from components.vector_db_interface import VectorDbInterface
from components.data_structures import GraphState, RetrievalDoc
from components.utils import mean_pool, fuse_vectors, sliding_chunks, _which_section, _pack_results


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
        
        doc_id = doc_id or os.path.basename(file_path)

        if file_path.lower().endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                doc_text = f.read()
        else:
            # 1. Use DocumentConverter to handle any file type
            converter = DocumentConverter()
            doc = converter.convert(file_path).document

            # 2. Extract the full text content
            doc_text = doc.export_to_text()
        
        # 3. Pass the extracted text to the original ingestion method
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
        token_vectors = self.embedding_model.embed_tokens(tokens)

        # 2) Sectioning (coarse units)
        section_spans = sliding_chunks(tokens, self.default_section_tokens, overlap=128)
        sections: List[str] = [span["text"] for span in section_spans]

        # Section summaries (LLM)
        section_summaries: List[str] = [
            self.llm.summarize(text=sec, max_tokens=256).summary for sec in sections
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
                                                   where={"$and": [{"type": "chunk"}, {"section_id": sid}]})
            hits.extend(_pack_results(res))
        
        seen = set()
        uniq: List[RetrievalDoc] = []
        for h in hits:
            if h.id not in seen:
                uniq.append(h)
                seen.add(h.id)
        return uniq

    # -------------------
    # LangGraph Nodes
    # -------------------
    def _node_query_expansion(self, state: GraphState) -> GraphState:
        q = state["query"]
        expansions = self.llm.expand_query(q, max_suggestions=3).expanded_queries
        state["sub_queries"] = [q] + expansions
        return state

    def _node_section_retrieval(self, state: GraphState) -> GraphState:
        q = state["query"]
        state["section_hits"] = self._section_retrieval(q, top_n=3)
        return state

    def _node_chunk_retrieval(self, state: GraphState) -> GraphState:
        q = state["query"]
        sec_ids = [h.metadata.get("section_id", h.id) for h in state.get("section_hits", [])]
        sec_ids = [sid for sid in sec_ids if sid is not None]
        chunks = self._chunk_retrieval_from_sections(q, sec_ids, k_per_section=4)
        state["chunk_hits"] = chunks
        return state

    def _node_answer(self, state: GraphState) -> GraphState:
        q = state["query"]
        chunks = state.get("chunk_hits", [])
        top_chunks = chunks[:6]
        context = "\n\n".join([c.text for c in top_chunks])
        final = self.llm.answer(q, context).answer
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
