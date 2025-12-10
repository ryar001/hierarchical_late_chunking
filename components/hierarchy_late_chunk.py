from __future__ import annotations
from typing import Any, Dict, List, Optional
from dataclasses import dataclass
import uuid
import os
import concurrent.futures
from langgraph.graph import StateGraph, END



from components.embeddings_llm.llm_interface import LLMInterface
from components.embedding_interface import EmbeddingInterface
from components.db.vector_db_interface import VectorDbInterface
from components.data_structures import GraphState, RetrievalDoc
from components.utils import mean_pool, fuse_vectors, sliding_chunks, _which_section


@dataclass
class HierarchyLateChunk:
    llm: LLMInterface
    embedding_model: EmbeddingInterface
    vectordb: VectorDbInterface
    sections_collection: str = "rag_sections_v3"
    chunks_collection: str = "rag_chunks_v3"
    feedback_collection: str = "rag_feedback"
    default_chunk_size: int = 480
    default_overlap: int = 64
    default_section_tokens: int = 2000
    status_callback: Optional[callable] = None

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
            return self.ingest_document(doc_text, doc_id=doc_id)
        else:
            # 1. Use DocumentConverter to handle any file type
            from docling.document_converter import DocumentConverter
            converter = DocumentConverter()
            doc = converter.convert(file_path).document
            
            # Pass the DoclingDocument object directly to preserve metadata
            return self.ingest_document(doc, doc_id=doc_id)

    def ingest_document(self, doc_input: Any, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Core ingestion logic.
        - doc_input: Can be a raw string (str) or a DoclingDocument object.
        """
        print(f"--- Processing document ID: {doc_id} ---")
        doc_id = doc_id or str(uuid.uuid4())
        
        # Extract text and metadata
        if isinstance(doc_input, str):
            full_text = doc_input
            # No page info available for raw string
            text_elements = [{"text": t, "page": None} for t in full_text.split('\n\n') if t.strip()]
        else:
            # Assume DoclingDocument
            full_text = doc_input.export_to_text()
            text_elements = []
            # Iterate over structural elements to get page numbers
            for item, level in doc_input.iterate_items():
                if hasattr(item, "text") and item.text.strip():
                    page_no = None
                    if hasattr(item, "prov") and item.prov:
                        # prov is a list of provenance items, usually we take the first one's page_no
                        page_no = item.prov[0].page_no
                    text_elements.append({"text": item.text, "page": page_no})

        # 1) Attempt true late chunking (token vectors) on full text
        tokens = full_text.split()

        token_vectors = self.embedding_model.embed_tokens(tokens)

        # 2) Sectioning (coarse units)
        # We still use sliding window on full text for sections to ensure continuity
        section_spans = sliding_chunks(tokens, self.default_section_tokens, overlap=128)
        sections: List[str] = [span["text"] for span in section_spans]

        # Section summaries (LLM) - Parallelized
        print(f"Summarizing {len(sections)} sections in parallel...")
        def _summarize_section(text):
            try:
                return self.llm.summarize(text=text, max_tokens=256).summary
            except Exception as e:
                print(f"Error summarizing section: {e}")
                return ""

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            section_summaries = list(executor.map(_summarize_section, sections))

        # Section embeddings
        section_embs: List[List[float]] = self.embedding_model.embed_documents(section_summaries)

        # Store sections
        section_ids = [f"{doc_id}_sec_{i}" for i in range(len(sections))]
        section_metas = [
            {"type": "section", "doc_id": doc_id, "section_index": i} for i in range(len(sections))
        ]
        self.vectordb.add(self.sections_collection, section_ids, section_summaries, section_embs, section_metas)

        # 3) Chunking (fine units) with Page Numbers
        # We will use the text_elements (which have page info) to build chunks.
        # However, sliding_chunks works on tokens.
        # To map tokens back to pages is complex.
        # Alternative: We chunk the text_elements directly or map spans.
        
        # Simplified approach: Use sliding window on tokens, but find which text_element (and thus page) 
        # the chunk mostly belongs to.
        
        chunk_spans = sliding_chunks(tokens, self.default_chunk_size, self.default_overlap)
        chunk_texts: List[str] = []
        chunk_pages: List[Optional[int]] = []
        
        # Create a mapping of token index -> page number
        # This is an approximation since we split by whitespace above
        token_page_map = []
        for elem in text_elements:
            elem_tokens = elem["text"].split()
            count = len(elem_tokens)
            page = elem["page"]
            token_page_map.extend([page] * count)
            # Add newline/spacing tokens if needed, but split() eats them.
            # This map might drift if split() behaves differently than docling's internal tokenization.
            # But it's a reasonable approximation for "where is this text".
            
        for span in chunk_spans:
            start = span["start"]
            end = span["end"]
            
            # Find the most frequent page in this span
            if start < len(token_page_map):
                # Safe slice
                slice_end = min(end, len(token_page_map))
                pages_in_span = token_page_map[start:slice_end]
                # Filter None
                valid_pages = [p for p in pages_in_span if p is not None]
                if valid_pages:
                    # Most common page
                    most_common_page = max(set(valid_pages), key=valid_pages.count)
                    chunk_pages.append(most_common_page)
                else:
                    chunk_pages.append(None)
            else:
                chunk_pages.append(None)

            # Contextual Chunking
            sec_idx = _which_section(start, section_spans)
            context = section_summaries[sec_idx] if sec_idx is not None else ""
            full_text = f"Context: {context}\n\nChunk: {span['text']}"
            chunk_texts.append(full_text)

        # Build chunk embeddings
        if token_vectors is not None:
            chunk_vecs: List[List[float]] = []
            for span in chunk_spans:
                start, end = span["start"], span["end"]
                pooled = mean_pool(token_vectors[start:end]) if end > start else []
                chunk_vecs.append(pooled)
        else:
            print("Token-level embeddings not available. Using global-fusion fallback.")
            global_vec = self.embedding_model.embed_text(full_text)
            raw_chunk_vecs = self.embedding_model.embed_documents(chunk_texts)
            chunk_vecs = [fuse_vectors(rv, global_vec, alpha=0.8) for rv in raw_chunk_vecs]

        # Store chunks
        chunk_ids = [f"{doc_id}_ch_{i}" for i in range(len(chunk_texts))]
        chunk_metas: List[Dict[str, Any]] = []
        for i, span in enumerate(chunk_spans):
            sec_idx = _which_section(span["start"], section_spans)
            # Clean metadata: remove None values as ChromaDB might not support them
            meta = {
                "type": "chunk",
                "doc_id": doc_id,
                "chunk_index": i,
                "section_id": section_ids[sec_idx] if sec_idx is not None else "", # Use empty string for None
                "page": chunk_pages[i] if chunk_pages[i] is not None else -1 # Use -1 for None
            }
            # Remove keys that might still be None if strict cleaning is needed, 
            # though we handled specific ones above.
            chunk_metas.append({k: v for k, v in meta.items() if v is not None})

        self.vectordb.add(self.chunks_collection, chunk_ids, chunk_texts, chunk_vecs, chunk_metas)

        return {
            "doc_id": doc_id,
            "num_sections": len(sections),
            "num_chunks": len(chunk_texts),
        }

    def list_documents(self) -> List[str]:
        """
        Lists all unique document IDs currently stored in the vector database.
        """
        try:
            coll = self.vectordb.get_or_create(self.sections_collection)
            # Depending on Chroma client version, get might return different structures, 
            # but usually dict with keys.
            # We fetch all (implied by no ids arg)
            results = coll.get(include=["metadatas"])
            metas = results.get("metadatas", [])
            
            unique_ids = set()
            for m in metas:
                if m and "doc_id" in m:
                    unique_ids.add(m["doc_id"])
            
            return list(unique_ids)
        except Exception as e:
            print(f"Error listing documents: {e}")
            return []

    # -------------------
    # Feedback / Memory
    # -------------------
    def submit_feedback(self, query: str, chunk_ids: List[str], score: int, comment: str = "") -> None:
        """
        Saves a successful interaction to the feedback store.
        """
        if score < 4:
            return # Only save positive feedback
            
        feedback_id = str(uuid.uuid4())
        q_emb = self.embedding_model.embed_text(query)
        
        # We store the query vector, and in metadata we store the list of chunk_ids that worked.
        # Chroma metadata must be primitives, so we join chunk_ids with a separator.
        meta = {
            "chunk_ids": ",".join(chunk_ids),
            "score": score,
            "comment": comment,
            "original_query": query
        }
        
        self.vectordb.add(
            collection=self.feedback_collection,
            ids=[feedback_id],
            documents=[query], # We store the query text as the document
            embeddings=[q_emb],
            metadatas=[meta]
        )
        print(f"Feedback saved for query: '{query}'")

    def _retrieve_feedback_chunks(self, query: str, threshold: float = 0.85) -> List[RetrievalDoc]:
        """
        Checks if a similar query has been answered successfully before.
        Returns the specific chunks that were useful last time.
        """
        q_emb = self.embedding_model.embed_text(query)
        hits = self.vectordb.query_by_embedding(self.feedback_collection, q_emb, n_results=1)
        
        if not hits:
            return []
            
        # Check similarity (distance). Chroma returns distance (lower is better).
        # Assuming cosine distance: 0 is identical, 1 is opposite.
        # We'll use a rough heuristic or just trust the top result if it's close enough.
        # Note: _pack_results doesn't currently return distance/score easily without modification.
        # For now, we'll assume if it returns a hit, it's relevant, but in production check distance.
        
        hit = hits[0]
        # Parse chunk_ids
        chunk_ids_str = hit.metadata.get("chunk_ids", "")
        if not chunk_ids_str:
            return []
            
        chunk_ids = chunk_ids_str.split(",")
        
        # Now fetch the actual chunk documents from the chunks collection
        # ChromaDb wrapper doesn't have a 'get_by_ids' method exposed in the interface easily,
        # but we can query by IDs using the underlying client or a specific where clause.
        # For simplicity, we will use query_by_text with a filter, but that's inefficient.
        # Let's add a simple 'get' method to VectorDbInterface or use the client directly if possible.
        # Since we can't easily change the interface right now, we'll iterate or use a large $or query.
        
        # Efficient way: use the underlying client if accessible, or add a 'get' method.
        # Let's assume we can use `vectordb.client.get_collection(...).get(...)`
        # But `vectordb` is an interface. We'll cast it or add a method.
        # For now, let's try to use `query_by_embedding` with a filter for IDs.
        
        # Hack: Query with dummy embedding and ID filter.
        # Actually, let's just use the `get` method of the collection object directly if we can access it.
        # The `ChromaDb` class has `get_or_create`.
        
        try:
            coll = self.vectordb.get_or_create(self.chunks_collection)
            # Chroma collection.get(ids=...) returns a dict
            data = coll.get(ids=chunk_ids, include=["documents", "metadatas", "embeddings"])
            
            # Convert back to RetrievalDoc
            docs = []
            for i, doc_id in enumerate(data["ids"]):
                docs.append(RetrievalDoc(
                    id=doc_id,
                    text=data["documents"][i],
                    metadata=data["metadatas"][i] if data["metadatas"] else {},
                    score=1.0 # Artificially high score for feedback chunks
                ))
            return docs
        except Exception as e:
            print(f"Error retrieving feedback chunks: {e}")
            return []

    # -------------------
    # Retrieval helpers
    # -------------------
    def _mix_embeddings(self, q_vec: List[float], h_vec: Optional[List[float]] = None, exp_vecs: Optional[List[List[float]]] = None) -> List[float]:
        """
        Mixes query, hypothetical answer, and expanded query embeddings.
        Strategy: Average all available vectors.
        """
        vectors = [q_vec]
        if h_vec:
            vectors.append(h_vec)
        if exp_vecs:
            vectors.extend(exp_vecs)
            
        # Element-wise mean
        num_vecs = len(vectors)
        if num_vecs == 1:
            return q_vec
            
        dim = len(q_vec)
        final_vec = [0.0] * dim
        for v in vectors:
            for i in range(dim):
                final_vec[i] += v[i]
                
        return [x / num_vecs for x in final_vec]

    def _build_where_filter(self, base_filter: Dict[str, Any], doc_ids: Optional[List[str]]) -> Dict[str, Any]:
        """Helper to construct ChromaDB where clause with optional doc_id filtering."""
        if not doc_ids:
            return base_filter
            
        doc_filter = {}
        if len(doc_ids) == 1:
            doc_filter = {"doc_id": doc_ids[0]}
        else:
            doc_filter = {"doc_id": {"$in": doc_ids}}
            
        # Combine with base filter
        # Chroma requires $and for multiple conditions
        return {
            "$and": [
                base_filter,
                doc_filter
            ]
        }

    def _section_retrieval(self, q_emb: List[float], top_n: int = 5, doc_ids: Optional[List[str]] = None) -> List[RetrievalDoc]:
        where_filter = self._build_where_filter({"type": "section"}, doc_ids)
        return self.vectordb.query_by_embedding(self.sections_collection, q_emb, n_results=top_n,
                                               where=where_filter)

    def _chunk_retrieval_from_sections(self, q_emb: List[float], section_ids: List[str], k_per_section: int = 6) -> List[RetrievalDoc]:
        if not section_ids:
            return []
            
        # Optimization: Single query with $in operator instead of iterating
        # We increase n_results to cover all potential chunks
        total_k = len(section_ids) * k_per_section
        
        where_filter = {
            "$and": [
                {"type": "chunk"},
                {"section_id": {"$in": section_ids}}
            ]
        }
        
        try:
            return self.vectordb.query_by_embedding(self.chunks_collection, q_emb, n_results=total_k,
                                                   where=where_filter)
        except Exception as e:
            print(f"Batch chunk retrieval failed (likely $in operator not supported by this Chroma version?): {e}")
            # Fallback to iterative approach
            hits: List[RetrievalDoc] = []
            for sid in section_ids:
                res = self.vectordb.query_by_embedding(self.chunks_collection, q_emb, n_results=k_per_section,
                                                       where={"$and": [{"type": "chunk"}, {"section_id": sid}]})
                hits.extend(res)
            return hits

    def _chunk_retrieval_global(self, q_emb: List[float], top_k: int = 10, doc_ids: Optional[List[str]] = None) -> List[RetrievalDoc]:
        """
        Directly searches the chunks collection, bypassing the section hierarchy.
        Useful for catching specific keywords that might be missed in section summaries.
        """
        where_filter = self._build_where_filter({"type": "chunk"}, doc_ids)
        return self.vectordb.query_by_embedding(self.chunks_collection, q_emb, n_results=top_k,
                                               where=where_filter)

    # -------------------
    # LangGraph Nodes
    # -------------------
    def _node_query_expansion(self, state: GraphState) -> GraphState:
        '''
        This node expands the query and generates HyDE (Parallel)
        Run in Parallel the following tasks:
        1. Expand Query
        2. HyDE -> Generate hypothetical answer ->
        3. initial query embedding
        '''
        if self.status_callback:
            self.status_callback("Thinking: Expanding query & Generating HyDE (Parallel)...")
        q = state["query"]
        
        # Parallel Execution of Query Expansion and HyDE
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Task 1: Expand Query
            future_expand = executor.submit(self.llm.expand_query, q, max_suggestions=3)
            
            # Task 2: HyDE
            hyde_prompt = f"Write a short, concise paragraph answering the following question based on general knowledge. Question: {q}"
            def run_hyde():
                try:
                    if hasattr(self.llm, "generate"):
                        return self.llm.generate(hyde_prompt).content
                    else:
                        return self.llm.answer(q, "").answer
                except Exception as e:
                    print(f"HyDE generation failed: {e}")
                    return None
            future_hyde = executor.submit(run_hyde)
            
            # Task 3 (Optimization): Pre-compute Query Embedding
            future_q_emb = executor.submit(self.embedding_model.embed_text, q)
            
            # Wait for results
            expansions = future_expand.result().expanded_queries
            hypothetical = future_hyde.result()
            q_emb = future_q_emb.result()
            
        state["sub_queries"] = [q] + expansions
        state["hypothetical_answer"] = hypothetical
        state["query_embedding"] = q_emb
        
        # Parallel-ish: Embed new text content
        # We can run these in parallel too if we wanted, but embedding is fast.
        
        # 1. Embed HyDE
        h_emb = None
        if hypothetical:
            h_emb = self.embedding_model.embed_text(hypothetical)
        state["hyde_embedding"] = h_emb
            
        # 2. Embed Expansions (Batch)
        exp_embs = None
        if expansions:
             exp_embs = self.embedding_model.embed_documents(expansions)
        state["expanded_embeddings"] = exp_embs
            
        return state

    def _node_section_retrieval(self, state: GraphState) -> GraphState:
        if self.status_callback:
            self.status_callback("Thinking: Retrieving relevant sections...")
        
        # Mix embeddings from cache
        q_emb = state.get("query_embedding")
        h_emb = state.get("hyde_embedding")
        e_embs = state.get("expanded_embeddings")
        
        # Fallback if not in state (e.g. older state object)
        if q_emb is None:
             q_emb = self.embedding_model.embed_text(state["query"])
             
        final_emb = self._mix_embeddings(q_emb, h_vec=h_emb, exp_vecs=e_embs)
        
        doc_ids = state.get("doc_ids")
        state["section_hits"] = self._section_retrieval(final_emb, top_n=5, doc_ids=doc_ids)
        return state

    def _node_chunk_retrieval(self, state: GraphState) -> GraphState:
        if self.status_callback:
            self.status_callback("Thinking: Retrieving specific chunks (Hierarchical + Global + Memory)...")
        q = state["query"]
        
        # Use cached embeddings
        q_emb = state.get("query_embedding")
        h_emb = state.get("hyde_embedding")
        e_embs = state.get("expanded_embeddings")

        if q_emb is None:
             q_emb = self.embedding_model.embed_text(q)
        
        final_emb = self._mix_embeddings(q_emb, h_vec=h_emb, exp_vecs=e_embs)
        
        # Path A: Hierarchical Retrieval (Context-aware)
        sec_ids = [h.metadata.get("section_id", h.id) for h in state.get("section_hits", [])]
        sec_ids = [sid for sid in sec_ids if sid is not None]
        # We don't need to pass doc_ids here because we are already restricted to the sections 
        # that were retrieved (which were filtered by doc_id)
        hierarchical_chunks = self._chunk_retrieval_from_sections(final_emb, section_ids=sec_ids, k_per_section=6)
        
        # Path B: Global Retrieval (Keyword-aware)
        doc_ids = state.get("doc_ids")
        global_chunks = self._chunk_retrieval_global(final_emb, top_k=10, doc_ids=doc_ids)
        
        # Path C: Feedback/Memory Retrieval (Experience-aware)
        feedback_chunks = self._retrieve_feedback_chunks(q)
        if feedback_chunks:
            print(f"  -> Found {len(feedback_chunks)} proven chunks from feedback memory.")
        
        # Merge and Deduplicate
        all_hits = feedback_chunks + hierarchical_chunks + global_chunks
        seen = set()
        uniq: List[RetrievalDoc] = []
        for h in all_hits:
            if h.id not in seen:
                uniq.append(h)
                seen.add(h.id)
        
        # Reranking (Jina Reranker)
        # Check if embedding_model has 'rerank' (JinaEmbeddingModel does)
        if hasattr(self.embedding_model, "rerank") and len(uniq) > 0:
            if self.status_callback:
                self.status_callback(f"Thinking: Reranking {len(uniq)} candidate chunks...")
            try:
                from components.llm.models_const import JinaModels
                # Extract text for reranking
                docs_to_rerank = [d.text for d in uniq]
                rerank_res = self.embedding_model.rerank(
                    model=JinaModels.RerankerModels.JINA_RERANKER_V2_BASE_MULTILINGUAL,
                    query=q,
                    documents=docs_to_rerank,
                    top_n=12 # Keep top 12
                )
                
                # Re-order uniq based on rerank results
                reranked_docs = []
                if rerank_res and rerank_res.results:
                    for item in rerank_res.results:
                        # item.index is the index in the original list
                        original_doc = uniq[item.index]
                        # Update score if possible, or just append
                        reranked_docs.append(original_doc)
                    state["chunk_hits"] = reranked_docs
                else:
                    state["chunk_hits"] = uniq
            except Exception as e:
                print(f"Reranking failed: {e}")
                state["chunk_hits"] = uniq
        else:
            state["chunk_hits"] = uniq
            
        return state

    def _node_answer(self, state: GraphState) -> GraphState:
        if self.status_callback:
            self.status_callback("Thinking: Generating final answer...")
        q = state["query"]
        chunks = state.get("chunk_hits", [])
        top_chunks = chunks[:12]
        context = "\n\n".join([c.text for c in top_chunks])
        final = self.llm.answer(q, context).answer
        
        # Append Sources
        sources_text = "\n\n**Sources:**\n"
        seen_sources = set()
        for c in top_chunks:
            
            # Let's just append the Chunk ID and any metadata we have.
            meta = c.metadata
            page_info = f"Page {meta.get('page')}" if meta.get('page') else "Page ?"
            source_str = f"- {page_info} (Chunk {meta.get('chunk_index', '?')})"
            if source_str not in seen_sources:
                sources_text += source_str + "\n"
                seen_sources.add(source_str)
                
        state["final_answer"] = final + sources_text
        state["used_chunk_ids"] = [c.id for c in top_chunks]
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

    def run(self, query: str, doc_ids: Optional[List[str]] = None) -> GraphState:
        app = self.build_graph()
        # Initialize state with doc_ids
        initial_state: GraphState = {"query": query}
        if doc_ids:
            initial_state["doc_ids"] = doc_ids
            
        out: GraphState = app.invoke(initial_state)
        return out
