import unittest
from unittest.mock import Mock, patch
import os
import shutil
import tempfile
from typing import List, Optional

from components.hierarchy_late_chunk import HierarchyLateChunk
from components.data_structures import GraphState, RetrievalDoc
from components.dummy_llm import DummyLLM
from components.db.chroma_db import ChromaDb
from components.embedding_interface import EmbeddingInterface
from components.models.llm_model import SummarizeOutput, ExpandQueryOutput, AnswerOutput

# Dummy Embedding Model for testing
class DummyEmbeddingModel(EmbeddingInterface):
    def embed_text(self, text: str) -> List[float]:
        return [0.1] * 10

    def embed_documents(self, docs: List[str]) -> List[List[float]]:
        return [[0.2] * 10 for _ in docs]

    def embed_tokens(self, tokens: List[str]) -> Optional[List[List[float]]]:
        return [[0.3] * 10 for _ in tokens]

class TestHierarchyLateChunk(unittest.TestCase):

    def setUp(self):
        self.mock_llm = Mock()
        self.mock_embedding_model = Mock()
        self.mock_vectordb = Mock()

        self.hierarchy_late_chunk = HierarchyLateChunk(
            llm=self.mock_llm,
            embedding_model=self.mock_embedding_model,
            vectordb=self.mock_vectordb
        )

    def test_ingest_document_with_token_embeddings(self):
        doc_text = "This is a test document. It has multiple sentences."
        doc_id = "test_doc_123"

        # Mock token-level embeddings
        self.mock_embedding_model.embed_tokens.return_value = [[0.1]*10 for _ in doc_text.split()]
        self.mock_llm.summarize.return_value = SummarizeOutput(summary="summary")
        self.mock_embedding_model.embed_documents.return_value = [[0.2]*10]
        self.mock_vectordb.add.return_value = None

        result = self.hierarchy_late_chunk.ingest_document(doc_text, doc_id=doc_id)

        self.assertIn("doc_id", result)
        self.assertEqual(result["doc_id"], doc_id)
        self.assertIn("num_sections", result)
        self.assertIn("num_chunks", result)

        self.mock_embedding_model.embed_tokens.assert_called_once_with(doc_text.split())
        self.mock_llm.summarize.assert_called()
        self.mock_embedding_model.embed_documents.assert_called()
        self.mock_vectordb.add.assert_called()

    def test_ingest_document_without_token_embeddings_fallback(self):
        doc_text = "Another test document for fallback."
        doc_id = "test_doc_fallback"

        # Mock token-level embeddings to return None, triggering fallback
        self.mock_embedding_model.embed_tokens.return_value = None
        self.mock_embedding_model.embed_text.return_value = [0.5]*10 # Global vector
        self.mock_embedding_model.embed_documents.return_value = [[0.6]*10] # Raw chunk vecs
        self.mock_llm.summarize.return_value = SummarizeOutput(summary="fallback summary")
        self.mock_vectordb.add.return_value = None

        result = self.hierarchy_late_chunk.ingest_document(doc_text, doc_id=doc_id)

        self.assertIn("doc_id", result)
        self.assertEqual(result["doc_id"], doc_id)
        self.assertIn("num_sections", result)
        self.assertIn("num_chunks", result)

        self.mock_embedding_model.embed_tokens.assert_called_once_with(doc_text.split())
        self.mock_embedding_model.embed_text.assert_called_once_with(doc_text)
        self.mock_llm.summarize.assert_called()
        self.mock_embedding_model.embed_documents.assert_called()
        self.mock_vectordb.add.assert_called()

    @patch("docling.document_converter.DocumentConverter")
    @patch("os.path.exists")
    def test_ingest_from_file(self, mock_os_path_exists, MockDocumentConverter):
        mock_os_path_exists.return_value = True
        mock_converter_instance = Mock()
        mock_doc_instance = Mock()
        mock_converter_instance.convert.return_value.document = mock_doc_instance
        mock_doc_instance.export_to_text.return_value = "Content from file."
        MockDocumentConverter.return_value = mock_converter_instance

        # Mock the internal ingest_document call
        self.hierarchy_late_chunk.ingest_document = Mock(return_value={"doc_id": "file_doc", "num_sections": 1, "num_chunks": 1})

        file_path = "/path/to/fake_doc.pdf"
        result = self.hierarchy_late_chunk.ingest_from_file(file_path)

        mock_os_path_exists.assert_called_once_with(file_path)
        MockDocumentConverter.assert_called_once()
        mock_converter_instance.convert.assert_called_once_with(file_path)
        mock_doc_instance.export_to_text.assert_called_once()
        self.hierarchy_late_chunk.ingest_document.assert_called_once_with("Content from file.", doc_id="fake_doc.pdf")
        self.assertEqual(result["doc_id"], "file_doc")

    @patch("os.path.exists")
    def test_ingest_from_file_not_found(self, mock_os_path_exists):
        mock_os_path_exists.return_value = False
        file_path = "/path/to/non_existent_file.txt"
        with self.assertRaises(FileNotFoundError):
            self.hierarchy_late_chunk.ingest_from_file(file_path)

    def test_section_retrieval(self):
        mock_q_emb = [0.1]*10
        mock_results = [
            RetrievalDoc(id="sec1", text="Section 1", metadata={"type": "section"}, embedding=[0.2]*10),
            RetrievalDoc(id="sec2", text="Section 2", metadata={"type": "section"}, embedding=[0.3]*10)
        ]
        self.mock_vectordb.query_by_embedding.return_value = mock_results

        retrieved_sections = self.hierarchy_late_chunk._section_retrieval(mock_q_emb, top_n=2)

        self.mock_vectordb.query_by_embedding.assert_called_once_with(
            self.hierarchy_late_chunk.sections_collection, mock_q_emb, n_results=2, where={"type": "section"}
        )
        self.assertEqual(len(retrieved_sections), 2)
        self.assertEqual(retrieved_sections[0].id, "sec1")

    def test_chunk_retrieval_from_sections(self):
        section_ids = ["sec_a", "sec_b"]
        mock_q_emb = [0.4]*10
        mock_results_batch = [
            RetrievalDoc(id="chunk_a1", text="Chunk A1", metadata={"type": "chunk", "section_id": "sec_a"}, embedding=[0.5]*10),
            RetrievalDoc(id="chunk_a2", text="Chunk A2", metadata={"type": "chunk", "section_id": "sec_a"}, embedding=[0.6]*10),
            RetrievalDoc(id="chunk_b1", text="Chunk B1", metadata={"type": "chunk", "section_id": "sec_b"}, embedding=[0.7]*10)
        ]

        self.mock_vectordb.query_by_embedding.return_value = mock_results_batch

        retrieved_chunks = self.hierarchy_late_chunk._chunk_retrieval_from_sections(mock_q_emb, section_ids, k_per_section=2)

        self.mock_vectordb.query_by_embedding.assert_called_once()
        self.assertEqual(len(retrieved_chunks), 3)

    def test_node_query_expansion(self):
        initial_state = GraphState(query="original query")
        self.mock_llm.expand_query.return_value = ExpandQueryOutput(expanded_queries=["expanded query 1", "expanded query 2"])

        new_state = self.hierarchy_late_chunk._node_query_expansion(initial_state)

        self.mock_llm.expand_query.assert_called_once_with("original query", max_suggestions=3)
        self.assertEqual(new_state["sub_queries"], ["original query", "expanded query 1", "expanded query 2"])

    def test_node_section_retrieval(self):
        initial_state = GraphState(query="section query", query_embedding=[0.1]*10, hyde_embedding=None)
        mock_sections = [
            RetrievalDoc(id="sec_x", text="Sec X", metadata={"type": "section"}),
            RetrievalDoc(id="sec_y", text="Sec Y", metadata={"type": "section"})
        ]
        self.hierarchy_late_chunk._section_retrieval = Mock(return_value=mock_sections)

        new_state = self.hierarchy_late_chunk._node_section_retrieval(initial_state)

        # Should be called with the [0.1]*10 embedding from state
        self.hierarchy_late_chunk._section_retrieval.assert_called_once_with([0.1]*10, top_n=5)
        self.assertEqual(new_state["section_hits"], mock_sections)

    def test_node_chunk_retrieval(self):
        initial_state = GraphState(query="chunk query", 
                                   query_embedding=[0.1]*10,
                                   hyde_embedding=None,
                                   section_hits=[
            RetrievalDoc(id="sec_1", text="Sec 1", metadata={"section_id": "sec_1"}),
            RetrievalDoc(id="sec_2", text="Sec 2", metadata={"section_id": "sec_2"})
        ])
        mock_chunks = [
            RetrievalDoc(id="ch_a", text="Chunk A", metadata={"type": "chunk"}),
            RetrievalDoc(id="ch_b", text="Chunk B", metadata={"type": "chunk"})
        ]
        self.hierarchy_late_chunk._chunk_retrieval_from_sections = Mock(return_value=mock_chunks)
        self.hierarchy_late_chunk._chunk_retrieval_global = Mock(return_value=[])
        self.hierarchy_late_chunk._retrieve_feedback_chunks = Mock(return_value=[])

        new_state = self.hierarchy_late_chunk._node_chunk_retrieval(initial_state)

        self.hierarchy_late_chunk._chunk_retrieval_from_sections.assert_called_once_with([0.1]*10, section_ids=["sec_1", "sec_2"], k_per_section=6)
        self.hierarchy_late_chunk._chunk_retrieval_global.assert_called_once_with([0.1]*10, top_k=10)
        self.assertEqual(new_state["chunk_hits"], mock_chunks)

    def test_node_answer(self):
        initial_state = GraphState(query="final question", chunk_hits=[
            RetrievalDoc(id="ch_1", text="Context 1", metadata={}),
            RetrievalDoc(id="ch_2", text="Context 2", metadata={})
        ])
        self.mock_llm.answer.return_value = AnswerOutput(answer="Final Answer Text")

        new_state = self.hierarchy_late_chunk._node_answer(initial_state)

        self.mock_llm.answer.assert_called_once_with("final question", "Context 1\n\nContext 2")
        self.assertIn("Final Answer Text", new_state["final_answer"])
        self.assertIn("**Sources:**", new_state["final_answer"])

    def test_build_graph_and_run(self):
        # Mock the nodes to control their behavior during graph execution
        self.hierarchy_late_chunk._node_query_expansion = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=["q1", "q2"]))
        self.hierarchy_late_chunk._node_section_retrieval = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=state["sub_queries"], section_hits=["sec_hit"]))
        self.hierarchy_late_chunk._node_chunk_retrieval = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=state["sub_queries"], section_hits=state["section_hits"], chunk_hits=[RetrievalDoc(id="ch", text="final context", metadata={})]))
        self.hierarchy_late_chunk._node_answer = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=state["sub_queries"], section_hits=state["section_hits"], chunk_hits=state["chunk_hits"], final_answer="Graph Final Answer"))

        query = "Graph test query"
        result = self.hierarchy_late_chunk.run(query)
        self.assertEqual(result["final_answer"], "Graph Final Answer")
        self.hierarchy_late_chunk._node_query_expansion.assert_called_once()
        self.hierarchy_late_chunk._node_section_retrieval.assert_called_once()
        self.hierarchy_late_chunk._node_chunk_retrieval.assert_called_once()
        self.hierarchy_late_chunk._node_answer.assert_called_once()



class TestHierarchyLateChunkIntegration(unittest.TestCase):

    def setUp(self):
        # Create a temporary directory for ChromaDB
        self.test_dir = tempfile.mkdtemp()
        self.db_path = os.path.join(self.test_dir, "chroma_test_db")

        # Create a dummy markdown file
        self.test_md_file_path = os.path.join(self.test_dir, "test_doc.md")
        with open(self.test_md_file_path, "w") as f:
            # Make it long enough to create multiple chunks and sections
            f.write("The quick brown fox jumps over the lazy dog. " * 200)

        # Create a dummy txt file
        self.test_txt_file_path = os.path.join(self.test_dir, "test_doc.txt")
        with open(self.test_txt_file_path, "w") as f:
            f.write("This is a sentence in a txt file. " * 150)

        # Setup components
        self.llm = DummyLLM()
        self.embedding_model = DummyEmbeddingModel()
        self.vectordb = ChromaDb(persist_directory=self.db_path)
        
        self.pipeline = HierarchyLateChunk(
            llm=self.llm,
            embedding_model=self.embedding_model,
            vectordb=self.vectordb
        )

    def tearDown(self):
        # Clean up the temporary directory
        shutil.rmtree(self.test_dir)

    def test_e2e_ingestion_and_query(self):
        # 1. Ingest the document
        ingest_info = self.pipeline.ingest_from_file(self.test_md_file_path)
        
        self.assertIsNotNone(ingest_info)
        self.assertEqual(ingest_info["doc_id"], "test_doc.md")
        self.assertTrue(ingest_info["num_chunks"] > 0)
        self.assertTrue(ingest_info["num_sections"] > 0)

        # Check if data is in vectordb
        sections_count = self.vectordb.get_or_create(self.pipeline.sections_collection).count()
        chunks_count = self.vectordb.get_or_create(self.pipeline.chunks_collection).count()
        self.assertEqual(sections_count, ingest_info["num_sections"])
        self.assertEqual(chunks_count, ingest_info["num_chunks"])

        # 2. Run a query
        query = "What this pdf contain?"
        answer = self.pipeline.run(query)

        self.assertIsNotNone(answer)
        # DummyLLM returns a canned response, so we check for keywords in the context it received
        self.assertIn("fox", answer["final_answer"].lower())
        self.assertIn("jumps", answer["final_answer"].lower())
        self.assertIn("lazy dog", answer["final_answer"].lower())
        self.assertIn("q: what this pdf contain?", answer["final_answer"].lower())

    def test_e2e_pdf_ingestion(self):
        # 1. Ingest the PDF document

        pdf_path = os.path.join(os.path.dirname(__file__), "test_files/test_pdf.pdf") # Adjust path to root
        ingest_info = self.pipeline.ingest_from_file(pdf_path)

        self.assertIsNotNone(ingest_info)
        self.assertEqual(ingest_info["doc_id"], "test_pdf.pdf")
        self.assertTrue(ingest_info["num_chunks"] > 0)
        self.assertTrue(ingest_info["num_sections"] > 0)

        # Check if data is in vectordb
        sections_count = self.vectordb.get_or_create(self.pipeline.sections_collection).count()
        chunks_count = self.vectordb.get_or_create(self.pipeline.chunks_collection).count()
        self.assertEqual(sections_count, ingest_info["num_sections"])
        self.assertEqual(chunks_count, ingest_info["num_chunks"])

    def test_e2e_txt_ingestion(self):
        # 1. Ingest the document
        ingest_info = self.pipeline.ingest_from_file(self.test_txt_file_path)
        
        self.assertIsNotNone(ingest_info)
        self.assertEqual(ingest_info["doc_id"], "test_doc.txt")
        self.assertTrue(ingest_info["num_chunks"] > 0)
        self.assertTrue(ingest_info["num_sections"] > 0)

        # Check if data is in vectordb
        sections_count = self.vectordb.get_or_create(self.pipeline.sections_collection).count()
        chunks_count = self.vectordb.get_or_create(self.pipeline.chunks_collection).count()
        self.assertEqual(sections_count, ingest_info["num_sections"])
        self.assertEqual(chunks_count, ingest_info["num_chunks"])


if __name__ == "__main__":
    unittest.main()

        