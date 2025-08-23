import unittest
from unittest.mock import Mock, patch
import os
import uuid

from components.hierarchy_late_chunk import HierarchyLateChunk
from components.data_structures import GraphState, RetrievalDoc

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
        self.mock_llm.summarize.return_value = "summary"
        self.mock_embedding_model.embed_documents.return_value = [[0.2]*10]
        self.mock_vectordb.add.return_value = None

        result = self.hierarchy_late_chunk.ingest_document(doc_text, doc_id=doc_id)

        self.assertIn("doc_id", result)
        self.assertEqual(result["doc_id"], doc_id)
        self.assertIn("num_sections", result)
        self.assertIn("num_chunks", result)

        self.mock_embedding_model.embed_tokens.assert_called_once_with(doc_text)
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
        self.mock_llm.summarize.return_value = "fallback summary"
        self.mock_vectordb.add.return_value = None

        result = self.hierarchy_late_chunk.ingest_document(doc_text, doc_id=doc_id)

        self.assertIn("doc_id", result)
        self.assertEqual(result["doc_id"], doc_id)
        self.assertIn("num_sections", result)
        self.assertIn("num_chunks", result)

        self.mock_embedding_model.embed_tokens.assert_called_once_with(doc_text)
        self.mock_embedding_model.embed_text.assert_called_once_with(doc_text)
        self.mock_llm.summarize.assert_called()
        self.mock_embedding_model.embed_documents.assert_called()
        self.mock_vectordb.add.assert_called()

    @patch("components.hierarchy_late_chunk.DocumentConverter")
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
        query = "test query"
        mock_q_emb = [0.1]*10
        mock_results = {
            "ids": [["sec1", "sec2"]],
            "documents": [["Section 1", "Section 2"]],
            "metadatas": [[{"type": "section"}, {"type": "section"}]],
            "embeddings": [[ [0.2]*10, [0.3]*10 ]]
        }
        self.mock_embedding_model.embed_text.return_value = mock_q_emb
        self.mock_vectordb.query_by_embedding.return_value = mock_results

        retrieved_sections = self.hierarchy_late_chunk._section_retrieval(query, top_n=2)

        self.mock_embedding_model.embed_text.assert_called_once_with(query)
        self.mock_vectordb.query_by_embedding.assert_called_once_with(
            self.hierarchy_late_chunk.sections_collection, mock_q_emb, n_results=2, where={"type": "section"}
        )
        self.assertEqual(len(retrieved_sections), 2)
        self.assertEqual(retrieved_sections[0].id, "sec1")
        self.assertEqual(retrieved_sections[1].id, "sec2")

    def test_chunk_retrieval_from_sections(self):
        query = "test query for chunks"
        section_ids = ["sec_a", "sec_b"]
        mock_q_emb = [0.4]*10
        mock_results_sec_a = {
            "ids": [["chunk_a1", "chunk_a2"]],
            "documents": [["Chunk A1", "Chunk A2"]],
            "metadatas": [[{"type": "chunk", "section_id": "sec_a"}, {"type": "chunk", "section_id": "sec_a"}]],
            "embeddings": [[ [0.5]*10, [0.6]*10 ]]
        }
        mock_results_sec_b = {
            "ids": [["chunk_b1"]],
            "documents": [["Chunk B1"]],
            "metadatas": [[{"type": "chunk", "section_id": "sec_b"}]],
            "embeddings": [[ [0.7]*10 ]]
        }

        self.mock_embedding_model.embed_text.return_value = mock_q_emb
        self.mock_vectordb.query_by_embedding.side_effect = [mock_results_sec_a, mock_results_sec_b]

        retrieved_chunks = self.hierarchy_late_chunk._chunk_retrieval_from_sections(query, section_ids, k_per_section=2)

        self.mock_embedding_model.embed_text.assert_called_once_with(query)
        self.assertEqual(self.mock_vectordb.query_by_embedding.call_count, 2)
        self.assertEqual(len(retrieved_chunks), 3)
        self.assertIn("chunk_a1", [c.id for c in retrieved_chunks])
        self.assertIn("chunk_a2", [c.id for c in retrieved_chunks])
        self.assertIn("chunk_b1", [c.id for c in retrieved_chunks])

    def test_node_query_expansion(self):
        initial_state = GraphState(query="original query")
        self.mock_llm.expand_query.return_value = ["expanded query 1", "expanded query 2"]

        new_state = self.hierarchy_late_chunk._node_query_expansion(initial_state)

        self.mock_llm.expand_query.assert_called_once_with("original query", max_suggestions=3)
        self.assertEqual(new_state["sub_queries"], ["original query", "expanded query 1", "expanded query 2"])

    def test_node_section_retrieval(self):
        initial_state = GraphState(query="section query")
        mock_sections = [
            RetrievalDoc(id="sec_x", text="Sec X", metadata={"type": "section"}),
            RetrievalDoc(id="sec_y", text="Sec Y", metadata={"type": "section"})
        ]
        self.hierarchy_late_chunk._section_retrieval = Mock(return_value=mock_sections)

        new_state = self.hierarchy_late_chunk._node_section_retrieval(initial_state)

        self.hierarchy_late_chunk._section_retrieval.assert_called_once_with("section query", top_n=3)
        self.assertEqual(new_state["section_hits"], mock_sections)

    def test_node_chunk_retrieval(self):
        initial_state = GraphState(query="chunk query", section_hits=[
            RetrievalDoc(id="sec_1", text="Sec 1", metadata={"section_id": "sec_1"}),
            RetrievalDoc(id="sec_2", text="Sec 2", metadata={"section_id": "sec_2"})
        ])
        mock_chunks = [
            RetrievalDoc(id="ch_a", text="Chunk A", metadata={"type": "chunk"}),
            RetrievalDoc(id="ch_b", text="Chunk B", metadata={"type": "chunk"})
        ]
        self.hierarchy_late_chunk._chunk_retrieval_from_sections = Mock(return_value=mock_chunks)

        new_state = self.hierarchy_late_chunk._node_chunk_retrieval(initial_state)

        self.hierarchy_late_chunk._chunk_retrieval_from_sections.assert_called_once_with("chunk query", ["sec_1", "sec_2"], k_per_section=4)
        self.assertEqual(new_state["chunk_hits"], mock_chunks)

    def test_node_answer(self):
        initial_state = GraphState(query="final question", chunk_hits=[
            RetrievalDoc(id="ch_1", text="Context 1", metadata={}),
            RetrievalDoc(id="ch_2", text="Context 2", metadata={})
        ])
        self.mock_llm.answer.return_value = "Final Answer Text"

        new_state = self.hierarchy_late_chunk._node_answer(initial_state)

        expected_context = "Context 1\n\nContext 2" # Note: \n\n is due to how join works with literal strings
        self.mock_llm.answer.assert_called_once_with("final question", "Context 1\n\nContext 2")
        self.assertEqual(new_state["final_answer"], "Final Answer Text")

    def test_build_graph_and_run(self):
        # Mock the nodes to control their behavior during graph execution
        self.hierarchy_late_chunk._node_query_expansion = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=["q1", "q2"]))
        self.hierarchy_late_chunk._node_section_retrieval = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=state["sub_queries"], section_hits=["sec_hit"]))
        self.hierarchy_late_chunk._node_chunk_retrieval = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=state["sub_queries"], section_hits=state["section_hits"], chunk_hits=[RetrievalDoc(id="ch", text="final context", metadata={})]))
        self.hierarchy_late_chunk._node_answer = Mock(side_effect=lambda state: GraphState(query=state["query"], sub_queries=state["sub_queries"], section_hits=state["section_hits"], chunk_hits=state["chunk_hits"], final_answer="Graph Final Answer"))

        query = "Graph test query"
        result = self.hierarchy_late_chunk.run(query)

        self.assertEqual(result, "Graph Final Answer")
        self.hierarchy_late_chunk._node_query_expansion.assert_called_once()
        self.hierarchy_late_chunk._node_section_retrieval.assert_called_once()
        self.hierarchy_late_chunk._node_chunk_retrieval.assert_called_once()
        self.hierarchy_late_chunk._node_answer.assert_called_once()

    def test_pack_results(self):
        from components.utils import _pack_results
        # Test case 1: Empty results
        empty_res = {"ids": [[]], "documents": [[]], "metadatas": [[]], "embeddings": [[]]}
        self.assertEqual(_pack_results(empty_res), [])

        # Test case 2: Single result
        single_res = {
            "ids": [["id1"]],
            "documents": [["doc1"]],
            "metadatas": [[{"source": "test"}]],
            "embeddings": [[ [0.1, 0.2] ]]
        }
        expected_single = [
            RetrievalDoc(id="id1", text="doc1", metadata={"source": "test"}, embedding=[0.1, 0.2])
        ]
        self.assertEqual(_pack_results(single_res), expected_single)

        # Test case 3: Multiple results
        multi_res = {
            "ids": [["id1", "id2"]],
            "documents": [["doc1", "doc2"]],
            "metadatas": [[{"source": "test1"}, {"source": "test2"}]],
            "embeddings": [[ [0.1, 0.2], [0.3, 0.4] ]]
        }
        expected_multi = [
            RetrievalDoc(id="id1", text="doc1", metadata={"source": "test1"}, embedding=[0.1, 0.2]),
            RetrievalDoc(id="id2", text="doc2", metadata={"source": "test2"}, embedding=[0.3, 0.4])
        ]
        self.assertEqual(_pack_results(multi_res), expected_multi)

        # Test case 4: Results with missing optional fields (embeddings)
        no_emb_res = {
            "ids": [["id3"]],
            "documents": [["doc3"]],
            "metadatas": [[{"source": "test3"}]],
            "embeddings": [[]]
        }
        expected_no_emb = [
            RetrievalDoc(id="id3", text="doc3", metadata={"source": "test3"}, embedding=None)
        ]
        self.assertEqual(_pack_results(no_emb_res), expected_no_emb)

        # Test case 5: Results with missing optional fields (metadatas)
        no_meta_res = {
            "ids": [["id4"]],
            "documents": [["doc4"]],
            "metadatas": [[]],
            "embeddings": [[ [0.5, 0.6] ]]
        }
        expected_no_meta = [
            RetrievalDoc(id="id4", text="doc4", metadata={}, embedding=[0.5, 0.6])
        ]
        self.assertEqual(_pack_results(no_meta_res), expected_no_meta)


if __name__ == "__main__":
    unittest.main()
