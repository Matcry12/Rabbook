import unittest
from unittest.mock import patch

from agents.services import AnswerResult, answer_query, build_metadata_filter, build_page_range


class FakeDoc:
    def __init__(self, content, metadata):
        self.page_content = content
        self.metadata = metadata


class Phase1ArchitectureTests(unittest.TestCase):
    def test_build_metadata_filter_swaps_reversed_page_range(self):
        result = build_metadata_filter(
            selected_file="roadmap.pdf",
            selected_file_type="pdf",
            page_start="5",
            page_end="2",
        )

        self.assertEqual(
            result,
            {
                "file_name": "roadmap.pdf",
                "file_type": "pdf",
                "page_range": {"start": 2, "end": 5},
            },
        )

    def test_build_page_range_rejects_zero_or_negative_pages(self):
        with self.assertRaises(ValueError):
            build_page_range("0", "")

    @patch("agents.services.build_citation_sources")
    @patch("agents.services.answer_has_valid_citations")
    @patch("agents.services.generate_answer")
    @patch("agents.services.format_context")
    @patch("agents.services.check_grounding_evidence")
    @patch("agents.services.expand_with_context_window")
    @patch("agents.services.retrieve_documents_with_query_transform")
    def test_answer_query_returns_grounded_answer_result(
        self,
        mock_retrieve,
        mock_expand,
        mock_grounding,
        mock_format_context,
        mock_generate_answer,
        mock_answer_has_valid_citations,
        mock_build_citation_sources,
    ):
        retrieved_documents = [
            (FakeDoc("retrieved chunk", {"file_name": "roadmap.pdf", "chunk_id": "c1"}), 0.91)
        ]
        expanded_documents = [
            (FakeDoc("expanded chunk", {"file_name": "roadmap.pdf", "chunk_id": "c1"}), 0.91)
        ]
        mock_retrieve.return_value = (retrieved_documents, {"stage_counts": {}})
        mock_expand.return_value = expanded_documents
        mock_grounding.return_value = {
            "passed": True,
            "top_rerank_score": 0.91,
            "retrieved_count": 1,
            "expanded_count": 1,
        }
        mock_format_context.return_value = "[1] expanded chunk"
        mock_generate_answer.return_value = "Grounded answer [1]"
        mock_answer_has_valid_citations.return_value = True
        mock_build_citation_sources.return_value = [
            {
                "number": 1,
                "source": "roadmap.pdf",
                "retrieval_score": 0.91,
                "rerank_score": 0.95,
                "content": "expanded chunk",
            }
        ]

        result = answer_query(
            "What is this roadmap about?",
            vectorstore=object(),
            chunk_registry={"by_chunk_id": {}},
            reranker=object(),
            bm25_index=object(),
            llm=object(),
            retrieval_k=4,
            rerank_candidate_k=8,
            bm25_candidate_k=8,
            context_window=1,
            max_expanded_chunks=12,
            min_grounded_rerank_score=1.0,
            min_grounded_chunks=1,
            grounded_fallback_message="fallback",
            enable_query_transform=True,
            debug_mode=True,
        )

        self.assertEqual(result.answer, "Grounded answer [1]")
        self.assertEqual(result.sources[0]["chunk_id"], "c1")
        self.assertEqual(result.citations[0]["number"], 1)
        self.assertTrue(result.debug_data["grounding"]["passed"])

    @patch("agents.services.run_rag_graph_answer")
    def test_answer_query_can_delegate_to_langgraph_runner(self, mock_run_rag_graph_answer):
        mock_run_rag_graph_answer.return_value = AnswerResult(
            answer="Graph answer [1]",
            sources=[{"chunk_id": "c1"}],
            citations=[{"number": 1}],
            debug_data={"grounding": {"reason": "answer_is_grounded"}},
        )

        result = answer_query(
            "What is this roadmap about?",
            vectorstore=object(),
            chunk_registry={"by_chunk_id": {}},
            reranker=object(),
            bm25_index=object(),
            llm=object(),
            retrieval_k=4,
            rerank_candidate_k=8,
            bm25_candidate_k=8,
            context_window=1,
            max_expanded_chunks=12,
            min_grounded_rerank_score=1.0,
            min_grounded_chunks=1,
            grounded_fallback_message="fallback",
            enable_query_transform=True,
            debug_mode=True,
            use_langgraph=True,
        )

        self.assertEqual(result.answer, "Graph answer [1]")
        self.assertEqual(result.sources, [{"chunk_id": "c1"}])
        self.assertEqual(result.citations, [{"number": 1}])


if __name__ == "__main__":
    unittest.main()
