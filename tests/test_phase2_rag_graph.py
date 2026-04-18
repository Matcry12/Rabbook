import unittest
from unittest.mock import patch

from agents.rag_graph import (
    build_initial_graph_state,
    check_grounding_node,
    decide_next_action_node,
    expand_context_node,
    fallback_answer_node,
    generate_answer_node,
    prepare_input_node,
    route_after_grounding,
    run_rag_graph_answer,
    retrieve_node,
)


class Phase2RagGraphTests(unittest.TestCase):
    def test_build_initial_graph_state_sets_expected_defaults(self):
        state = build_initial_graph_state(
            "What is this roadmap about?",
            selected_file="roadmap.pdf",
            debug_mode=True,
        )

        self.assertEqual(state["query"], "What is this roadmap about?")
        self.assertEqual(state["selected_file"], "roadmap.pdf")
        self.assertTrue(state["debug_mode"])
        self.assertEqual(state["retrieved_documents"], [])
        self.assertIsNone(state["metadata_filter"])
        self.assertIsNone(state["next_action"])

    def test_prepare_input_node_builds_metadata_filter(self):
        state = build_initial_graph_state(
            "What is this roadmap about?",
            selected_file="roadmap.pdf",
            selected_file_type="pdf",
            page_start="4",
            page_end="2",
        )

        updated_state = prepare_input_node(state)

        self.assertEqual(
            updated_state["metadata_filter"],
            {
                "file_name": "roadmap.pdf",
                "file_type": "pdf",
                "page_range": {"start": 2, "end": 4},
            },
        )

    @patch("agents.rag_graph.retrieve_documents_with_query_transform")
    def test_retrieve_node_stores_retrieved_documents_and_debug_data(self, mock_retrieve):
        state = build_initial_graph_state(
            "What is this roadmap about?",
            selected_file="roadmap.pdf",
            debug_mode=True,
        )
        state["metadata_filter"] = {"file_name": "roadmap.pdf"}
        mock_retrieve.return_value = (["doc-a", "doc-b"], {"stage_counts": {"search_queries": 1}})

        updated_state = retrieve_node(
            state,
            vectorstore=object(),
            reranker=object(),
            bm25_index=object(),
            llm=object(),
            retrieval_k=4,
            rerank_candidate_k=8,
            bm25_candidate_k=8,
            enable_query_transform=True,
        )

        self.assertEqual(updated_state["retrieved_documents"], ["doc-a", "doc-b"])
        self.assertEqual(updated_state["debug_data"]["metadata_filter"], {"file_name": "roadmap.pdf"})
        self.assertEqual(updated_state["debug_data"]["grounding"]["reason"], "not_checked")

    @patch("agents.rag_graph.expand_with_context_window")
    def test_expand_context_node_stores_expanded_documents(self, mock_expand):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["retrieved_documents"] = ["doc-a"]
        state["debug_data"] = {"stage_counts": {}, "grounding": {"reason": "not_checked"}}
        mock_expand.return_value = ["doc-a", "doc-a-next"]

        updated_state = expand_context_node(
            state,
            chunk_registry={"by_chunk_id": {}},
            context_window=1,
            max_expanded_chunks=12,
        )

        self.assertEqual(updated_state["expanded_documents"], ["doc-a", "doc-a-next"])
        self.assertEqual(updated_state["debug_data"]["expanded_hits"], ["doc-a", "doc-a-next"])
        self.assertEqual(updated_state["debug_data"]["stage_counts"]["expanded_context"], 2)

    @patch("agents.rag_graph.check_grounding_evidence")
    def test_check_grounding_node_stores_grounding_result(self, mock_check_grounding):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["retrieved_documents"] = ["doc-a"]
        state["expanded_documents"] = ["doc-a", "doc-a-next"]
        state["debug_data"] = {
            "stage_counts": {},
            "grounding": {"stage": "retrieval", "passed": None, "reason": "not_checked"},
        }
        mock_check_grounding.return_value = {
            "passed": True,
            "reason": "enough_evidence",
            "top_rerank_score": 1.2,
            "retrieved_count": 1,
            "expanded_count": 2,
        }

        updated_state = check_grounding_node(
            state,
            min_grounded_rerank_score=1.0,
            min_grounded_chunks=1,
        )

        self.assertTrue(updated_state["grounding"]["passed"])
        self.assertEqual(updated_state["debug_data"]["grounding"]["reason"], "enough_evidence")
        self.assertEqual(updated_state["debug_data"]["grounding"]["stage"], "retrieval")

    def test_route_after_grounding_chooses_fallback_when_evidence_is_weak(self):
        state = build_initial_graph_state("What is this roadmap about?")
        state["next_action"] = "retry_retrieval"

        next_node = route_after_grounding(state)

        self.assertEqual(next_node, "fallback_answer")

    def test_decide_next_action_node_marks_retry_when_local_evidence_is_partial(self):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["retrieved_documents"] = ["doc-a"]
        state["expanded_documents"] = ["doc-a", "doc-a-next"]
        state["grounding"] = {"passed": False}
        state["debug_data"] = {"grounding": {}, "stage_counts": {}}

        updated_state = decide_next_action_node(state)

        self.assertEqual(updated_state["next_action"], "retry_retrieval")
        self.assertEqual(updated_state["decision_reason"], "partial_local_evidence")
        self.assertEqual(updated_state["debug_data"]["next_action"], "retry_retrieval")

    def test_decide_next_action_node_marks_fallback_when_no_local_evidence_exists(self):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["grounding"] = {"passed": False}
        state["debug_data"] = {"grounding": {}, "stage_counts": {}}

        updated_state = decide_next_action_node(state)

        self.assertEqual(updated_state["next_action"], "fallback")
        self.assertEqual(updated_state["decision_reason"], "no_local_evidence")

    def test_decide_next_action_node_marks_answer_when_grounding_passes(self):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["grounding"] = {"passed": True}
        state["debug_data"] = {"grounding": {}, "stage_counts": {}}

        updated_state = decide_next_action_node(state)

        self.assertEqual(updated_state["next_action"], "answer")
        self.assertEqual(updated_state["decision_reason"], "grounding_passed")

    @patch("agents.rag_graph.build_sources")
    def test_fallback_answer_node_stores_fallback_answer_and_sources(self, mock_build_sources):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["retrieved_documents"] = ["doc-a"]
        state["debug_data"] = {"grounding": {}, "stage_counts": {}}
        mock_build_sources.return_value = [{"chunk_id": "c1"}]

        updated_state = fallback_answer_node(
            state,
            grounded_fallback_message="fallback",
        )

        self.assertEqual(updated_state["answer"], "fallback")
        self.assertEqual(updated_state["sources"], [{"chunk_id": "c1"}])
        self.assertEqual(updated_state["citations"], [])
        self.assertEqual(updated_state["debug_data"]["graph_path"], "fallback_answer")

    @patch("agents.rag_graph.build_citations")
    @patch("agents.rag_graph.answer_is_grounded")
    @patch("agents.rag_graph.generate_answer")
    @patch("agents.rag_graph.format_context")
    @patch("agents.rag_graph.build_sources")
    def test_generate_answer_node_stores_answer_sources_and_citations(
        self,
        mock_build_sources,
        mock_format_context,
        mock_generate_answer,
        mock_answer_is_grounded,
        mock_build_citations,
    ):
        state = build_initial_graph_state("What is this roadmap about?", debug_mode=True)
        state["retrieved_documents"] = ["doc-a"]
        state["expanded_documents"] = ["doc-a", "doc-a-next"]
        state["grounding"] = {
            "passed": True,
            "top_rerank_score": 1.2,
            "retrieved_count": 1,
            "expanded_count": 2,
        }
        state["debug_data"] = {"grounding": {}, "stage_counts": {}}
        mock_build_sources.return_value = [{"chunk_id": "c1"}]
        mock_format_context.return_value = "[1] expanded context"
        mock_generate_answer.return_value = "Grounded answer [1]"
        mock_answer_is_grounded.return_value = True
        mock_build_citations.return_value = [{"number": 1}]

        updated_state = generate_answer_node(
            state,
            llm=object(),
            grounded_fallback_message="fallback",
        )

        self.assertEqual(updated_state["answer"], "Grounded answer [1]")
        self.assertEqual(updated_state["sources"], [{"chunk_id": "c1"}])
        self.assertEqual(updated_state["citations"], [{"number": 1}])
        self.assertEqual(updated_state["debug_data"]["graph_path"], "generate_answer")
        self.assertEqual(updated_state["debug_data"]["grounding"]["reason"], "answer_is_grounded")

    @patch("agents.rag_graph.build_rag_graph")
    def test_run_rag_graph_answer_returns_answer_result_shape(self, mock_build_graph):
        fake_graph = mock_build_graph.return_value
        fake_graph.invoke.return_value = {
            "answer": "Graph answer [1]",
            "sources": [{"chunk_id": "c1"}],
            "citations": [{"number": 1}],
            "debug_data": {"grounding": {"reason": "answer_is_grounded"}},
        }

        result = run_rag_graph_answer(
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
            selected_file="roadmap.pdf",
            debug_mode=True,
        )

        self.assertEqual(result.answer, "Graph answer [1]")
        self.assertEqual(result.sources, [{"chunk_id": "c1"}])
        self.assertEqual(result.citations, [{"number": 1}])
        self.assertEqual(result.debug_data["pipeline_mode"], "langgraph_rag")
        self.assertEqual(result.debug_data["grounding"]["reason"], "answer_is_grounded")


if __name__ == "__main__":
    unittest.main()
