import unittest
from unittest.mock import MagicMock, patch

from agents.rag_graph import (
    build_initial_graph_state,
    decide_next_action_node,
    refine_query_node,
    retrieve_node,
)


class RefineQueryNodeTests(unittest.TestCase):
    def _make_llm(self, response_text):
        llm = MagicMock()
        llm.invoke.return_value = MagicMock(content=response_text)
        return llm

    def test_stores_refined_query_from_llm_response(self):
        state = build_initial_graph_state("What is machine learning?")
        llm = self._make_llm("machine learning definition and types")

        updated_state = refine_query_node(state, llm=llm)

        self.assertEqual(updated_state["refined_query"], "machine learning definition and types")

    def test_increments_retry_count(self):
        state = build_initial_graph_state("What is machine learning?")
        llm = self._make_llm("machine learning definition")

        updated_state = refine_query_node(state, llm=llm)

        self.assertEqual(updated_state["retry_count"], 1)

    def test_increments_retry_count_on_second_retry(self):
        state = build_initial_graph_state("What is machine learning?")
        state["retry_count"] = 1
        llm = self._make_llm("supervised learning algorithms")

        updated_state = refine_query_node(state, llm=llm)

        self.assertEqual(updated_state["retry_count"], 2)

    def test_falls_back_to_original_query_when_llm_returns_empty(self):
        state = build_initial_graph_state("What is machine learning?")
        llm = self._make_llm("   ")

        updated_state = refine_query_node(state, llm=llm)

        self.assertEqual(updated_state["refined_query"], "What is machine learning?")

    def test_records_refined_query_in_debug_data_when_debug_mode(self):
        state = build_initial_graph_state("What is ML?", debug_mode=True)
        state["debug_data"] = {"stage_counts": {}}
        llm = self._make_llm("machine learning overview")

        updated_state = refine_query_node(state, llm=llm)

        self.assertEqual(updated_state["debug_data"]["refined_query"], "machine learning overview")
        self.assertEqual(updated_state["debug_data"]["retry_count"], 1)


class RetryCapTests(unittest.TestCase):
    def _state_with_partial_evidence(self, retry_count):
        state = build_initial_graph_state("What is deep learning?")
        state["retry_count"] = retry_count
        state["retrieved_documents"] = [MagicMock()]
        state["expanded_documents"] = [MagicMock()]
        state["grounding"] = {"passed": False, "reason": "low_rerank_score"}
        return state

    def test_retries_when_partial_evidence_and_under_cap(self):
        state = self._state_with_partial_evidence(retry_count=0)

        updated_state = decide_next_action_node(state)

        self.assertEqual(updated_state["next_action"], "retry_retrieval")

    def test_retries_when_partial_evidence_and_one_retry_done(self):
        state = self._state_with_partial_evidence(retry_count=1)

        updated_state = decide_next_action_node(state)

        self.assertEqual(updated_state["next_action"], "retry_retrieval")

    def test_falls_back_when_retry_cap_reached(self):
        state = self._state_with_partial_evidence(retry_count=2)

        updated_state = decide_next_action_node(state)

        self.assertEqual(updated_state["next_action"], "fallback")
        self.assertEqual(updated_state["decision_reason"], "retry_cap_reached")


class RetrieveNodeUsesRefinedQueryTests(unittest.TestCase):
    def test_uses_refined_query_when_available(self):
        state = build_initial_graph_state("original query")
        state["refined_query"] = "refined query"

        with patch("agents.rag_graph.retrieve_documents_with_query_transform") as mock_retrieve:
            mock_retrieve.return_value = []
            retrieve_node(
                state,
                vectorstore=MagicMock(),
                reranker=None,
                bm25_index=None,
                llm=MagicMock(),
                retrieval_k=4,
                rerank_candidate_k=8,
                bm25_candidate_k=8,
                enable_query_transform=False,
            )
            used_query = mock_retrieve.call_args[0][1]
            self.assertEqual(used_query, "refined query")

    def test_uses_original_query_when_no_refined_query(self):
        state = build_initial_graph_state("original query")

        with patch("agents.rag_graph.retrieve_documents_with_query_transform") as mock_retrieve:
            mock_retrieve.return_value = []
            retrieve_node(
                state,
                vectorstore=MagicMock(),
                reranker=None,
                bm25_index=None,
                llm=MagicMock(),
                retrieval_k=4,
                rerank_candidate_k=8,
                bm25_candidate_k=8,
                enable_query_transform=False,
            )
            used_query = mock_retrieve.call_args[0][1]
            self.assertEqual(used_query, "original query")
