import unittest
from unittest.mock import MagicMock, patch

from agents.research_graph import (
    plan_search_node, 
    execute_search_node, 
    route_research,
    save_note_node,
    finalize_research_node
)


class ResearchAgentNodesTests(unittest.TestCase):
    @patch("agents.research_graph.generate_sub_queries")
    def test_plan_search_node_extracts_queries(self, mock_generate_sub_queries):
        state = {"topic": "AI agents", "debug_mode": False}
        llm = MagicMock()
        mock_generate_sub_queries.return_value = [
            "what are ai agents",
            "benefits of ai agents",
            "future of ai agents",
        ]
        
        updated_state = plan_search_node(state, llm=llm, max_queries=3)
        
        expected_queries = ["what are ai agents", "benefits of ai agents", "future of ai agents"]
        self.assertEqual(updated_state["search_queries"], expected_queries)

    @patch("agents.research_graph.generate_sub_queries")
    def test_plan_search_node_respects_max_queries(self, mock_generate_sub_queries):
        state = {"topic": "AI agents", "debug_mode": False}
        llm = MagicMock()
        mock_generate_sub_queries.return_value = ["q1", "q2"]
        
        updated_state = plan_search_node(state, llm=llm, max_queries=2)
        
        self.assertEqual(len(updated_state["search_queries"]), 2)
        self.assertEqual(updated_state["search_queries"], ["q1", "q2"])

    @patch("agents.research_graph.web_search")
    @patch("agents.research_graph.fetch_url_content")
    def test_execute_search_node_collects_results(self, mock_fetch, mock_search):
        mock_search.return_value = [
            {"url": "http://e1.com", "title": "T1", "snippet": "S1"},
            {"url": "http://e2.com", "title": "T2", "snippet": "S2"}
        ]
        mock_fetch.return_value = {"page_text": "Full Content"}
        
        state = {"search_queries": ["query 1"], "debug_mode": False}
        updated_state = execute_search_node(state)
        
        self.assertEqual(len(updated_state["search_results"]), 2)
        self.assertEqual(updated_state["search_results"][0]["content"], "Full Content")

    @patch("agents.research_graph.save_note")
    def test_save_note_node_calls_save_note_when_requested(self, mock_save):
        mock_save.return_value = "note_123"
        state = {
            "topic": "Research Topic",
            "synthesis": "The Synthesis",
            "save_to_notes": True,
            "search_results": [{"url": "U1", "snippet": "S1"}]
        }
        
        updated_state = save_note_node(state)
        
        self.assertEqual(updated_state["note_id"], "note_123")
        mock_save.assert_called_once()

    def test_save_note_node_skips_when_not_requested(self):
        state = {"save_to_notes": False}
        updated_state = save_note_node(state)
        self.assertNotIn("note_id", updated_state)

    def test_finalize_research_node_handles_no_results_error(self):
        state = {"topic": "Unknown topic", "error": "no_search_results"}
        updated_state = finalize_research_node(state)
        self.assertIn("unable to find", updated_state["synthesis"])

    def test_finalize_research_node_builds_lightweight_summary_when_results_exist(self):
        state = {
            "topic": "Topic",
            "search_results": [
                {"title": "Result One"},
                {"title": "Result Two"},
            ],
        }
        updated_state = finalize_research_node(state)
        self.assertIn("Collected web sources", updated_state["synthesis"])

    def test_route_research_returns_finalize_on_error(self):
        state = {"error": "no_search_results"}
        self.assertEqual(route_research(state), "finalize")

    def test_route_research_returns_finalize_when_no_error(self):
        state = {"search_results": [{"url": "http://e1.com"}]}
        self.assertEqual(route_research(state), "finalize")
