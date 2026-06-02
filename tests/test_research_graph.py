import unittest
from unittest.mock import patch

from agents.research_graph import run_research_agent


class ResearchGraphResultTests(unittest.TestCase):
    @patch("agents.research_graph.build_research_graph")
    def test_run_research_agent_returns_full_content_in_sources(self, mock_build_graph):
        fake_graph = mock_build_graph.return_value
        fake_graph.invoke.return_value = {
            "synthesis": "Research summary",
            "search_results": [
                {
                    "url": "https://example.com/page",
                    "title": "Example Page",
                    "snippet": "Short snippet",
                    "content": "Full fetched page text",
                }
            ],
            "note_id": None,
            "debug_data": None,
        }

        result = run_research_agent(
            topic="example topic",
            llm=object(),
            debug_mode=False,
        )

        self.assertEqual(result.synthesis, "Research summary")
        self.assertEqual(result.sources[0]["url"], "https://example.com/page")
        self.assertEqual(result.sources[0]["snippet"], "Short snippet")
        self.assertEqual(result.sources[0]["content"], "Full fetched page text")


if __name__ == "__main__":
    unittest.main()
