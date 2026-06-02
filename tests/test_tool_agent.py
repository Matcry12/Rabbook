import unittest
from unittest.mock import MagicMock, patch

from agents.tool_agent import run_tool_agent


def make_ai_response(content="", tool_calls=None):
    response = MagicMock()
    response.content = content
    response.tool_calls = tool_calls or []
    return response


class ToolAgentLoopTests(unittest.TestCase):
    def test_returns_final_answer_when_no_tool_calls(self):
        llm = MagicMock()
        llm.bind_tools.return_value.invoke.return_value = make_ai_response(
            content="The answer is 42."
        )

        result = run_tool_agent("What is the answer?", llm=llm)

        self.assertEqual(result, "The answer is 42.")

    @patch("agents.tool_agent._web_search")
    def test_executes_web_search_tool_call_then_returns_answer(self, mock_web_search):
        mock_web_search.return_value = [{"url": "http://x.com", "title": "X", "snippet": "stuff"}]

        tool_call = {"name": "web_search", "args": {"query": "test"}, "id": "call_1"}
        first_response = make_ai_response(tool_calls=[tool_call])
        final_response = make_ai_response(content="Here is what I found.")

        llm = MagicMock()
        llm.bind_tools.return_value.invoke.side_effect = [first_response, final_response]

        result = run_tool_agent("Search for test", llm=llm)

        self.assertEqual(result, "Here is what I found.")
        mock_web_search.assert_called_once_with("test", max_results=3)

    def test_returns_limit_message_when_max_iterations_reached(self):
        tool_call = {"name": "web_search", "args": {"query": "loop"}, "id": "call_x"}
        response = make_ai_response(tool_calls=[tool_call])

        llm = MagicMock()
        llm.bind_tools.return_value.invoke.return_value = response

        with patch("agents.tool_agent._web_search") as mock_ws:
            mock_ws.return_value = []
            result = run_tool_agent("Loop forever", llm=llm)

        self.assertIn("iteration limit", result)

    def test_unknown_tool_name_returns_error_message(self):
        tool_call = {"name": "nonexistent_tool", "args": {}, "id": "call_2"}
        first_response = make_ai_response(tool_calls=[tool_call])
        final_response = make_ai_response(content="Done.")

        llm = MagicMock()
        llm.bind_tools.return_value.invoke.side_effect = [first_response, final_response]

        result = run_tool_agent("Use unknown tool", llm=llm)
        self.assertEqual(result, "Done.")

    @patch("agents.tool_agent.load_vectorstore")
    @patch("agents.tool_agent.load_chunk_registry")
    @patch("agents.tool_agent.load_bm25_index")
    @patch("agents.tool_agent.retrieve_documents_with_query_transform")
    def test_query_documents_reloads_vectorstore_from_disk(
        self, mock_retrieve, mock_bm25, mock_registry, mock_vectorstore
    ):
        mock_retrieve.return_value = []
        tool_call = {"name": "query_documents", "args": {"question": "test"}, "id": "call_3"}
        first_response = make_ai_response(tool_calls=[tool_call])
        final_response = make_ai_response(content="Answer from docs.")

        llm = MagicMock()
        llm.bind_tools.return_value.invoke.side_effect = [first_response, final_response]

        run_tool_agent("test", llm=llm, embeddings=MagicMock(), reranker=MagicMock())

        mock_vectorstore.assert_called_once()
        mock_registry.assert_called_once()
        mock_bm25.assert_called_once()

    @patch("agents.tool_agent.ingest_saved_document")
    @patch("agents.tool_agent.save_url_import")
    @patch("agents.tool_agent.fetch_url_content")
    def test_fetch_url_embeds_after_fetching(self, mock_fetch, mock_save, mock_ingest):
        mock_fetch.return_value = {
            "page_text": "x" * 500,
            "source_url": "http://x.com",
            "title": "X",
            "domain": "x.com",
            "fetched_at": "2026-01-01T00:00:00+00:00",
            "file_name": "url-x-abc.txt",
        }
        # fetch_url now returns a confirmation message, not raw text
        mock_save.return_value = MagicMock()

        tool_call = {"name": "fetch_url", "args": {"url": "http://x.com"}, "id": "call_4"}
        first_response = make_ai_response(tool_calls=[tool_call])
        final_response = make_ai_response(content="Fetched and embedded.")

        llm = MagicMock()
        llm.bind_tools.return_value.invoke.side_effect = [first_response, final_response]

        run_tool_agent("fetch this", llm=llm, embeddings=MagicMock(), reranker=MagicMock())

        mock_fetch.assert_called_once_with("http://x.com")
        mock_save.assert_called_once()
        mock_ingest.assert_called_once()
