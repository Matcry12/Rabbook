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
    def test_executes_tool_call_then_returns_answer(self, mock_web_search):
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

        with patch("agents.tool_agent.web_search") as mock_ws:
            mock_ws.invoke.return_value = "some result"
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
