import unittest

from rag.retrieve import (
    QueryRewriteResult,
    generate_sub_queries,
    is_valid_retrieval_query,
    parse_structured_sub_queries,
    parse_sub_queries_json,
)


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeTransformer:
    def __init__(self, content):
        self.content = content

    def invoke(self, prompt):
        return FakeResponse(self.content)


class FakeStructuredRunnable:
    def __init__(self, structured_response):
        self.structured_response = structured_response

    def invoke(self, prompt):
        return self.structured_response


class FakeStructuredTransformer:
    def __init__(self, structured_response):
        self.structured_response = structured_response
        self.prompts = []

    def with_structured_output(self, schema, **kwargs):
        self.schema = schema
        return FakeStructuredRunnable(self.structured_response)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse("")


class QueryTransformTests(unittest.TestCase):
    def test_parse_sub_queries_json_returns_clean_query_list(self):
        response_text = """
Here is the retrieval output:
{"sub_queries": ["trump background", "trump family", "trump business"]}
"""

        result = parse_sub_queries_json(
            response_text,
            "5 facts about trump",
            max_queries=4,
        )

        self.assertEqual(
            result,
            ["trump background", "trump family", "trump business"],
        )

    def test_generate_sub_queries_prefers_json_over_explanatory_text(self):
        transformer = FakeTransformer(
            """I'll break down the original query.
{"sub_queries": ["climate change biodiversity", "climate change oceans", "climate change agriculture"]}
"""
        )

        result = generate_sub_queries(
            "What are the impacts of climate change on the environment?",
            transformer,
            max_queries=3,
        )

        self.assertEqual(
            result,
            [
                "climate change biodiversity",
                "climate change oceans",
                "climate change agriculture",
            ],
        )

    def test_generate_sub_queries_uses_structured_output_when_available(self):
        transformer = FakeStructuredTransformer(
            QueryRewriteResult(
                sub_queries=[
                    "trump background",
                    "trump family",
                    "trump business",
                ]
            )
        )

        result = generate_sub_queries(
            "5 facts about trump",
            transformer,
            max_queries=3,
        )

        self.assertEqual(
            result,
            ["trump background", "trump family", "trump business"],
        )
        self.assertIs(transformer.schema, QueryRewriteResult)

    def test_parse_structured_sub_queries_handles_plain_dict_payload(self):
        result = parse_structured_sub_queries(
            {"sub_queries": ["trump background", "trump family", "trump business"]},
            "5 facts about trump",
            max_queries=3,
        )

        self.assertEqual(
            result,
            ["trump background", "trump family", "trump business"],
        )

    def test_generate_sub_queries_falls_back_to_line_parsing_when_json_is_missing(self):
        transformer = FakeTransformer(
            """Sub-queries:
1. trump background
2. trump family
3. trump business
"""
        )

        result = generate_sub_queries(
            "5 facts about trump",
            transformer,
            max_queries=3,
        )

        self.assertEqual(
            result,
            ["trump background", "trump family", "trump business"],
        )

    def test_rejects_assistant_style_refusal_text_as_retrieval_query(self):
        self.assertFalse(
            is_valid_retrieval_query(
                'Please provide a specific question or topic you would like me to research.',
                original_query="hi",
            )
        )


if __name__ == "__main__":
    unittest.main()
