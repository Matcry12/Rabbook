import unittest

from rag.retrieve import AnswerDraftResult, generate_answer


class FakeResponse:
    def __init__(self, content):
        self.content = content


class FakeStructuredRunnable:
    def __init__(self, structured_response):
        self.structured_response = structured_response

    def invoke(self, prompt):
        return self.structured_response


class FakeStructuredLLM:
    def __init__(self, structured_response):
        self.structured_response = structured_response
        self.prompts = []

    def with_structured_output(self, schema):
        self.schema = schema
        return FakeStructuredRunnable(self.structured_response)

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeResponse("")


class AnswerStructuredOutputTests(unittest.TestCase):
    def test_generate_answer_uses_structured_output_when_available(self):
        llm = FakeStructuredLLM(AnswerDraftResult(answer="Structured answer [1]"))
        context = "Source [1] climate note"

        result = generate_answer("What is climate change?", context, llm)

        self.assertEqual(result, "Structured answer [1]")
        self.assertIs(llm.schema, AnswerDraftResult)

    def test_generate_answer_falls_back_to_plain_text_when_structured_output_is_missing(self):
        class PlainLLM:
            def __init__(self):
                self.prompts = []

            def invoke(self, prompt):
                self.prompts.append(prompt)
                return FakeResponse("Plain answer [1]")

        llm = PlainLLM()
        context = "Source [1] climate note"

        result = generate_answer("What is climate change?", context, llm)

        self.assertEqual(result, "Plain answer [1]")


if __name__ == "__main__":
    unittest.main()
