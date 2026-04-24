import unittest

from app.web import GemmaPromptWrapper


class FakeRunnable:
    def __init__(self):
        self.last_input = None

    def invoke(self, input_value, *args, **kwargs):
        self.last_input = input_value
        return {"ok": True}


class FakeLLM:
    def __init__(self):
        self.last_input = None
        self.structured_runnable = FakeRunnable()

    def invoke(self, input_value, *args, **kwargs):
        self.last_input = input_value
        return {"ok": True}

    def with_structured_output(self, *args, **kwargs):
        return self.structured_runnable


class GemmaPromptWrapperTests(unittest.TestCase):
    def test_prepends_thought_off_for_gemma_string_prompts(self):
        llm = FakeLLM()
        wrapped = GemmaPromptWrapper(llm, "gemma-3-27b-it")

        wrapped.invoke("hello world")

        self.assertEqual(llm.last_input, "<thought off>\nhello world")

    def test_does_not_duplicate_thought_off(self):
        llm = FakeLLM()
        wrapped = GemmaPromptWrapper(llm, "gemma-3-27b-it")

        wrapped.invoke("<thought off>\nhello world")

        self.assertEqual(llm.last_input, "<thought off>\nhello world")

    def test_leaves_non_gemma_models_unchanged(self):
        llm = FakeLLM()
        wrapped = GemmaPromptWrapper(llm, "llama-3.1-8b-instant")

        wrapped.invoke("hello world")

        self.assertEqual(llm.last_input, "hello world")

    def test_wraps_structured_output_prompts_too(self):
        llm = FakeLLM()
        wrapped = GemmaPromptWrapper(llm, "gemma-3-27b-it")

        runnable = wrapped.with_structured_output(object)
        runnable.invoke("structured prompt")

        self.assertEqual(llm.structured_runnable.last_input, "<thought off>\nstructured prompt")


if __name__ == "__main__":
    unittest.main()
