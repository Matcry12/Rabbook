import unittest

from rag.web_ingest import build_research_import_payload


class WebIngestTests(unittest.TestCase):
    def test_build_research_import_payload_uses_content_and_builds_txt_filename(self):
        result = {
            "url": "https://example.com/page",
            "title": "Example Page",
            "snippet": "Short snippet",
            "content": "Full fetched content",
        }

        payload = build_research_import_payload(result)

        self.assertEqual(payload["source_url"], "https://example.com/page")
        self.assertEqual(payload["title"], "Example Page")
        self.assertEqual(payload["page_text"], "Full fetched content")
        self.assertEqual(payload["domain"], "example.com")
        self.assertTrue(payload["file_name"].startswith("research-example-page-"))
        self.assertTrue(payload["file_name"].endswith(".txt"))


if __name__ == "__main__":
    unittest.main()
