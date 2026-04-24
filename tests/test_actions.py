import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from app.actions import ingest_saved_document


class ActionsTests(unittest.TestCase):
    def test_ingest_saved_document_uses_shared_saved_file_path(self):
        calls = []

        def fake_add_documents_to_vectorstore(path, embeddings, persist_dir):
            calls.append((path, embeddings, persist_dir))

        with TemporaryDirectory() as tmp_dir:
            target_path = Path(tmp_dir) / "sample.txt"
            target_path.write_text("sample", encoding="utf-8")

            ingest_saved_document(
                target_path,
                add_documents_to_vectorstore=fake_add_documents_to_vectorstore,
                embeddings="embeddings-object",
            )

        self.assertEqual(len(calls), 1)
        self.assertEqual(calls[0][0], str(target_path))
        self.assertEqual(calls[0][1], "embeddings-object")


if __name__ == "__main__":
    unittest.main()
