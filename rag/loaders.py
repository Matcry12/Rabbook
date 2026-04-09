import mimetypes
import os

from langchain_community.document_loaders import PyPDFLoader, TextLoader


def load_documents(data_dir):
    """
    Load supported documents from a file path or directory.
    """

    documents = []
    data_dir = str(data_dir)

    if data_dir.endswith(".txt") or data_dir.endswith(".pdf"):
        loader = _build_loader(data_dir)
        return loader.load() if loader else []

    if os.path.isdir(data_dir):
        for root, _, files in os.walk(data_dir):
            for file_name in files:
                file_path = os.path.join(root, file_name)
                loader = _build_loader(file_path)
                if loader is None:
                    continue
                documents.extend(loader.load())
        return documents

    print(f"Invalid path: {data_dir}")
    return []


def _build_loader(file_path):
    data_type = mimetypes.guess_type(file_path)[0]
    if data_type == "text/plain":
        return TextLoader(file_path)
    if data_type == "application/pdf":
        return PyPDFLoader(file_path)

    print(f"Unsupported file type: {file_path}")
    return None
