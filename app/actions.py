import json
from pathlib import Path

from core.config import DB_DIR, REGISTRY_PATH, SUPPORTED_EXTENSIONS, UPLOAD_DIR, URL_IMPORT_DIR
from rag.history import save_history_entry
from rag.registry import delete_document_from_registry
from rag.web_ingest import fetch_url_content, save_url_import


def get_upload_target(document):
    if not document.filename:
        raise ValueError("Please choose a file to upload.")

    filename = Path(document.filename).name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")

    return filename, UPLOAD_DIR / filename


async def save_uploaded_document(document, target_path):
    target_path.write_bytes(await document.read())
    await document.close()


def ingest_uploaded_document(target_path, *, add_documents_to_vectorstore, get_embeddings, refresh_runtime_state):
    add_documents_to_vectorstore(str(target_path), get_embeddings(), str(DB_DIR))
    refresh_runtime_state()


def ingest_url_document(
    url,
    *,
    add_documents_to_vectorstore,
    get_embeddings,
    refresh_runtime_state,
):
    payload = fetch_url_content(url)
    saved_path = save_url_import(payload, URL_IMPORT_DIR)
    ingest_uploaded_document(
        saved_path,
        add_documents_to_vectorstore=add_documents_to_vectorstore,
        get_embeddings=get_embeddings,
        refresh_runtime_state=refresh_runtime_state,
    )
    return payload


def parse_saved_citations(citations_json):
    if not citations_json:
        return []

    citations = json.loads(citations_json)
    return citations if isinstance(citations, list) else []


def save_history_item(
    query,
    answer,
    citations,
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
):
    return save_history_entry(
        query=query,
        answer=answer,
        citations=citations,
        selected_file=selected_file,
        selected_file_type=selected_file_type,
        page_start=page_start,
        page_end=page_end,
    )


def delete_document(
    document_id,
    *,
    get_library_documents,
    get_vectorstore,
    refresh_runtime_state,
):
    document = next(
        (item for item in get_library_documents() if item["document_id"] == document_id),
        None,
    )
    if document is None:
        raise ValueError("Document not found.")

    get_vectorstore()._collection.delete(where={"document_id": document_id})
    delete_document_from_registry(document_id, str(REGISTRY_PATH))

    source_path = Path(document.get("source", ""))
    if source_path.exists() and source_path.parent == UPLOAD_DIR:
        source_path.unlink()

    refresh_runtime_state()
    return document
