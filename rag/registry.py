import json
from pathlib import Path

from core.config import REGISTRY_PATH


def load_chunk_registry(registry_path=REGISTRY_PATH):
    registry_file = Path(registry_path)
    if not registry_file.exists():
        return {"by_document": {}, "by_chunk_id": {}}

    return json.loads(registry_file.read_text(encoding="utf-8"))


def update_chunk_registry(chunks, registry_path=REGISTRY_PATH):
    """
    Persist a lightweight chunk registry for fast neighbor lookup at query time.
    """

    registry = load_chunk_registry(registry_path)
    by_document = registry.setdefault("by_document", {})
    by_chunk_id = registry.setdefault("by_chunk_id", {})

    for chunk in chunks:
        metadata = dict(chunk.metadata)
        document_id = metadata.get("document_id")
        chunk_id = metadata.get("chunk_id")
        chunk_index = metadata.get("chunk_index")

        if document_id is None or chunk_id is None or chunk_index is None:
            continue

        chunk_record = {
            "page_content": chunk.page_content,
            "metadata": metadata,
        }

        by_document.setdefault(document_id, {})[str(int(chunk_index))] = chunk_record
        by_chunk_id[chunk_id] = chunk_record

    Path(registry_path).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def list_documents(registry_path=REGISTRY_PATH):
    registry = load_chunk_registry(registry_path)
    documents = {}

    for document_id, chunks in registry.get("by_document", {}).items():
        chunk_records = list(chunks.values())
        if not chunk_records:
            continue

        first_metadata = chunk_records[0].get("metadata", {})
        pages = [
            chunk_record.get("metadata", {}).get("page")
            for chunk_record in chunk_records
            if chunk_record.get("metadata", {}).get("page") is not None
        ]
        documents[document_id] = {
            "document_id": document_id,
            "file_name": first_metadata.get("file_name", "Unknown"),
            "file_type": first_metadata.get("file_type", "unknown"),
            "chunk_count": len(chunk_records),
            "page_count": (max(int(page) for page in pages) + 1) if pages else None,
            "source": first_metadata.get("source", ""),
        }

    return sorted(documents.values(), key=lambda item: item["file_name"].lower())


def delete_document_from_registry(document_id, registry_path=REGISTRY_PATH):
    registry = load_chunk_registry(registry_path)
    by_document = registry.get("by_document", {})
    by_chunk_id = registry.get("by_chunk_id", {})

    removed_document = by_document.pop(document_id, None)
    if not removed_document:
        return False

    for chunk_record in removed_document.values():
        chunk_id = chunk_record.get("metadata", {}).get("chunk_id")
        if not chunk_id:
            continue
        by_chunk_id.pop(chunk_id, None)

    Path(registry_path).write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return True
