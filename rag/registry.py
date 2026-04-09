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
