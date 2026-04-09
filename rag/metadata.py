import hashlib
from pathlib import Path


def build_document_id(source: str) -> str:
    """
    Build a stable identifier for a source document.
    """

    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def enrich_document_metadata(documents):
    """
    Normalize file-level metadata before splitting into chunks.
    """

    enriched = []

    for doc in documents:
        source = str(doc.metadata.get("source", "")).strip()
        path = Path(source) if source else None
        page = doc.metadata.get("page")

        doc.metadata["source"] = source or "unknown"
        doc.metadata["file_name"] = path.name if path else "unknown"
        doc.metadata["file_type"] = path.suffix.lower().lstrip(".") if path else "unknown"
        doc.metadata["document_id"] = build_document_id(source or doc.metadata["file_name"])
        doc.metadata["page"] = page if page is not None else None

        enriched.append(doc)

    return enriched


def enrich_chunk_metadata(chunks):
    """
    Add chunk-level metadata after splitting so each chunk is traceable.
    """

    chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata.get("document_id", "unknown")
        chunk_index = chunk_counts.get(document_id, 0)
        chunk_counts[document_id] = chunk_index + 1

        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = f"{document_id}-chunk-{chunk_index}"

    return chunks
