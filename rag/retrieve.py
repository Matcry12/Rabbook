from pathlib import Path
import json

from langchain_chroma import Chroma
from langchain_core.documents import Document

from core.config import DEFAULT_CONTEXT_WINDOW, DEFAULT_MAX_EXPANDED_CHUNKS, REGISTRY_PATH
from rag.prompt import build_rag_prompt


def load_vectorstore(persist_dir, embeddings):
    """
    Load a Chroma vector store from the given directory.
    """

    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def retrieve_documents(vectorstore, query, k=4):
    """
    Retrieve the top unique chunks for a query.
    """

    docs = vectorstore.similarity_search_with_score(query, k=k * 2)

    unique_docs = []
    seen_content = set()
    for doc, score in docs:
        content = doc.page_content.strip()
        if content in seen_content:
            continue
        unique_docs.append((doc, score))
        seen_content.add(content)
        if len(unique_docs) >= k:
            break

    return unique_docs


def load_chunk_registry(registry_path=REGISTRY_PATH):
    registry_file = Path(registry_path)
    if not registry_file.exists():
        return {"by_document": {}, "by_chunk_id": {}}

    return json.loads(registry_file.read_text(encoding="utf-8"))


def _document_from_record(record):
    if not record:
        return None

    return Document(
        page_content=record.get("page_content", ""),
        metadata=record.get("metadata", {}),
    )


def expand_with_context_window(
    documents,
    chunk_registry,
    window_size=DEFAULT_CONTEXT_WINDOW,
    max_expanded_chunks=DEFAULT_MAX_EXPANDED_CHUNKS,
):
    """
    Expand each retrieved chunk with neighbors from the same document.
    """

    if window_size <= 0:
        return documents

    # The registry gives us O(1)-style neighbor lookup by document and chunk index.
    # Chroma finds the relevant chunk; the registry finds the chunks around it.
    by_document = chunk_registry.get("by_document", {})
    expanded_documents = []
    seen_chunk_ids = set()

    for hit_order, (doc, score) in enumerate(documents):
        hit_group = _expand_single_hit(
            doc=doc,
            score=score,
            hit_order=hit_order,
            by_document=by_document,
            seen_chunk_ids=seen_chunk_ids,
            window_size=window_size,
        )
        expanded_documents.extend(hit_group)

        if len(expanded_documents) >= max_expanded_chunks:
            break

    # Final ordering preserves which retrieval hit came first, then keeps chunks
    # in document order inside that hit's local window.
    expanded_documents.sort(key=_hit_order_key)
    return expanded_documents[:max_expanded_chunks]


def _expand_single_hit(doc, score, hit_order, by_document, seen_chunk_ids, window_size):
    document_id = doc.metadata.get("document_id")
    chunk_index = doc.metadata.get("chunk_index")

    if document_id is None or chunk_index is None:
        return _include_unindexed_hit(doc, score, hit_order, seen_chunk_ids)

    document_chunks = by_document.get(document_id, {})
    center_index = int(chunk_index)
    hit_group = []

    # Build a local window around the matched chunk so the LLM sees nearby
    # supporting text instead of one isolated chunk.
    for neighbor_index in range(center_index - window_size, center_index + window_size + 1):
        neighbor = _load_neighbor_chunk(document_chunks, neighbor_index)
        if neighbor is None:
            continue

        neighbor_chunk_id = neighbor.metadata.get("chunk_id")
        if not neighbor_chunk_id or neighbor_chunk_id in seen_chunk_ids:
            continue

        _mark_window_metadata(
            neighbor=neighbor,
            original_doc=doc,
            hit_order=hit_order,
            center_index=center_index,
            neighbor_index=neighbor_index,
        )
        hit_group.append((neighbor, score))
        seen_chunk_ids.add(neighbor_chunk_id)

    # Keep each hit group ordered like the original document: previous, hit, next.
    hit_group.sort(key=_document_position_key)
    return hit_group


def _include_unindexed_hit(doc, score, hit_order, seen_chunk_ids):
    chunk_id = doc.metadata.get("chunk_id")
    if not chunk_id or chunk_id in seen_chunk_ids:
        return []

    _mark_window_metadata(
        neighbor=doc,
        original_doc=doc,
        hit_order=hit_order,
        center_index=0,
        neighbor_index=0,
    )
    seen_chunk_ids.add(chunk_id)
    return [(doc, score)]


def _load_neighbor_chunk(document_chunks, neighbor_index):
    neighbor_record = document_chunks.get(str(neighbor_index))
    if neighbor_record is None:
        return None
    return _document_from_record(neighbor_record)


def _mark_window_metadata(neighbor, original_doc, hit_order, center_index, neighbor_index):
    neighbor.metadata["is_retrieved_hit"] = (
        neighbor.metadata.get("chunk_id") == original_doc.metadata.get("chunk_id")
    )
    neighbor.metadata["hit_order"] = hit_order
    neighbor.metadata["window_offset"] = neighbor_index - center_index


def _document_position_key(item):
    doc, _ = item
    return (
        doc.metadata.get("document_id", ""),
        int(doc.metadata.get("chunk_index", -1)),
    )


def _hit_order_key(item):
    doc, _ = item
    return (
        int(doc.metadata.get("hit_order", 9999)),
        doc.metadata.get("document_id", ""),
        int(doc.metadata.get("chunk_index", -1)),
    )


def format_context(documents):
    """
    Format retrieved chunks into one context string.
    """

    parts = []
    for index, (doc, score) in enumerate(documents):
        file_name = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        document_id = doc.metadata.get("document_id", "unknown")
        page_label = page if page is not None else "n/a"
        is_retrieved_hit = doc.metadata.get("is_retrieved_hit", False)
        window_offset = doc.metadata.get("window_offset", 0)

        parts.append(
            (
                f"Chunk {index}\n"
                f"File: {file_name}\n"
                f"Document ID: {document_id}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Page: {page_label}\n"
                f"Matched Hit: {'yes' if is_retrieved_hit else 'no'}\n"
                f"Window Offset: {window_offset}\n"
                f"Score: {1 - score / 4}\n"
                f"Content: {doc.page_content}"
            )
        )
    return "\n\n".join(parts)


def generate_answer(query, context, llm):
    """
    Generate an answer from the retrieved context.
    """

    if not context.strip():
        return "No relevant information found in the documents."

    if llm is None:
        return "Language model is not available."

    response = llm.invoke(build_rag_prompt(context, query))
    return response.text.strip() if response else "No response from language model."
