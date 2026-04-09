from langchain_core.documents import Document
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter

from core.config import (
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_PERCENTILE,
)

DEFAULT_SEPARATORS = ["\n\n", "\n", " ", ""]


def split_documents(
    documents,
    embeddings,
    separators=None,
    chunk_size=DEFAULT_CHUNK_SIZE,
    chunk_overlap=DEFAULT_CHUNK_OVERLAP,
    percentile=DEFAULT_SEMANTIC_PERCENTILE,
):
    """
    Split documents with SemanticChunker using percentile breakpoints.
    Oversized semantic chunks still fall back to recursive splitting.
    """

    if separators is None:
        separators = DEFAULT_SEPARATORS

    chunker = SemanticChunker(
        embeddings=embeddings,
        breakpoint_threshold_type="percentile",
        breakpoint_threshold_amount=percentile,
    )
    fallback_splitter = RecursiveCharacterTextSplitter(
        separators=separators,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    chunks: list[Document] = []
    for document in documents:
        chunks.extend(
            split_document_semantically(
                document=document,
                chunker=chunker,
                fallback_splitter=fallback_splitter,
                chunk_size=chunk_size,
            )
        )

    return chunks


def split_document_semantically(document, chunker, fallback_splitter, chunk_size=DEFAULT_CHUNK_SIZE):
    """
    Use embedding-based semantic chunking first, then keep a size guardrail.
    """

    text = document.page_content.strip()
    if not text:
        return []

    semantic_chunks = chunker.create_documents(
        texts=[text],
        metadatas=[dict(document.metadata)],
    )

    chunks: list[Document] = []
    for chunk in semantic_chunks:
        clean_text = chunk.page_content.strip()
        if not clean_text:
            continue

        # SemanticChunker may still return a large block, so we keep a size cap
        # before storing chunks in Chroma.
        if len(clean_text) > chunk_size:
            chunks.extend(
                fallback_splitter.create_documents(
                    texts=[clean_text],
                    metadatas=[dict(document.metadata)],
                )
            )
            continue

        chunks.append(
            Document(
                page_content=clean_text,
                metadata=dict(document.metadata),
            )
        )

    return chunks
