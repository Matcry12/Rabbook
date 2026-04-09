from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import (
    DATA_DIR,
    DB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_PERCENTILE,
)
from rag.chunking import split_documents
from rag.loaders import load_documents
from rag.metadata import enrich_chunk_metadata, enrich_document_metadata
from rag.registry import update_chunk_registry


def build_vectorstore(chunks, embeddings, persist_dir):
    """
    Build a Chroma vector store from the provided document chunks and embeddings.
    """

    return Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )


def prepare_chunks(documents, embeddings):
    """
    Run the ingestion preparation pipeline before writing to Chroma.
    """

    documents = enrich_document_metadata(documents)
    chunks = split_documents(
        documents,
        embeddings=embeddings,
        chunk_size=DEFAULT_CHUNK_SIZE,
        chunk_overlap=DEFAULT_CHUNK_OVERLAP,
        percentile=DEFAULT_SEMANTIC_PERCENTILE,
    )
    return enrich_chunk_metadata(chunks)


def add_documents_to_vectorstore(data_dir, embeddings, persist_dir):
    """
    Add new documents into the existing Chroma vector store.
    """

    documents = load_documents(data_dir)
    enriched_chunks = prepare_chunks(documents, embeddings)

    if not enriched_chunks:
        raise ValueError("No supported documents were found to ingest.")

    persist_path = Path(persist_dir)
    
    if persist_path.exists() and any(persist_path.iterdir()):
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        vector_db.add_documents(enriched_chunks)
        update_chunk_registry(enriched_chunks)
        return vector_db

    vector_db = build_vectorstore(enriched_chunks, embeddings, str(persist_path))
    update_chunk_registry(enriched_chunks)
    return vector_db


def main():
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    print("Starting ingestion process...")
    documents = load_documents(DATA_DIR)
    enriched_chunks = prepare_chunks(documents, embeddings)
    build_vectorstore(enriched_chunks, embeddings, str(DB_DIR))
    update_chunk_registry(enriched_chunks)

    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(enriched_chunks)} chunks")
    print(f"Saved vector DB to {DB_DIR}")
    print("Ingestion completed.")


if __name__ == "__main__":
    main()
