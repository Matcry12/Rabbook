from pathlib import Path

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from core.config import (
    DATA_DIR,
    DB_DIR,
    DEFAULT_CHUNK_OVERLAP,
    DEFAULT_CHUNK_SIZE,
    DEFAULT_SEMANTIC_PERCENTILE,
    REGISTRY_PATH,
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
    return add_loaded_documents_to_vectorstore(documents, embeddings, persist_dir)


def add_loaded_documents_to_vectorstore(documents, embeddings, persist_dir):
    """
    Add already-loaded documents into the existing Chroma vector store.
    """

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


def reingest_directory(data_dir, embeddings, persist_dir, registry_path=REGISTRY_PATH):
    """
    Rebuild Chroma and the chunk registry from a directory of uploaded files.
    """

    documents = load_documents(data_dir)
    enriched_chunks = prepare_chunks(documents, embeddings)

    persist_path = Path(persist_dir)
    persist_path.mkdir(parents=True, exist_ok=True)

    vector_db = None
    if any(persist_path.iterdir()):
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        existing = vector_db._collection.get()
        existing_ids = existing.get("ids", [])
        if existing_ids:
            vector_db._collection.delete(ids=existing_ids)

    registry_file = Path(registry_path)
    registry_file.write_text(
        '{"by_document": {}, "by_chunk_id": {}}',
        encoding="utf-8",
    )

    if not enriched_chunks:
        return {
            "document_count": len(documents),
            "chunk_count": 0,
        }

    if vector_db is not None:
        vector_db.add_documents(enriched_chunks)
    else:
        build_vectorstore(enriched_chunks, embeddings, str(persist_path))
    update_chunk_registry(enriched_chunks, registry_path=registry_path)
    return {
        "document_count": len(documents),
        "chunk_count": len(enriched_chunks),
    }


def main():

    load_dotenv()  # Load environment variables from .env file

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
    raise SystemExit(
        "Run this from the project root with `python ingest_docs.py`, not `python rag/ingest.py`."
    )
