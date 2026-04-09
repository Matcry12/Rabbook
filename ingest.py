import hashlib
import json
import mimetypes
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, DB_DIR, REGISTRY_PATH


def build_document_id(source: str) -> str:
    """
    Build a stable identifier for a source document.
    The same source path will always map to the same document ID.
    """

    return hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]


def load_documents(data_dir):
    """
    Load documents from the specified directory, supporting text and PDF files.
    """
    documents = []

    data_dir = str(data_dir)
    if data_dir.endswith(".txt") or data_dir.endswith(".pdf"):
        data_type = mimetypes.guess_type(data_dir)[0]
        if data_type == "text/plain":
            loader = TextLoader(data_dir)
        elif data_type == "application/pdf":
            loader = PyPDFLoader(data_dir)
        else:
            print(f"Unsupported file type: {data_dir}")
            return []
        return loader.load()
    
    elif os.path.isdir(data_dir):
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = os.path.join(root, file)
                data_type = mimetypes.guess_type(file_path)[0]
                if data_type == "text/plain":
                    loader = TextLoader(file_path)
                elif data_type == "application/pdf":
                    loader = PyPDFLoader(file_path)
                else:
                    print(f"Unsupported file type: {file_path}")
                    continue
                documents.extend(loader.load())
        return documents
    else:
        print(f"Invalid path: {data_dir}")
        return []


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


def split_documents(documents, separators=None, chunk_size=1200, chunk_overlap=150):

    """
    Split documents into smaller chunks using the RecursiveCharacterTextSplitter.
     - separators: List of separators to use for splitting. If None, defaults to ["\n\n", "\n", " ", ""]
     - chunk_size: The maximum size of each chunk in characters.
     - chunk_overlap: The number of characters to overlap between chunks to maintain context
    """

    if separators is None:
        separators = ["\n\n", "\n", " ", ""]

    text_splitter = RecursiveCharacterTextSplitter(
        separators=separators, 
        chunk_size=chunk_size, 
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks


def enrich_chunk_metadata(chunks):
    """
    Add chunk-level metadata after splitting so each chunk is traceable.
    Chunk indexes are grouped per document, not globally.
    """

    chunk_counts: dict[str, int] = {}

    for chunk in chunks:
        document_id = chunk.metadata.get("document_id", "unknown")
        chunk_index = chunk_counts.get(document_id, 0)
        chunk_counts[document_id] = chunk_index + 1

        chunk.metadata["chunk_index"] = chunk_index
        chunk.metadata["chunk_id"] = f"{document_id}-chunk-{chunk_index}"

    return chunks


def update_chunk_registry(chunks, registry_path=REGISTRY_PATH):
    """
    Persist a lightweight chunk registry for fast neighbor lookup at query time.
    This avoids scanning the full vector store on every request.
    """

    registry_file = Path(registry_path)
    if registry_file.exists():
        registry = json.loads(registry_file.read_text(encoding="utf-8"))
    else:
        registry = {"by_document": {}, "by_chunk_id": {}}

    by_document = registry.setdefault("by_document", {})
    by_chunk_id = registry.setdefault("by_chunk_id", {})

    for chunk in chunks:
        metadata = dict(chunk.metadata)
        document_id = metadata.get("document_id")
        chunk_id = metadata.get("chunk_id")
        chunk_index = metadata.get("chunk_index")

        if document_id is None or chunk_id is None or chunk_index is None:
            continue

        chunk_index_str = str(int(chunk_index))
        chunk_record = {
            "page_content": chunk.page_content,
            "metadata": metadata,
        }

        document_entry = by_document.setdefault(document_id, {})
        document_entry[chunk_index_str] = chunk_record
        by_chunk_id[chunk_id] = chunk_record

    registry_file.write_text(
        json.dumps(registry, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

def build_vectorstore(chunks, embeddings, persist_dir):

    """
    Build a Chroma vector store from the provided document chunks and embeddings, and persist it to the specified directory.
     - chunks: List of document chunks to be embedded and stored.
     - embeddings: An instance of HuggingFaceEmbeddings to generate vector representations of the chunks.
     - persist_dir: The directory where the Chroma vector store will be saved.
    """

    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )
    return vector_db

def add_documents_to_vectorstore(data_dir, embeddings, persist_dir):
    """
    Add new document chunks to an existing Chroma vector store by embedding the new chunks and persisting the updated vector store.
     - data_dir: The directory containing the new documents to be added.
     - embeddings: An instance of HuggingFaceEmbeddings to generate vector representations of the new chunks.
     - persist_dir: The directory where the existing Chroma vector store is saved and where the updated vector store will be persisted.
    """

    docs = load_documents(data_dir)
    enriched_docs = enrich_document_metadata(docs)
    chunks = split_documents(enriched_docs)
    enriched_chunks = enrich_chunk_metadata(chunks)

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
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    data_dir = DATA_DIR
    db_dir = DB_DIR

    print("Starting ingestion process...")

    docs = load_documents(data_dir)
    enriched_docs = enrich_document_metadata(docs)
    chunks = split_documents(enriched_docs)
    enriched_chunks = enrich_chunk_metadata(chunks)
    build_vectorstore(enriched_chunks, embeddings, str(db_dir))
    update_chunk_registry(enriched_chunks)

    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved vector DB to {db_dir}")

    print("Ingestion completed.")

if __name__ == "__main__":
    main()
