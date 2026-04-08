import mimetypes
import os
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import DATA_DIR, DB_DIR

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
    chunks = split_documents(docs)

    if not chunks:
        raise ValueError("No supported documents were found to ingest.")

    persist_path = Path(persist_dir)
    if persist_path.exists() and any(persist_path.iterdir()):
        vector_db = Chroma(
            embedding_function=embeddings,
            persist_directory=str(persist_path),
        )
        vector_db.add_documents(chunks)
        return vector_db

    vector_db = build_vectorstore(chunks, embeddings, str(persist_path))

    return vector_db

def main():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
    )

    data_dir = DATA_DIR
    db_dir = DB_DIR

    print("Starting ingestion process...")

    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    build_vectorstore(chunks, embeddings, str(db_dir))

    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved vector DB to {db_dir}")

    print("Ingestion completed.")

if __name__ == "__main__":
    main()
