import mimetypes
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)

def load_documents(data_dir):
    documents = []
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

def split_documents(documents, separators=None, chunk_size=1000, chunk_overlap=100):
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
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir
    )
    return vector_db
def main():
    data_dir = "data/"
    db_dir = "chroma_db/"

    print("Starting ingestion process...")

    docs = load_documents(data_dir)
    chunks = split_documents(docs)
    build_vectorstore(chunks, embeddings, db_dir)

    print(f"Loaded {len(docs)} documents")
    print(f"Created {len(chunks)} chunks")
    print(f"Saved vector DB to {db_dir}")

    print("Ingestion completed.")

if __name__ == "__main__":
    main()