from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from prompt import build_rag_prompt, rewrite_query
from langchain_google_genai import ChatGoogleGenerativeAI
from config import DEFAULT_LLM_MODEL, get_google_api_key

def load_vectorstore(persist_dir, embeddings):

    """
    Load a Chroma vector store from the specified directory using the provided embeddings.
     - persist_dir: The directory where the Chroma vector store is saved.
     - embeddings: An instance of HuggingFaceEmbeddings to generate vector representations of the chunks when loading the vector store.
    """

    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vector_db

def retrieve_documents(vectorstore, query, k=4):

    """
    Retrieve the most relevant documents from the vector store based on the query.
     - vectorstore: The Chroma vector store to search.
     - query: The query string.
     - k: The number of top documents to retrieve.
    """

    docs = vectorstore.similarity_search_with_score(query, k=k*2)

    unique_docs = []
    seen_sources = set()
    for doc, score in docs:
        if doc.page_content.strip() not in seen_sources:
            unique_docs.append((doc, score))
            seen_sources.add(doc.page_content.strip())
        if len(unique_docs) >= k:
            break

    return unique_docs

def get_chunk_by_id(vectorstore, chunk_id):

    """
    Retrieve a specific document chunk from the vector store using its unique chunk ID.
     - vectorstore: The Chroma vector store to search.
     - chunk_id: The unique identifier of the document chunk to retrieve.
     - Returns: The document chunk object if found, or None if no matching chunk is found in the vector store.
    """

    all_docs = vectorstore.get(include=["documents", "metadatas"])
    documents = all_docs.get("documents", [])
    metadatas = all_docs.get("metadatas", [])

    for page_content, metadata in zip(documents, metadatas):
        if metadata.get("chunk_id") == chunk_id:
            return {"page_content": page_content, "metadata": metadata}

    return None

def format_context(documents):

    """
    Format the retrieved documents into a single context string for the RAG prompt.
     - documents: A list of document objects retrieved from the vector store.
    """
    parts = []
    for i, (doc, score) in enumerate(documents):
        file_name = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        document_id = doc.metadata.get("document_id", "unknown")
        page_label = page if page is not None else "n/a"

        parts.append(
            (
                f"Chunk {i}\n"
                f"File: {file_name}\n"
                f"Document ID: {document_id}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Page: {page_label}\n"
                f"Score: {1 - score/4}\n"
                f"Content: {doc.page_content}"
            )
        )
    return "\n\n".join(parts)

def query_trasnform(query, rewriter):
    """ Using sub query rewriting to improve retrieval performance. --- IGNORE ---"""
    rewrite_query_prompt = rewrite_query(query)
    response = rewriter.invoke(rewrite_query_prompt).text.strip()
    sub_queries = [q.strip() for q in response.split('\n') if q.strip() and not q.strip().startswith('Sub-queries:')]
    
    return sub_queries

def generate_answer(query, context, llm):

    """
    Generate an answer to the query using the provided context and language model.
     - query: The original question or query string.
     - context: The formatted context string containing relevant information from the retrieved documents.
     - llm: An instance of a language model (e.g., ChatOllama) to generate the answer based on the prompt.
     - Returns: The generated answer as a string.
     - If the context is empty or does not contain relevant information, returns a message indicating that no relevant information was found in the documents.
     - The function constructs a RAG prompt using the build_rag_prompt function, invokes the language model with the prompt, and returns the generated answer content
    """

    if not context.strip():
        return "No relevant information found in the documents."
    
    prompt = build_rag_prompt(context, query)

    if llm is None:
        return "Language model is not available."
        
    response = llm.invoke(prompt)
    return response.text.strip() if response else "No response from language model."

def ask_question(query, vectorstore, llm, k=4):

    """
    Ask a question and retrieve an answer using the RAG system.
     - query: The question or query string.
     - vectorstore: The Chroma vector store to search.
     - llm: An instance of a language model (e.g., ChatOllama) to generate the answer based on the prompt.
     - k: The number of top documents to retrieve.
    """

    documents = retrieve_documents(vectorstore, query, k=k)
    context = format_context(documents)
    answer = generate_answer(query, context, llm)
    return answer

def main():
    load_dotenv()

    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    rewriter = ChatGoogleGenerativeAI(
        model=DEFAULT_LLM_MODEL,
        google_api_key= get_google_api_key(),
        temperature=0.3,
    )
    chroma_dir = "chroma_db"
    # Load the vector store
    vectorstore = load_vectorstore(chroma_dir, embeddings=embeddings)

    query = "What is RAG?"

    sub_queries = query_trasnform(query, rewriter)
    documents = []
    for sub_query in sub_queries:
        print(f"Sub-query: {sub_query}")
        docs = retrieve_documents(vectorstore, sub_query)
        documents.extend(docs)

    for doc, score in documents:
        print(f"Metadata: {doc.metadata}")
        print(f"Score: {1 - score/4}")
        #print(f"Content: {doc.page_content}\n")

    print("\n\n---\n\n")
    print("Get chunk by id:")
    chunk_id = "2"  # Replace with the actual chunk ID you want to retrieve
    chunk = get_chunk_by_id(vectorstore, chunk_id)
    if chunk:
        print(f"Found chunk: {chunk.page_content}")
    else:
        print("Chunk not found.")

if __name__ == "__main__":
    main()
