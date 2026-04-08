from langchain_chroma import Chroma

from prompt import build_rag_prompt

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

def format_context(documents):

    """
    Format the retrieved documents into a single context string for the RAG prompt.
     - documents: A list of document objects retrieved from the vector store.
    """

    parts = []
    for i, (doc, score) in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        parts.append(
            f"Chunk {i}\nSource: {source}\nScore: {1 - score/4}\nContent: {doc.page_content}"
        )
    return "\n\n".join(parts)

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
