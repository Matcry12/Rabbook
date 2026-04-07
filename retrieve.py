from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from prompt import build_rag_prompt
from langchain_core.globals import set_debug

load_dotenv()

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
llm = ChatOllama(model="gemma4:e2b", temperature=0.3, think = False)

def load_vectorstore(persist_dir, embeddings):
    vector_db = Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )
    return vector_db

def retrieve_documents(vectorstore, query, k=4):
    docs = vectorstore.similarity_search(query, k=k*2)

    unique_docs = []
    seen_sources = set()
    for doc in docs:
        if doc.page_content.strip() not in seen_sources:
            unique_docs.append(doc)
            seen_sources.add(doc.page_content.strip())
        if len(unique_docs) >= k:
            break

    return unique_docs

def format_context(documents):
    parts = []
    for i, doc in enumerate(documents):
        source = doc.metadata.get("source", "Unknown")
        parts.append(
            f"Chunk {i}\nSource: {source}\nContent: {doc.page_content}"
        )
    return "\n\n".join(parts)

def generate_answer(query, context, llm):
    if not context.strip():
        return "No relevant information found in the documents."
    
    prompt = build_rag_prompt(context, query)
    response = llm.invoke(prompt)
    return response.content

def ask_question(query, vectorstore, llm, k=4):
    documents = retrieve_documents(vectorstore, query, k=k)
    context = format_context(documents)
    answer = generate_answer(query, context, llm)
    return answer

def main():
    db_dir = "chroma_db/"

    #set_debug(True)
    
    print("Starting retrieve process...")
    vectorstore = load_vectorstore(db_dir, embeddings)
    while True:
        query = input("\nEnter your question (or 'exit' to quit): ")
        if query.lower() == "exit":
            break
        answer = ask_question(query, vectorstore, llm)
        print(f"\nAnswer:\n{answer}")
        print("Top 4 relevant documents:")
        for i, doc in enumerate(retrieve_documents(vectorstore, query, k=4)):
            source = doc.metadata.get("source", "Unknown")
            print(f"Document {i+1} - Source: {source}\nScore: {doc.metadata.get('score', 'Unknown')}\nContent: {doc.page_content}\n")

if __name__ == "__main__":
    main()