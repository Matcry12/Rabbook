from retrieve import load_vectorstore, ask_question
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from dotenv import load_dotenv
from langchain_core.globals import set_debug
from ingest import add_documents_to_vectorstore

load_dotenv()
set_debug(True)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
)
llm = ChatOllama(model="gemma4:e2b", temperature=0.3, think = False)

def setup_app():
    
    chroma_db_dir = "chroma_db/"
    vectorstore = load_vectorstore(chroma_db_dir, embeddings)
    return vectorstore


def get_user_input():
    query = input("Enter your question: ")
    return query

def validate_query(query):
    if not query.strip():
        print("Please enter a valid question.")
        return False
    if len(query) > 500:
        print("Question is too long. Please limit it to 500 characters.")
        return False
    return True


def run_rag(query, vectorstore, llm):
    answer = ask_question(query, vectorstore, llm)
    return answer

def display_answer(answer):
    print(f"Answer: {answer}")

def run_app():
    
    vectorstore = setup_app()
    while True:
        query = get_user_input()
        if not validate_query(query):
            continue
        if query.lower() == "exit":
            break
        if query.lower() == "help":
            print("You can ask any question related to the documents in the vector store. Type 'exit' to quit.")
            continue
        if query.lower() == "add document":
            doc_path = input("Enter the path to the document or folder you want to add: ")
            
            add_documents_to_vectorstore(doc_path, embeddings, "chroma_db/")  # This is a placeholder. You would need to implement the actual document loading and vector store updating logic.

            print(f"Document at {doc_path} added to the vector store.")
            continue

        answer = run_rag(query, vectorstore, llm)
        display_answer(answer)

def main():
    run_app()

if __name__ == "__main__":
    main()