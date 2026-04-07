from retrieve import load_vectorstore, retrieve_documents, format_context
import json

def test_retrieve(db_dir="chroma_db/", query=None, json_dir=None):

    if json_dir:
        with open(json_dir, 'r') as f:
            questions = json.load(f)
        json_output = {"results": []}
        for query in questions:
            question = query.get("question", "")
            #print(f"Question: {question}")
            vectorstore = load_vectorstore(db_dir, None)
            retrieved_docs = retrieve_documents(vectorstore, question, k=4)
            json_output["results"].append({
                "question": question,
                "retrieved_docs": [
                    {
                        "source": doc.metadata.get("source", "Unknown"),
                        "score": 1 - score/4,
                        "content": doc.page_content
                    }                    for doc, score in retrieved_docs
                ]
            })
        with open("retrieved_docs.json", "w") as out_file:
            json.dump(json_output, out_file, indent=4)
    else:
        # Mock data for testing
        mock_vectorstore = load_vectorstore(db_dir, None)  # Replace with actual embeddings if needed
        
        # Retrieve documents
        retrieved_docs = retrieve_documents(mock_vectorstore, query, k=4)
        
        # Print results for verification
        print("Retrieved Documents:")
        for (doc, score) in retrieved_docs:
            print(f"Source: {doc.metadata.get('source', 'Unknown')}")
            print(f"Score: {1 - score/4}")
            print(f"Content: {doc.page_content}\n")

def list_chunks(db_dir="chroma_db/"):
    vectorstore = load_vectorstore(db_dir, None)  # Replace with actual embeddings if needed
    all_docs = vectorstore.get()
    
    metadatas = all_docs.get('metadatas', [])
    documents = all_docs.get('documents', [])
    print(f"Total chunks in vector store: {len(documents)}")
    for i, (metadata, doc) in enumerate(zip(metadatas, documents)):
        source = metadata.get('source', 'Unknown')
        print(f"Chunk {i}: Source: {source}, Content: {doc[:50]}...{doc[-50:]}")  # Print first and last 50 chars of content

if __name__ == "__main__":
    test_retrieve(json_dir="/home/matcry/Documents/AI engineer roadmap/Yournotebook/tests/eval_question.json")
    #list_chunks()