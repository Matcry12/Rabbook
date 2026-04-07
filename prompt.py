def build_rag_prompt(context, question):
    prompt = f"""
You are a helpful assistant.
Answer the question using only the provided context.
If the context does not contain the answer, say that the information is insufficient.
Do not make up facts.

Context:
{context}

Question:
{question}

Answer:
"""
    return prompt