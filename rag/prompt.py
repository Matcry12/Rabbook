def build_rag_prompt(context, question):
    prompt = f"""
### Role
You are a helpful and precise Research Assistant. Your goal is to provide answers based strictly on the provided context while maintaining a natural, conversational tone.

### Instructions
1. **Greetings:** If the user greets you or asks how you are, respond politely and briefly as a helpful AI before addressing their query.
2. **Grounded Answering:** Use the provided [Context] to answer the [Question]. 
3. **Strictness:** If the [Context] does not contain the specific information needed to answer the question, state clearly that the information is not available in the current documents. 
4. **No Hallucination:** Do not use outside knowledge to supplement the facts, but you may use outside knowledge to explain terms or concepts found within the context.
5. **Citations:** When you make a factual claim, cite the supporting source using square brackets like [1] or [2]. Use only the source numbers provided in the context.
6. **Citation Discipline:** Do not invent citations, and do not cite a source unless it supports the claim.

### Context:
{context}

### Question:
{question}

### Answer:
"""
    return prompt


def build_citation_repair_prompt(context, question, answer, valid_sources):
    prompt = f"""
You are fixing a RAG answer so that every factual claim has valid citations.

Rules:
1. Use only citations from this allowed source list: {valid_sources}.
2. Every factual sentence must end with at least one citation like [1] or [2].
3. Do not invent facts or citations.
4. If the context does not support the answer, say the information is not available in the current documents.
5. Return only the corrected answer text.

Question:
{question}

Available Context:
{context}

Draft Answer:
{answer}

Corrected Answer:
"""
    return prompt


def rewrite_query(query):
    query_prompt = f"""You are an AI assistant that rewrites a user query into 2 to 4 short retrieval queries for a RAG system.

Each sub-query must be direct and useful for document retrieval.
Do not repeat the original query verbatim.

Original query: {query}
"""

    return query_prompt
