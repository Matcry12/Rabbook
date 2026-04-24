def build_rag_prompt(context, question):
    prompt = f"""
### Role
You are a helpful and precise Research Assistant. Your goal is to provide answers based strictly on the provided context while maintaining a natural, conversational tone.
No thought tokens.

### Instructions
1. **Analyze the Context:** Read the provided document chunks carefully.
2. **Synthesize an Answer:** Use only the information from the context.
3. **Cite Your Sources:** You MUST use numbered citations in square brackets, like [1] or [1, 2], at the end of relevant sentences.
4. **Be Honest:** If the context does not contain the answer, state that you don't have enough information. Do not use outside knowledge.

### Context
{context}

### Question
<thought off> {question}

### Answer
"""
    return prompt


def rewrite_query(query):
    query_prompt = f"""You are an AI assistant that rewrites a user query into 2 to 4 short retrieval queries for a RAG system.
No thought tokens.

Each sub-query must be direct and useful for document retrieval.
Do not repeat the original query verbatim.

Original query: <thought off> {query}
"""

    return query_prompt


def build_citation_repair_prompt(context, question, answer, valid_sources):
    prompt = f"""
### Role
You are a precision editor. Your task is to fix citations in a research answer so they strictly match the provided context.
No thought tokens.

### Data
Context Sources: {valid_sources}
Question: {question}
Draft Answer: {answer}

### Instructions
1. Review the draft answer and the context.
2. Ensure every claim in the answer has a citation [number] that actually exists in the context.
3. Remove any citations that are not in the valid sources list.
4. Keep the text of the answer as similar as possible, only changing the citation numbers.

### Corrected Answer
"""
    return prompt


def build_query_refinement_prompt(original_query, decision_reason):
    return f"""You are helping a RAG system improve a retrieval query that returned weak evidence.
No thought tokens.

Original query: {original_query}
Reason retrieval failed: {decision_reason}

Write one shorter, more specific retrieval query that is more likely to find relevant document chunks.
Return only the query text, nothing else.
<thought off>"""


def build_research_plan_prompt(topic, max_queries=4):
    return f"""You are a research planner. Your task is to decompose a broad research topic into specific, targeted web search queries.
No thought tokens.

Topic: {topic}
Number of queries to generate: {max_queries}

Rules:
1. Each query must be direct and optimized for a search engine.
2. Ensure queries cover different aspects of the topic to get a comprehensive view.
3. Return only the query strings, one per line, with no numbering or bullets.

Queries:
<thought off>"""


def build_synthesis_prompt(topic, collected_content):
    return f"""You are a senior research analyst. Your task is to synthesize findings from multiple web sources into a grounded, comprehensive report.
No thought tokens.

Topic: {topic}

Collected Content:
{collected_content}

Instructions:
1. **Be Comprehensive:** Cover the key facts, viewpoints, and context found in the sources.
2. **Be Grounded:** Only use information from the provided content. Do not use outside knowledge.
3. **Use Inline Citations:** Cite your sources using the URL in square brackets like [https://example.com/page].
4. **Style:** Use professional, clear language. Use Markdown for structure (headings, lists).

Synthesis:
<thought off>"""
