import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from core.config import DEFAULT_BM25_CANDIDATE_K, DEFAULT_RERANK_CANDIDATE_K, DEFAULT_RETRIEVAL_K
from rag.retrieve import format_context, retrieve_documents_with_query_transform
from rag.web_ingest import fetch_url_content, web_search as _web_search


SYSTEM_PROMPT = """You are a helpful research assistant with access to tools.

IMPORTANT: You must ALWAYS call a tool before answering. Never answer from memory alone.
- Call query_documents first for any question that could be in the local library.
- Call web_search only if query_documents returns no useful results.
- Only produce a final answer after you have called at least one tool."""

MAX_ITERATIONS = 8


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns a list of results with url, title, and snippet."""
    results = _web_search(query, max_results=3)
    return json.dumps(results, ensure_ascii=False)


@tool
def fetch_url(url: str) -> str:
    """Fetch and read the full text content of a web page."""
    try:
        payload = fetch_url_content(url)
        return payload["page_text"][:3000]
    except ValueError as exc:
        return f"Could not fetch page: {exc}"


WEB_TOOLS = [web_search, fetch_url]


def make_query_documents_tool(*, vectorstore, reranker, bm25_index, llm):
    @tool
    def query_documents(question: str) -> str:
        """Search the local document library for relevant information. Use this before searching the web."""
        documents = retrieve_documents_with_query_transform(
            vectorstore,
            question,
            k=DEFAULT_RETRIEVAL_K,
            reranker=reranker,
            bm25_index=bm25_index,
            query_transformer=llm,
            enable_query_transform=False,
            candidate_k=DEFAULT_RERANK_CANDIDATE_K,
            bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
            metadata_filter=None,
            include_debug=False,
        )
        if not documents:
            return "No relevant documents found in the local library."
        return format_context(documents)

    return query_documents


def run_tool_agent(
    query: str,
    *,
    llm,
    vectorstore=None,
    reranker=None,
    bm25_index=None,
) -> str:
    """
    Minimal agent loop.

    The LLM decides which tool to call each turn.
    We execute the tool and feed the result back.
    Loop ends when the LLM stops calling tools.

    If vectorstore, reranker, and bm25_index are provided, a query_documents
    tool is added so the agent can search local documents before the web.
    """
    tools = list(WEB_TOOLS)
    if vectorstore is not None and reranker is not None and bm25_index is not None:
        tools.append(make_query_documents_tool(
            vectorstore=vectorstore,
            reranker=reranker,
            bm25_index=bm25_index,
            llm=llm,
        ))

    llm_with_tools = llm.bind_tools(tools)
    tool_map = {t.name: t for t in tools}

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=query),
    ]

    for iteration in range(MAX_ITERATIONS):
        print(f"[Tool Agent] Iteration {iteration + 1}")
        response = llm_with_tools.invoke(messages)
        messages.append(response)

        if not response.tool_calls:
            print("[Tool Agent] No tool calls — returning final answer.")
            return response.content

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"[Tool Agent] Tool call: {name}({args})")

            fn = tool_map.get(name)
            result = fn.invoke(args) if fn else f"Unknown tool: {name}"
            print(f"[Tool Agent] Result length: {len(str(result))} chars")

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    return "Agent reached the iteration limit without a final answer."
