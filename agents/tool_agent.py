import json

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from app.actions import ingest_saved_document
from core.config import DB_DIR, DEFAULT_BM25_CANDIDATE_K, DEFAULT_RERANK_CANDIDATE_K, DEFAULT_RETRIEVAL_K, URL_IMPORT_DIR
from rag.ingest import add_documents_to_vectorstore
from rag.registry import load_chunk_registry
from rag.retrieve import (
    format_context,
    load_bm25_index,
    load_vectorstore,
    retrieve_documents_with_query_transform,
)
from rag.web_ingest import fetch_url_content, save_url_import, web_search as _web_search


SYSTEM_PROMPT = """You are a research assistant with three tools: query_documents (local document corpus), web_search (short web snippets), and fetch_url (read a full web page). Answer using evidence from tools, never from memory alone.

FINDING THE ANSWER
1. Always start with query_documents. Use the facts in the returned context — do not ignore retrieved evidence.
2. If the local documents do not contain the answer, you MUST try web_search before giving up. Never refuse after checking only local documents.
3. web_search returns only short snippets. If a snippet directly states the answer, use it. If it does not, call fetch_url on the most relevant result to read the full page — do NOT run another similar web_search.
4. Never repeat a search you have already run.

ANSWERING
5. Answer at the level the question asks for: if it asks for a city, name the city (not the institution); if it asks what two things share, give the simplest common term; if it asks "born and raised", give both.
6. For yes/no questions, begin with "Yes" or "No". For "which has more / who won more", you MUST pick one based on the evidence — never say it cannot be compared.
7. For multi-part questions, resolve each part, then combine into one answer.

WHEN TO STOP
8. Once you have enough evidence, give a direct, complete final answer.
9. Only state facts a tool result directly supports. If, after checking both local documents and the web, you still cannot find the answer, say so plainly — never guess and never stitch unrelated fragments into a confident answer. Never return an empty answer."""

MAX_ITERATIONS = 8


@tool
def web_search(query: str) -> str:
    """Search the web for current information. Returns a list of results with url, title, and snippet."""
    results = _web_search(query, max_results=3)
    return json.dumps(results, ensure_ascii=False)


def make_fetch_and_embed_tool(*, embeddings):
    @tool
    def fetch_url(url: str) -> str:
        """Fetch a web page, embed it into the local library, then use query_documents to read it."""
        try:
            payload = fetch_url_content(url)
            saved_path = save_url_import(payload, URL_IMPORT_DIR)
            ingest_saved_document(
                saved_path,
                add_documents_to_vectorstore=add_documents_to_vectorstore,
                embeddings=embeddings,
            )
            print(f"[Tool Agent] Embedded and saved: {saved_path.name}")
            return f"Page '{payload['title']}' has been indexed. Use query_documents to search its content."
        except ValueError as exc:
            return f"Could not fetch page: {exc}"

    return fetch_url


def make_query_documents_tool(*, embeddings, reranker, llm):
    @tool
    def query_documents(question: str) -> str:
        """Search the local document library for relevant information. Use this before searching the web."""
        vectorstore = load_vectorstore(str(DB_DIR), embeddings)
        chunk_registry = load_chunk_registry()
        bm25_index = load_bm25_index(chunk_registry=chunk_registry, vectorstore=vectorstore)

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
    embeddings=None,
    reranker=None,
    trace: list | None = None,
) -> str:
    """
    Minimal agent loop.

    The LLM decides which tool to call each turn.
    We execute the tool and feed the result back.
    Loop ends when the LLM stops calling tools.

    If embeddings and reranker are provided:
    - query_documents reloads vectorstore from disk on each call (sees newly embedded content)
    - fetch_url auto-embeds fetched pages into the local library
    """
    tools = [web_search]
    if embeddings is not None and reranker is not None:
        tools.append(make_fetch_and_embed_tool(embeddings=embeddings))
        tools.append(make_query_documents_tool(embeddings=embeddings, reranker=reranker, llm=llm))

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
            if trace is not None:
                trace.append({"final_answer": response.content})
            return response.content

        for tool_call in response.tool_calls:
            name = tool_call["name"]
            args = tool_call["args"]
            print(f"[Tool Agent] Tool call: {name}({args})")

            fn = tool_map.get(name)
            result = fn.invoke(args) if fn else f"Unknown tool: {name}"
            print(f"[Tool Agent] Result length: {len(str(result))} chars")

            if trace is not None:
                trace.append({"tool": name, "args": args, "result_chars": len(str(result))})

            messages.append(
                ToolMessage(content=str(result), tool_call_id=tool_call["id"])
            )

    limit_message = "Agent reached the iteration limit without a final answer."
    if trace is not None:
        trace.append({"final_answer": limit_message})
    return limit_message
