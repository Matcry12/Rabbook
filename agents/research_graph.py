from typing import Any, TypedDict

from agents.services import ResearchResult
from rag.retrieve import extract_response_text, generate_sub_queries
from rag.web_ingest import fetch_url_content, web_search
from rag.notes import save_note

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover
    END = None
    StateGraph = None


class ResearchGraphState(TypedDict, total=False):
    topic: str
    search_queries: list[str]
    search_results: list[dict]   # {url, title, snippet, content}
    synthesis: str | None
    save_to_notes: bool
    note_id: str | None
    error: str | None
    debug_mode: bool
    debug_data: dict[str, Any] | None


def plan_search_node(state: ResearchGraphState, *, llm, max_queries: int) -> ResearchGraphState:
    print(f"[Research Agent] Node: plan_search | Topic: '{state['topic']}'")
    updated_state = dict(state)
    updated_state["search_queries"] = generate_sub_queries(
        state["topic"],
        llm,
        max_queries=max_queries,
    )
    print(f"[Research Agent] Planned {len(updated_state['search_queries'])} query(s): {updated_state['search_queries']}")
    
    if state.get("debug_mode") and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["search_queries"] = updated_state["search_queries"]
    
    return updated_state


def execute_search_node(state: ResearchGraphState) -> ResearchGraphState:
    print("[Research Agent] Node: execute_search | Starting web search execution")
    updated_state = dict(state)
    all_results = []
    seen_urls = set()
    
    for query in state.get("search_queries", []):
        try:
            print(f"[Research Agent] Search Query: '{query}'")
            results = web_search(query, max_results=3)
            print(f"[Research Agent] Search returned {len(results)} raw result(s) for '{query}'")
            for r in results:
                if r["url"] not in seen_urls:
                    all_results.append(r)
                    seen_urls.add(r["url"])
        except Exception as exc:
            print(f"[Research Agent] Search FAILED for '{query}': {exc}")
            continue

    if not all_results:
        print("[Research Agent] No search results found across all queries.")
        updated_state["error"] = "no_search_results"
        return updated_state

    # Fetch full content only for the top 3 results to keep context lean
    for res in all_results[:3]:
        try:
            print(f"[Research Agent] Fetching content for: {res['url']}")
            content_payload = fetch_url_content(res["url"])
            res["content"] = content_payload["page_text"]
            print(f"[Research Agent] Fetch SUCCESS for: {res['url']} | chars={len(res['content'])}")
        except Exception as exc:
            print(f"[Research Agent] Fetch FAILED for: {res['url']} | using snippet fallback | error={exc}")
            res["content"] = res["snippet"]

    updated_state["search_results"] = all_results
    print(f"[Research Agent] Search execution complete | unique results={len(all_results)}")
    
    if state.get("debug_mode") and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["results_found"] = len(all_results)
        
    return updated_state


def save_note_node(state: ResearchGraphState) -> ResearchGraphState:
    print("[Research Agent] Node: save_note | Checking whether note should be saved")
    updated_state = dict(state)
    if not state.get("save_to_notes") or not state.get("synthesis"):
        print("[Research Agent] Save note skipped.")
        return updated_state

    # For research notes, we use the topic as the query and the synthesis as the answer
    citations = [
        {
            "number": i + 1,
            "source": r["url"],
            "content": r["snippet"]
        }
        for i, r in enumerate(state.get("search_results", []))
    ]
    
    try:
        note_id = save_note(
            query=state["topic"],
            answer=state["synthesis"],
            citations=citations
        )
        updated_state["note_id"] = note_id
        print(f"[Research Agent] Note saved successfully: {note_id}")
    except Exception as exc:
        print(f"[Research Agent] Save note FAILED: {exc}")
        
    return updated_state


def finalize_research_node(state: ResearchGraphState) -> ResearchGraphState:
    print("[Research Agent] Node: finalize")
    updated_state = dict(state)
    if state.get("error") == "no_search_results":
        updated_state["synthesis"] = f"I was unable to find any web search results for the topic: {state['topic']}"
        print("[Research Agent] Finalized with no search results.")
    elif not state.get("synthesis"):
        search_results = state.get("search_results", [])
        if search_results:
            top_titles = ", ".join(result.get("title", "Untitled") for result in search_results[:3])
            updated_state["synthesis"] = f"Collected web sources for {state['topic']}: {top_titles}"
        else:
            updated_state["synthesis"] = f"Collected web sources for {state['topic']}."
    
    return updated_state


def route_research(state: ResearchGraphState) -> str:
    return "finalize"


def build_research_graph(*, llm, max_queries: int = 4):
    if StateGraph is None:
        raise RuntimeError("langgraph is not installed.")

    graph = StateGraph(ResearchGraphState)
    
    graph.add_node("plan_search", lambda s: plan_search_node(s, llm=llm, max_queries=max_queries))
    graph.add_node("execute_search", execute_search_node)
    graph.add_node("save_note", save_note_node)
    graph.add_node("finalize", finalize_research_node)
    
    graph.set_entry_point("plan_search")
    graph.add_edge("plan_search", "execute_search")
    
    graph.add_conditional_edges(
        "execute_search",
        route_research,
        {
            "finalize": "finalize"
        }
    )
    
    graph.add_edge("finalize", END)
    
    return graph.compile()


def run_research_agent(
    topic: str,
    *,
    llm,
    save_to_notes: bool = False,
    max_queries: int = 4,
    debug_mode: bool = False,
) -> ResearchResult:
    print(f"[Research Agent] Starting session for topic: '{topic}'")
    graph = build_research_graph(llm=llm, max_queries=max_queries)
    
    initial_state = {
        "topic": topic,
        "search_queries": [],
        "search_results": [],
        "synthesis": None,
        "save_to_notes": save_to_notes,
        "note_id": None,
        "error": None,
        "debug_mode": debug_mode,
        "debug_data": {} if debug_mode else None,
    }
    
    final_state = graph.invoke(initial_state)
    print(
        f"[Research Agent] Session complete | "
        f"error={final_state.get('error')} | "
        f"results={len(final_state.get('search_results', []))} | "
        f"note_id={final_state.get('note_id')}"
    )
    
    # Format sources for the result
    sources = [
        {
            "url": r["url"],
            "title": r["title"],
            "snippet": r["snippet"],
            "content": r.get("content", r["snippet"]),
        }
        for r in final_state.get("search_results", [])
    ]
    
    return ResearchResult(
        synthesis=final_state.get("synthesis") or "Research failed.",
        sources=sources,
        note_id=final_state.get("note_id"),
        debug_data=final_state.get("debug_data")
    )
