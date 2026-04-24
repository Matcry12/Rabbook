from typing import Any, TypedDict

from agents.services import (
    AnswerResult,
    answer_is_grounded,
    build_citations,
    build_metadata_filter,
    build_sources,
)
from app.actions import ingest_saved_document
from agents.research_graph import run_research_agent
from rag.registry import load_chunk_registry
from rag.retrieve import (
    check_grounding_evidence,
    expand_with_context_window,
    format_context,
    generate_answer,
    generate_sub_queries,
    load_bm25_index,
    load_vectorstore,
    retrieve_documents_with_query_transform,
)
from core.config import DB_DIR, URL_IMPORT_DIR
from rag.ingest import add_documents_to_vectorstore
from rag.web_ingest import build_research_import_payload, save_url_import

try:
    from langgraph.graph import END, StateGraph
except ImportError:  # pragma: no cover - handled at runtime if dependency is missing
    END = None
    StateGraph = None


class RagGraphState(TypedDict, total=False):
    query: str
    selected_file: str
    selected_file_type: str
    page_start: str
    page_end: str
    debug_mode: bool
    metadata_filter: dict[str, Any] | None
    retrieved_documents: list[Any]
    expanded_documents: list[Any]
    grounding: dict[str, Any] | None
    next_action: str | None
    decision_reason: str | None
    retry_count: int
    refined_query: str | None
    web_research_attempted: bool
    web_research_performed: bool
    refreshed_vectorstore: Any
    refreshed_chunk_registry: dict[str, Any] | None
    refreshed_bm25_index: Any
    answer: str | None
    citations: list[dict]
    sources: list[dict]
    debug_data: dict[str, Any] | None


def build_initial_graph_state(
    query,
    *,
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
    debug_mode=False,
):
    return {
        "query": query,
        "selected_file": selected_file,
        "selected_file_type": selected_file_type,
        "page_start": page_start,
        "page_end": page_end,
        "debug_mode": debug_mode,
        "metadata_filter": None,
        "retrieved_documents": [],
        "expanded_documents": [],
        "grounding": None,
        "next_action": None,
        "decision_reason": None,
        "retry_count": 0,
        "refined_query": None,
        "web_research_attempted": False,
        "web_research_performed": False,
        "refreshed_vectorstore": None,
        "refreshed_chunk_registry": None,
        "refreshed_bm25_index": None,
        "answer": None,
        "citations": [],
        "sources": [],
        "debug_data": None,
    }


def retrieve_node(
    state: RagGraphState,
    *,
    vectorstore,
    reranker,
    bm25_index,
    llm,
    retrieval_k,
    rerank_candidate_k,
    bm25_candidate_k,
    enable_query_transform,
) -> RagGraphState:
    active_vectorstore = state.get("refreshed_vectorstore") or vectorstore
    active_bm25_index = state.get("refreshed_bm25_index") or bm25_index
    current_query = state.get("refined_query") or state["query"]
    should_transform_query = enable_query_transform and not state.get("refined_query")
    print(f"\n[RAG Agent] Node: retrieve | Query: '{current_query}'")
    print(
        f"[RAG Agent] Retrieve Context | Retry Count: {state.get('retry_count', 0)}"
        f" | Has Refined Query: {bool(state.get('refined_query'))}"
        f" | Query Transform Enabled: {should_transform_query}"
    )
    
    updated_state = dict(state)
    print("[RAG Agent] Retrieve Step: calling retrieve_documents_with_query_transform...")
    retrieval_result = retrieve_documents_with_query_transform(
        active_vectorstore,
        current_query,
        k=retrieval_k,
        reranker=reranker,
        bm25_index=active_bm25_index,
        query_transformer=llm,
        enable_query_transform=should_transform_query,
        candidate_k=rerank_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        metadata_filter=state.get("metadata_filter"),
        include_debug=state.get("debug_mode", False),
    )
    print("[RAG Agent] Retrieve Step: retrieve_documents_with_query_transform completed.")

    if state.get("debug_mode", False):
        retrieved_documents, new_debug_data = retrieval_result
        debug_data = dict(state.get("debug_data") or {})
        debug_data.update(new_debug_data)
        debug_data["metadata_filter"] = state.get("metadata_filter")
        debug_data["grounding"] = {
            "stage": "retrieval",
            "passed": None,
            "reason": "not_checked",
        }
    else:
        retrieved_documents = retrieval_result
        debug_data = None

    print(f"[RAG Agent] Retrieved {len(retrieved_documents)} documents.")
    updated_state["retrieved_documents"] = retrieved_documents
    updated_state["debug_data"] = debug_data
    return updated_state


def expand_context_node(
    state: RagGraphState,
    *,
    chunk_registry,
    context_window,
    max_expanded_chunks,
) -> RagGraphState:
    print(f"[RAG Agent] Node: expand_context | Window Size: {context_window}")
    updated_state = dict(state)
    active_chunk_registry = state.get("refreshed_chunk_registry") or chunk_registry
    expanded_documents = expand_with_context_window(
        state.get("retrieved_documents", []),
        active_chunk_registry,
        window_size=context_window,
        max_expanded_chunks=max_expanded_chunks,
    )
    updated_state["expanded_documents"] = expanded_documents

    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["expanded_hits"] = expanded_documents
        updated_state["debug_data"]["stage_counts"]["expanded_context"] = len(expanded_documents)

    return updated_state


def check_grounding_node(
    state: RagGraphState,
    *,
    min_grounded_rerank_score,
    min_grounded_chunks,
) -> RagGraphState:
    print("[RAG Agent] Node: check_grounding | Evaluating evidence...")
    updated_state = dict(state)
    grounding = check_grounding_evidence(
        state.get("retrieved_documents", []),
        state.get("expanded_documents", []),
        min_rerank_score=min_grounded_rerank_score,
        min_expanded_chunks=min_grounded_chunks,
    )
    updated_state["grounding"] = grounding

    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["grounding"].update(grounding)
        updated_state["debug_data"]["grounding"]["stage"] = "retrieval"

    print(f"[RAG Agent] Grounding Result: {'PASS' if grounding['passed'] else 'FAIL'} (Score: {grounding.get('top_rerank_score')})")
    return updated_state


def route_after_grounding(state: RagGraphState) -> str:
    action = state.get("next_action")
    print(f"[RAG Agent] Routing Decision: {action}")
    if action == "answer":
        return "generate_answer"
    if action == "retry_retrieval":
        return "refine_query"
    if action == "web_research":
        return "web_research"
    return "fallback_answer"


def decide_local_action(grounding, *, has_any_local_evidence, retry_count):
    if grounding.get("passed"):
        return "answer", "grounding_passed"

    if has_any_local_evidence and retry_count < 2:
        return "retry_retrieval", f"partial_local_evidence_retry_{retry_count + 1}"

    return None, None


def decide_terminal_action(*, enable_research, web_research_attempted, web_research_performed, retry_count):
    if enable_research and not web_research_attempted:
        return "web_research", "local_failed_switching_to_web"

    if web_research_attempted and not web_research_performed:
        return "fallback", "web_research_failed_no_documents_ingested"

    if web_research_performed:
        return "fallback", "web_research_also_failed"

    if retry_count >= 2:
        return "fallback", "local_retry_cap_reached_research_disabled"

    return "fallback", "no_evidence_found_research_disabled"


def decide_next_action_node(state: RagGraphState, *, enable_research: bool = False) -> RagGraphState:
    updated_state = dict(state)
    grounding = state.get("grounding") or {}
    retrieved_documents = state.get("retrieved_documents", [])
    expanded_documents = state.get("expanded_documents", [])

    retry_count = state.get("retry_count", 0)
    web_research_attempted = state.get("web_research_attempted", False)
    web_research_performed = state.get("web_research_performed", False)
    has_any_local_evidence = len(retrieved_documents) > 0 or len(expanded_documents) > 0
    print(
        f"[RAG Agent] Decision State | "
        f"retry_count={retry_count} | "
        f"web_research_attempted={web_research_attempted} | "
        f"web_research_performed={web_research_performed} | "
        f"has_any_local_evidence={has_any_local_evidence} | "
        f"enable_research={enable_research}"
    )

    next_action, decision_reason = decide_local_action(
        grounding,
        has_any_local_evidence=has_any_local_evidence,
        retry_count=retry_count,
    )
    if next_action is None:
        next_action, decision_reason = decide_terminal_action(
            enable_research=enable_research,
            web_research_attempted=web_research_attempted,
            web_research_performed=web_research_performed,
            retry_count=retry_count,
        )

    print(f"[RAG Agent] Decision: {next_action} | Reason: {decision_reason}")
    updated_state["next_action"] = next_action
    updated_state["decision_reason"] = decision_reason

    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["next_action"] = next_action
        updated_state["debug_data"]["decision_reason"] = decision_reason
        updated_state["debug_data"]["enable_research_flag"] = enable_research

    return updated_state


def refine_query_node(state: RagGraphState, *, llm) -> RagGraphState:
    print(f"[RAG Agent] Node: refine_query | Attempting to improve query (Retry {state.get('retry_count', 0) + 1})")
    updated_state = dict(state)
    print(f"[RAG Agent] Refine Step: original query='{state['query']}'")
    print(f"[RAG Agent] Refine Step: decision reason='{state.get('decision_reason', 'unknown')}'")
    print("[RAG Agent] Refine Step: calling shared query transform...")
    sub_queries = generate_sub_queries(
        state["query"],
        llm,
        max_queries=1,
    )
    print(f"[RAG Agent] Refine Step: shared query transform returned {len(sub_queries)} candidate(s): {sub_queries}")
    refined = sub_queries[0] if sub_queries else ""
    updated_state["refined_query"] = refined or state["query"]
    updated_state["retry_count"] = state.get("retry_count", 0) + 1

    print(f"[RAG Agent] Refined Query: '{updated_state['refined_query']}'")
    if state.get("debug_mode") and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["refined_query"] = updated_state["refined_query"]
        updated_state["debug_data"]["retry_count"] = updated_state["retry_count"]

    return updated_state


def web_research_node(
    state: RagGraphState,
    *,
    llm,
    vectorstore,
) -> RagGraphState:
    print(f"[RAG Agent] Node: web_research | Handing off to Research Agent for topic: '{state['query']}'")
    updated_state = dict(state)
    
    # Run the Research Agent to get web findings
    research_result = run_research_agent(
        topic=state["query"],
        llm=llm,
        debug_mode=state.get("debug_mode", False),
    )

    saved_paths = []
    skipped_results = 0
    for rs in research_result.sources:
        content = rs.get("content") or rs.get("snippet") or ""
        if len(content) < 50:
            print(f"[RAG Agent] Web Research Skip: content too short for '{rs.get('title', 'unknown')}'")
            skipped_results += 1
            continue

        payload = build_research_import_payload(rs)
        try:
            saved_path = save_url_import(payload, URL_IMPORT_DIR)
            print(f"[RAG Agent] Web Research Saved Import: {saved_path.name}")
            saved_paths.append(saved_path)
        except Exception as exc:
            print(f"[RAG Agent] Web Research Save FAILED for '{rs.get('url', '')}': {exc}")

    ingested_paths = 0
    for saved_path in saved_paths:
        try:
            embedding_function = getattr(vectorstore, "embeddings", None) or getattr(vectorstore, "_embedding_function", None)
            ingest_saved_document(
                saved_path,
                add_documents_to_vectorstore=add_documents_to_vectorstore,
                embeddings=embedding_function,
            )
            ingested_paths += 1
            print(f"[RAG Agent] Web Research Ingested: {saved_path.name}")
        except Exception as exc:
            print(f"[RAG Agent] Web Research Ingest FAILED for '{saved_path.name}': {exc}")

    if ingested_paths:
        refreshed_vectorstore = load_vectorstore(str(DB_DIR), embedding_function)
        refreshed_chunk_registry = load_chunk_registry()
        refreshed_bm25_index = load_bm25_index(
            chunk_registry=refreshed_chunk_registry,
            vectorstore=refreshed_vectorstore,
        )
        updated_state["refreshed_vectorstore"] = refreshed_vectorstore
        updated_state["refreshed_chunk_registry"] = refreshed_chunk_registry
        updated_state["refreshed_bm25_index"] = refreshed_bm25_index
        print(
            f"[RAG Agent] Web Research Refresh Complete | "
            f"Saved: {len(saved_paths)} | Ingested: {ingested_paths} | Skipped: {skipped_results}"
        )
    else:
        print(
            f"[RAG Agent] Web Research Produced No Ingested Documents | "
            f"Saved: {len(saved_paths)} | Ingested: 0 | Skipped: {skipped_results}"
        )
    
    updated_state["web_research_attempted"] = True
    updated_state["web_research_performed"] = ingested_paths > 0
    updated_state["retry_count"] = 0 
    
    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["web_research_attempted"] = True
        updated_state["debug_data"]["web_research_performed"] = updated_state["web_research_performed"]
        updated_state["debug_data"]["web_docs_saved"] = len(saved_paths)
        updated_state["debug_data"]["web_docs_ingested"] = ingested_paths
        updated_state["debug_data"]["web_docs_skipped"] = skipped_results

    print(
        f"[RAG Agent] Web Research Complete | "
        f"web_research_attempted={updated_state['web_research_attempted']} | "
        f"web_research_performed={updated_state['web_research_performed']} | "
        f"returning_to=retrieve"
    )
        
    return updated_state


def fallback_answer_node(
    state: RagGraphState,
    *,
    grounded_fallback_message,
) -> RagGraphState:
    print("[RAG Agent] Node: fallback_answer | Providing final fallback response.")
    updated_state = dict(state)
    updated_state["sources"] = build_sources(state.get("retrieved_documents", []))
    updated_state["answer"] = grounded_fallback_message
    updated_state["citations"] = []
    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["graph_path"] = "fallback_answer"
    return updated_state


def generate_answer_node(
    state: RagGraphState,
    *,
    llm,
    grounded_fallback_message,
) -> RagGraphState:
    print("[RAG Agent] Node: generate_answer | Creating grounded response...")
    updated_state = dict(state)
    retrieved_documents = state.get("retrieved_documents", [])
    updated_state["sources"] = build_sources(retrieved_documents)
    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["graph_path"] = "generate_answer"

    context = format_context(state.get("expanded_documents", []))
    answer = generate_answer(state["query"], context, llm)
    grounding = state.get("grounding") or {}
    if not answer_is_grounded(answer, context):
        print("[RAG Agent] Final Answer Validation FAILED. Falling back.")
        if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
            updated_state["debug_data"]["grounding"] = {
                "stage": "answer",
                "passed": False,
                "reason": "citation_validation_failed",
                "top_rerank_score": grounding.get("top_rerank_score"),
                "retrieved_count": grounding.get("retrieved_count"),
                "expanded_count": grounding.get("expanded_count"),
            }
        updated_state["answer"] = grounded_fallback_message
        updated_state["citations"] = []
        return updated_state

    print("[RAG Agent] Final Answer Validation PASSED.")
    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["grounding"] = {
            "stage": "answer",
            "passed": True,
            "reason": "answer_is_grounded",
            "top_rerank_score": grounding.get("top_rerank_score"),
            "retrieved_count": grounding.get("retrieved_count"),
            "expanded_count": grounding.get("expanded_count"),
        }

    updated_state["answer"] = answer
    updated_state["citations"] = build_citations(state.get("expanded_documents", []), answer)
    return updated_state


def prepare_input_node(state: RagGraphState) -> RagGraphState:
    print(f"\n--- Starting RAG Graph Session: '{state['query'][:50]}...' ---")
    updated_state = dict(state)
    updated_state["metadata_filter"] = build_metadata_filter(
        selected_file=state.get("selected_file", ""),
        selected_file_type=state.get("selected_file_type", ""),
        page_start=state.get("page_start", ""),
        page_end=state.get("page_end", ""),
    )
    return updated_state


def finalize_response_node(state: RagGraphState) -> RagGraphState:
    print("--- RAG Graph Session Complete ---\n")
    return dict(state)


def build_rag_graph(
    *,
    vectorstore=None,
    chunk_registry=None,
    reranker=None,
    bm25_index=None,
    llm=None,
    retrieval_k=4,
    rerank_candidate_k=8,
    bm25_candidate_k=8,
    context_window=1,
    max_expanded_chunks=12,
    min_grounded_rerank_score=1.0,
    min_grounded_chunks=1,
    grounded_fallback_message="I don't have enough support in the current documents to answer that confidently.",
    enable_query_transform=True,
    enable_research: bool = False,
):
    if StateGraph is None or END is None:
        raise RuntimeError("langgraph is not installed. Install requirements before using the graph.")

    graph = StateGraph(RagGraphState)
    graph.add_node("prepare_input", prepare_input_node)
    graph.add_node(
        "retrieve",
        lambda state: retrieve_node(
            state,
            vectorstore=vectorstore,
            reranker=reranker,
            bm25_index=bm25_index,
            llm=llm,
            retrieval_k=retrieval_k,
            rerank_candidate_k=rerank_candidate_k,
            bm25_candidate_k=bm25_candidate_k,
            enable_query_transform=enable_query_transform,
        ),
    )
    graph.add_node(
        "expand_context",
        lambda state: expand_context_node(
            state,
            chunk_registry=chunk_registry,
            context_window=context_window,
            max_expanded_chunks=max_expanded_chunks,
        ),
    )
    graph.add_node(
        "check_grounding",
        lambda state: check_grounding_node(
            state,
            min_grounded_rerank_score=min_grounded_rerank_score,
            min_grounded_chunks=min_grounded_chunks,
        ),
    )
    graph.add_node(
        "decide_next_action", 
        lambda state: decide_next_action_node(state, enable_research=enable_research)
    )
    graph.add_node(
        "refine_query",
        lambda state: refine_query_node(state, llm=llm),
    )
    graph.add_node(
        "web_research",
        lambda state: web_research_node(state, llm=llm, vectorstore=vectorstore),
    )
    graph.add_node(
        "generate_answer",
        lambda state: generate_answer_node(
            state,
            llm=llm,
            grounded_fallback_message=grounded_fallback_message,
        ),
    )
    graph.add_node(
        "fallback_answer",
        lambda state: fallback_answer_node(
            state,
            grounded_fallback_message=grounded_fallback_message,
        ),
    )
    graph.add_node("finalize_response", finalize_response_node)
    
    graph.set_entry_point("prepare_input")
    graph.add_edge("prepare_input", "retrieve")
    graph.add_edge("retrieve", "expand_context")
    graph.add_edge("expand_context", "check_grounding")
    graph.add_edge("check_grounding", "decide_next_action")
    
    graph.add_conditional_edges(
        "decide_next_action",
        route_after_grounding,
        {
            "generate_answer": "generate_answer",
            "refine_query": "refine_query",
            "web_research": "web_research",
            "fallback_answer": "fallback_answer",
        },
    )
    
    graph.add_edge("refine_query", "retrieve")
    graph.add_edge("web_research", "retrieve") 
    
    graph.add_edge("generate_answer", "finalize_response")
    graph.add_edge("fallback_answer", "finalize_response")
    graph.add_edge("finalize_response", END)
    
    return graph.compile()


def run_rag_graph_answer(
    query,
    *,
    vectorstore,
    chunk_registry,
    reranker,
    bm25_index,
    llm,
    retrieval_k,
    rerank_candidate_k,
    bm25_candidate_k,
    context_window,
    max_expanded_chunks,
    min_grounded_rerank_score,
    min_grounded_chunks,
    grounded_fallback_message,
    enable_query_transform,
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
    debug_mode=False,
    enable_research: bool = False,
):
    graph = build_rag_graph(
        vectorstore=vectorstore,
        chunk_registry=chunk_registry,
        reranker=reranker,
        bm25_index=bm25_index,
        llm=llm,
        retrieval_k=retrieval_k,
        rerank_candidate_k=rerank_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        context_window=context_window,
        max_expanded_chunks=max_expanded_chunks,
        min_grounded_rerank_score=min_grounded_rerank_score,
        min_grounded_chunks=min_grounded_chunks,
        grounded_fallback_message=grounded_fallback_message,
        enable_query_transform=enable_query_transform,
        enable_research=enable_research,
    )
    initial_state = build_initial_graph_state(
        query,
        selected_file=selected_file,
        selected_file_type=selected_file_type,
        page_start=page_start,
        page_end=page_end,
        debug_mode=debug_mode,
    )
    final_state = graph.invoke(initial_state)
    debug_data = final_state.get("debug_data")
    if debug_mode and debug_data is not None:
        debug_data["pipeline_mode"] = "langgraph_rag"
    return AnswerResult(
        answer=final_state.get("answer", grounded_fallback_message),
        sources=final_state.get("sources", []),
        citations=final_state.get("citations", []),
        debug_data=debug_data,
    )
