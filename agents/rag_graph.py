from typing import Any, TypedDict

from agents.services import (
    AnswerResult,
    answer_is_grounded,
    build_citations,
    build_metadata_filter,
    build_sources,
)
from rag.prompt import build_query_refinement_prompt
from rag.retrieve import (
    check_grounding_evidence,
    expand_with_context_window,
    format_context,
    generate_answer,
    retrieve_documents_with_query_transform,
)

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
    updated_state = dict(state)
    retrieval_result = retrieve_documents_with_query_transform(
        vectorstore,
        state.get("refined_query") or state["query"],
        k=retrieval_k,
        reranker=reranker,
        bm25_index=bm25_index,
        query_transformer=llm,
        enable_query_transform=enable_query_transform,
        candidate_k=rerank_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        metadata_filter=state.get("metadata_filter"),
        include_debug=state.get("debug_mode", False),
    )

    if state.get("debug_mode", False):
        retrieved_documents, debug_data = retrieval_result
        debug_data["metadata_filter"] = state.get("metadata_filter")
        debug_data["grounding"] = {
            "stage": "retrieval",
            "passed": None,
            "reason": "not_checked",
        }
    else:
        retrieved_documents = retrieval_result
        debug_data = None

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
    updated_state = dict(state)
    expanded_documents = expand_with_context_window(
        state.get("retrieved_documents", []),
        chunk_registry,
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

    return updated_state


def route_after_grounding(state: RagGraphState) -> str:
    if state.get("next_action") == "answer":
        return "generate_answer"
    if state.get("next_action") == "retry_retrieval":
        return "refine_query"
    return "fallback_answer"


def decide_next_action_node(state: RagGraphState) -> RagGraphState:
    updated_state = dict(state)
    grounding = state.get("grounding") or {}
    retrieved_documents = state.get("retrieved_documents", [])
    expanded_documents = state.get("expanded_documents", [])

    retry_count = state.get("retry_count", 0)

    if grounding.get("passed"):
        next_action = "answer"
        decision_reason = "grounding_passed"
    elif (retrieved_documents or expanded_documents) and retry_count < 2:
        next_action = "retry_retrieval"
        decision_reason = "partial_local_evidence"
    else:
        next_action = "fallback"
        decision_reason = "retry_cap_reached" if retry_count >= 2 else "no_local_evidence"

    updated_state["next_action"] = next_action
    updated_state["decision_reason"] = decision_reason

    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["next_action"] = next_action
        updated_state["debug_data"]["decision_reason"] = decision_reason

    return updated_state


def refine_query_node(state: RagGraphState, *, llm) -> RagGraphState:
    updated_state = dict(state)
    prompt = build_query_refinement_prompt(
        original_query=state["query"],
        decision_reason=state.get("decision_reason", "unknown"),
    )
    response = llm.invoke(prompt)
    refined = (getattr(response, "content", None) or getattr(response, "text", "") or "").strip()
    updated_state["refined_query"] = refined or state["query"]
    updated_state["retry_count"] = state.get("retry_count", 0) + 1

    if state.get("debug_mode") and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["refined_query"] = updated_state["refined_query"]
        updated_state["debug_data"]["retry_count"] = updated_state["retry_count"]

    return updated_state


def fallback_answer_node(
    state: RagGraphState,
    *,
    grounded_fallback_message,
) -> RagGraphState:
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
    updated_state = dict(state)
    retrieved_documents = state.get("retrieved_documents", [])
    updated_state["sources"] = build_sources(retrieved_documents)
    if state.get("debug_mode", False) and updated_state.get("debug_data") is not None:
        updated_state["debug_data"]["graph_path"] = "generate_answer"

    context = format_context(state.get("expanded_documents", []))
    answer = generate_answer(state["query"], context, llm)
    grounding = state.get("grounding") or {}
    if not answer_is_grounded(answer, context):
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
    updated_state = dict(state)
    updated_state["metadata_filter"] = build_metadata_filter(
        selected_file=state.get("selected_file", ""),
        selected_file_type=state.get("selected_file_type", ""),
        page_start=state.get("page_start", ""),
        page_end=state.get("page_end", ""),
    )
    return updated_state


def finalize_response_node(state: RagGraphState) -> RagGraphState:
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
    graph.add_node("decide_next_action", decide_next_action_node)
    graph.add_node(
        "refine_query",
        lambda state: refine_query_node(state, llm=llm),
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
            "fallback_answer": "fallback_answer",
        },
    )
    graph.add_edge("refine_query", "retrieve")
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
