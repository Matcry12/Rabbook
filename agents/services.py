from dataclasses import dataclass

from rag.retrieve import (
    answer_has_valid_citations,
    build_citation_sources,
    build_hit_debug,
    check_grounding_evidence,
    expand_with_context_window,
    extract_citation_numbers,
    extract_valid_source_numbers,
    format_context,
    generate_answer,
    retrieve_documents_with_query_transform,
)


@dataclass
class AnswerResult:
    answer: str
    sources: list[dict]
    citations: list[dict]
    debug_data: dict | None


@dataclass
class ResearchResult:
    synthesis: str
    sources: list[dict]   # {url, title, snippet, content}
    note_id: str | None
    debug_data: dict | None


def run_rag_graph_answer(*args, **kwargs):
    from agents.rag_graph import run_rag_graph_answer as graph_runner

    return graph_runner(*args, **kwargs)


def answer_query(
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
    use_langgraph=False,
    enable_research=False,
):
    if use_langgraph:
        result = run_rag_graph_answer(
            query,
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
            selected_file=selected_file,
            selected_file_type=selected_file_type,
            page_start=page_start,
            page_end=page_end,
            debug_mode=debug_mode,
            enable_research=enable_research,
        )
        if debug_mode and result.debug_data is not None:
            result.debug_data["pipeline_mode"] = "langgraph_rag"
        return result

    # Non-agentic path (Direct RAG) does not support research fallback yet
    metadata_filter = build_metadata_filter(
        selected_file=selected_file,
        selected_file_type=selected_file_type,
        page_start=page_start,
        page_end=page_end,
    )
    retrieval_result = retrieve_documents_with_query_transform(
        vectorstore,
        query,
        k=retrieval_k,
        reranker=reranker,
        bm25_index=bm25_index,
        query_transformer=llm,
        enable_query_transform=enable_query_transform,
        candidate_k=rerank_candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        metadata_filter=metadata_filter,
        include_debug=debug_mode,
    )
    if debug_mode:
        retrieved_documents, debug_data = retrieval_result
        debug_data["metadata_filter"] = metadata_filter
        debug_data["pipeline_mode"] = "direct_rag"
        debug_data["grounding"] = {
            "stage": "retrieval",
            "passed": None,
            "reason": "not_checked",
        }
    else:
        retrieved_documents = retrieval_result
        debug_data = None

    expanded_documents = expand_with_context_window(
        retrieved_documents,
        chunk_registry,
        window_size=context_window,
        max_expanded_chunks=max_expanded_chunks,
    )
    if debug_mode:
        debug_data["expanded_hits"] = build_hit_debug(expanded_documents)
        debug_data["stage_counts"]["expanded_context"] = len(expanded_documents)

    grounding = check_grounding_evidence(
        retrieved_documents,
        expanded_documents,
        min_rerank_score=min_grounded_rerank_score,
        min_expanded_chunks=min_grounded_chunks,
    )
    if debug_mode:
        debug_data["grounding"].update(grounding)
        debug_data["grounding"]["stage"] = "retrieval"

    if not grounding["passed"]:
        return AnswerResult(
            answer=grounded_fallback_message,
            sources=build_sources(retrieved_documents),
            citations=[],
            debug_data=debug_data,
        )

    context = format_context(expanded_documents)
    answer = generate_answer(query, context, llm)
    if not answer_is_grounded(answer, context):
        if debug_mode:
            debug_data["grounding"] = {
                "stage": "answer",
                "passed": False,
                "reason": "citation_validation_failed",
                "top_rerank_score": grounding["top_rerank_score"],
                "retrieved_count": grounding["retrieved_count"],
                "expanded_count": grounding["expanded_count"],
            }
        return AnswerResult(
            answer=grounded_fallback_message,
            sources=build_sources(retrieved_documents),
            citations=[],
            debug_data=debug_data,
        )

    if debug_mode:
        debug_data["grounding"] = {
            "stage": "answer",
            "passed": True,
            "reason": "answer_is_grounded",
            "top_rerank_score": grounding["top_rerank_score"],
            "retrieved_count": grounding["retrieved_count"],
            "expanded_count": grounding["expanded_count"],
        }

    return AnswerResult(
        answer=answer,
        sources=build_sources(retrieved_documents),
        citations=build_citations(expanded_documents, answer),
        debug_data=debug_data,
    )


def build_sources(documents):
    return [
        {
            "source": doc.metadata.get("file_name", "Unknown"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "retrieval_score": format_score(doc.metadata.get("retrieval_score", score)),
            "rerank_score": format_score(doc.metadata.get("rerank_score", score)),
            "content": doc.page_content,
        }
        for doc, score in documents
    ]


def build_citations(documents, answer):
    citations = build_citation_sources(documents)
    used_numbers = set(extract_citation_numbers(answer))

    filtered_citations = []
    for item in citations:
        if item["number"] not in used_numbers:
            continue
        item["retrieval_score"] = format_score(item.get("retrieval_score"))
        item["rerank_score"] = format_score(item.get("rerank_score"))
        filtered_citations.append(item)

    return filtered_citations


def format_score(score):
    if score is None:
        return "n/a"
    return f"{float(score):.4f}"


def answer_is_grounded(answer, context):
    valid_sources = extract_valid_source_numbers(context)
    return answer_has_valid_citations(answer, valid_sources)


def build_metadata_filter(
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
):
    metadata_filter = {}

    if selected_file:
        metadata_filter["file_name"] = selected_file

    if selected_file_type:
        metadata_filter["file_type"] = selected_file_type

    page_range = build_page_range(page_start, page_end)
    if page_range is not None:
        metadata_filter["page_range"] = page_range

    return metadata_filter or None


def build_page_range(page_start, page_end):
    start = parse_page_number(page_start)
    end = parse_page_number(page_end)

    if start is None and end is None:
        return None

    if start is not None and end is not None and start > end:
        start, end = end, start

    return {"start": start, "end": end}


def parse_page_number(value):
    if value in (None, ""):
        return None

    page_number = int(str(value).strip())
    if page_number < 1:
        raise ValueError("Page filters must be 1 or greater.")
    return page_number
