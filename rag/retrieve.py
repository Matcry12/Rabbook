from pathlib import Path
import json
import re
import time
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from pydantic import BaseModel, Field

from core.config import (
    DEFAULT_BM25_CANDIDATE_K,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_ENABLE_QUERY_TRANSFORM,
    DEFAULT_MAX_EXPANDED_CHUNKS,
    DEFAULT_MIN_GROUNDED_CHUNKS,
    DEFAULT_MIN_GROUNDED_RERANK_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RRF_K,
    DEFAULT_SUBQUERY_COUNT,
    REGISTRY_PATH,
)
from rag.prompt import build_citation_repair_prompt, build_rag_prompt, rewrite_query


class QueryRewriteResult(BaseModel):
    sub_queries: list[str] = Field(default_factory=list)


class AnswerDraftResult(BaseModel):
    answer: str = ""


def load_vectorstore(persist_dir, embeddings):
    """
    Load a Chroma vector store from the given directory.
    """

    return Chroma(
        embedding_function=embeddings,
        persist_directory=persist_dir,
    )


def retrieve_documents(vectorstore, query, k=4, reranker=None, candidate_k=DEFAULT_RERANK_CANDIDATE_K):
    """
    Retrieve candidate chunks with embeddings first, then optionally rerank them.
    """

    docs = vectorstore.similarity_search_with_score(query, k=max(k, candidate_k))
    unique_docs = deduplicate_documents(docs)

    if reranker is None:
        return unique_docs[:k]

    return rerank_documents(query, unique_docs, reranker, top_n=k)


def retrieve_documents_with_query_transform(
    vectorstore,
    query,
    k=4,
    reranker=None,
    bm25_index=None,
    query_transformer=None,
    enable_query_transform=DEFAULT_ENABLE_QUERY_TRANSFORM,
    candidate_k=DEFAULT_RERANK_CANDIDATE_K,
    bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
    rrf_k=DEFAULT_RRF_K,
    subquery_count=DEFAULT_SUBQUERY_COUNT,
    metadata_filter=None,
    include_debug=False,
):
    """
    Retrieve with optional query transformation.
    The important part is that reranking happens once on the merged candidate set,
    using the original user query as the final ranking signal.
    """

    search_queries = [query]
    debug = {
        "search_queries": [],
        "dense_hits": {},
        "bm25_hits": {},
        "fused_hits": [],
        "reranked_hits": [],
        "stage_counts": {},
    }
    if enable_query_transform and query_transformer is not None:
        sub_queries = generate_sub_queries(query, query_transformer, max_queries=subquery_count)
        search_queries.extend(sub_queries)
    debug["search_queries"] = search_queries

    candidate_documents, fusion_debug = collect_candidate_documents(
        vectorstore,
        search_queries,
        bm25_index=bm25_index,
        candidate_k=candidate_k,
        bm25_candidate_k=bm25_candidate_k,
        rrf_k=rrf_k,
        metadata_filter=metadata_filter,
        include_debug=include_debug,
    )
    if include_debug:
        debug.update(fusion_debug)
        debug["stage_counts"] = {
            "search_queries": len(search_queries),
            "fused_candidates": len(candidate_documents),
        }

    if reranker is None:
        top_documents = candidate_documents[:k]
        if include_debug:
            debug["reranked_hits"] = build_hit_debug(top_documents)
            debug["stage_counts"]["final_hits"] = len(top_documents)
            return top_documents, debug
        return top_documents

    top_documents = rerank_documents(query, candidate_documents, reranker, top_n=k)
    if include_debug:
        debug["reranked_hits"] = build_hit_debug(top_documents)
        debug["stage_counts"]["final_hits"] = len(top_documents)
        return top_documents, debug
    return top_documents


def deduplicate_documents(documents):
    unique_docs = []
    seen_content = set()

    for doc, score in documents:
        content = doc.page_content.strip()
        if content in seen_content:
            continue
        unique_docs.append((doc, score))
        seen_content.add(content)

    return unique_docs


def extract_response_text(response):
    if response is None:
        return ""

    content = getattr(response, "content", None)
    if isinstance(content, list):
        text_parts = []
        for part in content:
            if isinstance(part, str):
                text_parts.append(part.strip())
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(str(part["text"]).strip())
        return " ".join(part for part in text_parts if part).strip()

    if isinstance(content, str):
        return content.strip()

    text = getattr(response, "text", "")
    if isinstance(text, str):
        return text.strip()

    return ""


def generate_sub_queries(query, query_transformer, max_queries=DEFAULT_SUBQUERY_COUNT):
    print(f"[Query Transform] Start | Query='{query}' | Max Queries={max_queries}")
    prompt = rewrite_query(query)
    print("[Query Transform] Prompt built.")
    structured_transformer = get_structured_query_transformer(query_transformer)
    if structured_transformer is not None:
        print("[Query Transform] Structured transformer available. Starting structured invoke...")
        started_at = time.perf_counter()
        try:
            structured_result = structured_transformer.invoke(prompt)
            elapsed = time.perf_counter() - started_at
            print(f"[Query Transform] Structured invoke completed in {elapsed:.2f}s.")
        except Exception:
            elapsed = time.perf_counter() - started_at
            print(f"[Query Transform] Structured invoke failed after {elapsed:.2f}s.")
            structured_result = None
        sub_queries = parse_structured_sub_queries(structured_result, query, max_queries=max_queries)
        if sub_queries:
            print(f"[Query Transform] Structured parse returned {len(sub_queries)} query(s): {sub_queries}")
            return sub_queries
        print("[Query Transform] Structured parse returned no usable queries.")
    else:
        print("[Query Transform] Structured transformer unavailable.")

    print("[Query Transform] Falling back to direct invoke...")
    fallback_started_at = time.perf_counter()
    response = query_transformer.invoke(prompt)
    fallback_elapsed = time.perf_counter() - fallback_started_at
    print(f"[Query Transform] Direct invoke completed in {fallback_elapsed:.2f}s.")
    response_text = extract_response_text(response)
    print(f"[Query Transform] Extracted response text length: {len(response_text)}")

    json_sub_queries = parse_sub_queries_json(response_text, query, max_queries=max_queries)
    if json_sub_queries:
        print(f"[Query Transform] JSON parse returned {len(json_sub_queries)} query(s): {json_sub_queries}")
        return json_sub_queries

    sub_queries = []
    for line in response_text.splitlines():
        cleaned = re.sub(r"^\s*\d+[\).\-\s]+", "", line).strip()
        if not cleaned or cleaned.lower().startswith("sub-queries"):
            continue
        if not is_valid_retrieval_query(cleaned, original_query=query):
            continue
        sub_queries.append(cleaned)
        if len(sub_queries) >= max_queries:
            break

    print(f"[Query Transform] Line parse returned {len(sub_queries)} query(s): {sub_queries}")
    return sub_queries


def get_structured_query_transformer(query_transformer):
    if query_transformer is None:
        return None

    with_structured_output = getattr(query_transformer, "with_structured_output", None)
    if with_structured_output is None:
        return None

    try:
        # Explicitly use function_calling method for better compatibility 
        # and much faster response times with Gemma models.
        return with_structured_output(QueryRewriteResult, method="function_calling")
    except Exception:
        return None


def parse_structured_sub_queries(response: Any, query, max_queries=DEFAULT_SUBQUERY_COUNT):
    if isinstance(response, QueryRewriteResult):
        raw_sub_queries = response.sub_queries
    elif isinstance(response, dict):
        raw_sub_queries = response.get("sub_queries")
    else:
        raw_sub_queries = getattr(response, "sub_queries", None)
        if raw_sub_queries is None and isinstance(response, str):
            return parse_sub_queries_json(response, query, max_queries=max_queries)

    if not isinstance(raw_sub_queries, list):
        return []

    cleaned_sub_queries = []
    seen_queries = set()
    for item in raw_sub_queries:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not is_valid_retrieval_query(cleaned, original_query=query):
            continue
        dedupe_key = cleaned.lower()
        if dedupe_key in seen_queries:
            continue
        cleaned_sub_queries.append(cleaned)
        seen_queries.add(dedupe_key)
        if len(cleaned_sub_queries) >= max_queries:
            break

    return cleaned_sub_queries


def parse_sub_queries_json(response_text, query, max_queries=DEFAULT_SUBQUERY_COUNT):
    json_match = re.search(r"\{[\s\S]*\}", response_text)
    if not json_match:
        return []

    try:
        payload = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return []

    raw_sub_queries = payload.get("sub_queries")
    if not isinstance(raw_sub_queries, list):
        return []

    cleaned_sub_queries = []
    seen_queries = set()
    for item in raw_sub_queries:
        if not isinstance(item, str):
            continue
        cleaned = item.strip()
        if not is_valid_retrieval_query(cleaned, original_query=query):
            continue
        dedupe_key = cleaned.lower()
        if dedupe_key in seen_queries:
            continue
        cleaned_sub_queries.append(cleaned)
        seen_queries.add(dedupe_key)
        if len(cleaned_sub_queries) >= max_queries:
            break

    return cleaned_sub_queries


def is_valid_retrieval_query(candidate, *, original_query):
    if not candidate:
        return False

    normalized = candidate.strip()
    lowered = normalized.lower()
    if not normalized or lowered == original_query.lower():
        return False

    blocked_prefixes = (
        "i'm sorry",
        "i am sorry",
        "please provide",
        "no retrieval queries needed",
        "the query",
    )
    blocked_phrases = (
        "does not contain enough information",
        "does not contain any information",
        "please provide a specific question",
        "please provide a topic",
        "is a greeting",
        "cannot be converted into retrieval queries",
        "no retrieval queries needed",
    )

    if lowered.startswith(blocked_prefixes):
        return False

    return not any(phrase in lowered for phrase in blocked_phrases)


def collect_candidate_documents(
    vectorstore,
    queries,
    bm25_index=None,
    candidate_k=DEFAULT_RERANK_CANDIDATE_K,
    bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
    rrf_k=DEFAULT_RRF_K,
    metadata_filter=None,
    include_debug=False,
):
    fused_rankings = []
    debug = {
        "dense_hits": {},
        "bm25_hits": {},
        "dense_total_hits": 0,
        "bm25_total_hits": 0,
    }
    chroma_filter = build_chroma_filter(metadata_filter)
    dense_fetch_k = candidate_k * 4 if metadata_filter else candidate_k

    for query in queries:
        dense_docs = filter_documents_by_metadata(
            deduplicate_documents(
                vectorstore.similarity_search_with_score(
                    query,
                    k=dense_fetch_k,
                    filter=chroma_filter,
                )
            ),
            metadata_filter,
        )
        dense_docs = dense_docs[:candidate_k]
        fused_rankings.append(dense_docs)
        if include_debug:
            debug["dense_hits"][query] = build_hit_debug(dense_docs)
            debug["dense_total_hits"] += len(dense_docs)

        if bm25_index is not None:
            bm25_docs = retrieve_bm25_documents(
                query,
                bm25_index,
                top_k=bm25_candidate_k,
                metadata_filter=metadata_filter,
            )
            fused_rankings.append(bm25_docs)
            if include_debug:
                debug["bm25_hits"][query] = build_hit_debug(bm25_docs)
                debug["bm25_total_hits"] += len(bm25_docs)

    fused_documents = fuse_ranked_documents(fused_rankings, rrf_k=rrf_k)
    if include_debug:
        debug["fused_hits"] = build_hit_debug(fused_documents)
    return fused_documents, debug


def load_bm25_index(chunk_registry=None, vectorstore=None):
    documents = load_corpus_documents(chunk_registry=chunk_registry, vectorstore=vectorstore)
    tokenized_documents = [tokenize_for_bm25(doc.page_content) for doc in documents]

    if not tokenized_documents:
        return None

    return {
        "documents": documents,
        "tokenized_documents": tokenized_documents,
        "retriever": BM25Okapi(tokenized_documents),
    }


def load_corpus_documents(chunk_registry=None, vectorstore=None):
    documents = documents_from_registry(chunk_registry or {})
    if documents:
        return documents
    if vectorstore is not None:
        return documents_from_vectorstore(vectorstore)
    return []


def documents_from_registry(chunk_registry):
    documents = []
    records = chunk_registry.get("by_chunk_id", {})

    for chunk_id, record in records.items():
        document = _document_from_record(record)
        if document is None:
            continue
        if document.metadata.get("chunk_id") is None:
            document.metadata["chunk_id"] = chunk_id
        documents.append(document)

    documents.sort(key=lambda doc: doc.metadata.get("chunk_id", ""))
    return documents


def documents_from_vectorstore(vectorstore):
    collection = vectorstore._collection.get(include=["documents", "metadatas"])
    documents = []

    for page_content, metadata in zip(collection.get("documents", []), collection.get("metadatas", [])):
        documents.append(
            Document(
                page_content=page_content,
                metadata=metadata or {},
            )
        )

    return documents


def tokenize_for_bm25(text):
    return re.findall(r"\w+", text.lower())


def retrieve_bm25_documents(
    query,
    bm25_index,
    top_k=DEFAULT_BM25_CANDIDATE_K,
    metadata_filter=None,
):
    tokens = tokenize_for_bm25(query)
    if not tokens:
        return []

    scores = bm25_index["retriever"].get_scores(tokens)
    ranked_indexes = sorted(
        range(len(scores)),
        key=lambda index: scores[index],
        reverse=True,
    )

    documents = []
    for index in ranked_indexes[:top_k]:
        score = float(scores[index])
        if score <= 0:
            continue
        doc = bm25_index["documents"][index]
        if not _matches_metadata_filter(doc, metadata_filter):
            continue
        documents.append((doc, score))

    return documents


def _matches_metadata_filter(doc, metadata_filter):
    if not metadata_filter:
        return True

    for key, expected_value in metadata_filter.items():
        if key == "page_range":
            page_value = doc.metadata.get("page")
            if page_value is None:
                return False
            page_number = int(page_value) + 1
            start = expected_value.get("start")
            end = expected_value.get("end")
            if start is not None and page_number < start:
                return False
            if end is not None and page_number > end:
                return False
            continue
        if doc.metadata.get(key) != expected_value:
            return False
    return True


def filter_documents_by_metadata(documents, metadata_filter):
    if not metadata_filter:
        return documents

    filtered_documents = []
    for doc, score in documents:
        if _matches_metadata_filter(doc, metadata_filter):
            filtered_documents.append((doc, score))
    return filtered_documents


def build_chroma_filter(metadata_filter):
    if not metadata_filter:
        return None

    chroma_filter = {}
    for key, value in metadata_filter.items():
        if key == "page_range":
            continue
        chroma_filter[key] = value

    return chroma_filter or None


def fuse_ranked_documents(rankings, rrf_k=DEFAULT_RRF_K):
    fused_scores = {}
    fused_documents = {}

    for ranking in rankings:
        for rank, (doc, _) in enumerate(ranking, start=1):
            chunk_id = doc.metadata.get("chunk_id") or doc.page_content.strip()
            fused_documents[chunk_id] = doc
            fused_scores[chunk_id] = fused_scores.get(chunk_id, 0.0) + 1.0 / (rrf_k + rank)

    rerank_candidates = []
    for chunk_id, fused_score in sorted(fused_scores.items(), key=lambda item: item[1], reverse=True):
        doc = fused_documents[chunk_id]
        doc.metadata["fusion_score"] = fused_score
        rerank_candidates.append((doc, fused_score))

    return rerank_candidates


def build_hit_debug(documents):
    hits = []
    for doc, score in documents:
        hits.append(
            {
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "source": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page"),
                "score": round(float(score), 4),
                "preview": doc.page_content[:180].replace("\n", " "),
            }
        )
    return hits


def load_reranker(model_name):
    return CrossEncoder(model_name)


def rerank_documents(query, documents, reranker, top_n):
    if not documents:
        return []

    pairs = [(query, doc.page_content) for doc, _ in documents]
    rerank_scores = reranker.predict(pairs)

    reranked = []
    for (doc, original_score), rerank_score in zip(documents, rerank_scores):
        reranked.append((doc, float(rerank_score), float(original_score)))

    reranked.sort(key=lambda item: item[1], reverse=True)

    top_documents = []
    for doc, rerank_score, original_score in reranked[:top_n]:
        doc.metadata["rerank_score"] = rerank_score
        if doc.metadata.get("retrieval_score") is None:
            doc.metadata["retrieval_score"] = original_score
        top_documents.append((doc, rerank_score))

    return top_documents


def load_chunk_registry(registry_path=REGISTRY_PATH):
    registry_file = Path(registry_path)
    if not registry_file.exists():
        return {"by_document": {}, "by_chunk_id": {}}

    return json.loads(registry_file.read_text(encoding="utf-8"))


def _document_from_record(record):
    if not record:
        return None

    return Document(
        page_content=record.get("page_content", ""),
        metadata=record.get("metadata", {}),
    )


def expand_with_context_window(
    documents,
    chunk_registry,
    window_size=DEFAULT_CONTEXT_WINDOW,
    max_expanded_chunks=DEFAULT_MAX_EXPANDED_CHUNKS,
):
    """
    Expand each retrieved chunk with neighbors from the same document.
    """

    if window_size <= 0:
        return documents

    # The registry gives us O(1)-style neighbor lookup by document and chunk index.
    # Chroma finds the relevant chunk; the registry finds the chunks around it.
    by_document = chunk_registry.get("by_document", {})
    expanded_documents = []
    seen_chunk_ids = set()

    for hit_order, (doc, score) in enumerate(documents):
        hit_group = _expand_single_hit(
            doc=doc,
            score=score,
            hit_order=hit_order,
            by_document=by_document,
            seen_chunk_ids=seen_chunk_ids,
            window_size=window_size,
        )
        expanded_documents.extend(hit_group)

        if len(expanded_documents) >= max_expanded_chunks:
            break

    # Final ordering preserves which retrieval hit came first, then keeps chunks
    # in document order inside that hit's local window.
    expanded_documents.sort(key=_hit_order_key)
    return expanded_documents[:max_expanded_chunks]


def _expand_single_hit(doc, score, hit_order, by_document, seen_chunk_ids, window_size):
    document_id = doc.metadata.get("document_id")
    chunk_index = doc.metadata.get("chunk_index")

    if document_id is None or chunk_index is None:
        return _include_unindexed_hit(doc, score, hit_order, seen_chunk_ids)

    document_chunks = by_document.get(document_id, {})
    center_index = int(chunk_index)
    hit_group = []

    # Build a local window around the matched chunk so the LLM sees nearby
    # supporting text instead of one isolated chunk.
    for neighbor_index in range(center_index - window_size, center_index + window_size + 1):
        neighbor = _load_neighbor_chunk(document_chunks, neighbor_index)
        if neighbor is None:
            continue

        neighbor_chunk_id = neighbor.metadata.get("chunk_id")
        if not neighbor_chunk_id or neighbor_chunk_id in seen_chunk_ids:
            continue

        _mark_window_metadata(
            neighbor=neighbor,
            original_doc=doc,
            hit_order=hit_order,
            center_index=center_index,
            neighbor_index=neighbor_index,
        )
        hit_group.append((neighbor, score))
        seen_chunk_ids.add(neighbor_chunk_id)

    # Keep each hit group ordered like the original document: previous, hit, next.
    hit_group.sort(key=_document_position_key)
    return hit_group


def _include_unindexed_hit(doc, score, hit_order, seen_chunk_ids):
    chunk_id = doc.metadata.get("chunk_id")
    if not chunk_id or chunk_id in seen_chunk_ids:
        return []

    _mark_window_metadata(
        neighbor=doc,
        original_doc=doc,
        hit_order=hit_order,
        center_index=0,
        neighbor_index=0,
    )
    seen_chunk_ids.add(chunk_id)
    return [(doc, score)]


def _load_neighbor_chunk(document_chunks, neighbor_index):
    neighbor_record = document_chunks.get(str(neighbor_index))
    if neighbor_record is None:
        return None
    return _document_from_record(neighbor_record)


def _mark_window_metadata(neighbor, original_doc, hit_order, center_index, neighbor_index):
    neighbor.metadata["is_retrieved_hit"] = (
        neighbor.metadata.get("chunk_id") == original_doc.metadata.get("chunk_id")
    )
    neighbor.metadata["hit_order"] = hit_order
    neighbor.metadata["window_offset"] = neighbor_index - center_index


def _document_position_key(item):
    doc, _ = item
    return (
        doc.metadata.get("document_id", ""),
        int(doc.metadata.get("chunk_index", -1)),
    )


def _hit_order_key(item):
    doc, _ = item
    return (
        int(doc.metadata.get("hit_order", 9999)),
        doc.metadata.get("document_id", ""),
        int(doc.metadata.get("chunk_index", -1)),
    )


def format_context(documents):
    """
    Format retrieved chunks into one context string with source numbers.
    """

    parts = []
    for index, (doc, score) in enumerate(documents, start=1):
        file_name = doc.metadata.get("file_name", "unknown")
        page = doc.metadata.get("page")
        chunk_id = doc.metadata.get("chunk_id", "unknown")
        document_id = doc.metadata.get("document_id", "unknown")
        page_label = page if page is not None else "n/a"
        is_retrieved_hit = doc.metadata.get("is_retrieved_hit", False)
        window_offset = doc.metadata.get("window_offset", 0)
        rerank_score = doc.metadata.get("rerank_score")

        score_label = rerank_score if rerank_score is not None else score
        parts.append(
            (
                f"Source [{index}]\n"
                f"File: {file_name}\n"
                f"Document ID: {document_id}\n"
                f"Chunk ID: {chunk_id}\n"
                f"Page: {page_label}\n"
                f"Matched Hit: {'yes' if is_retrieved_hit else 'no'}\n"
                f"Window Offset: {window_offset}\n"
                f"Score: {score_label}\n"
                f"Content: {doc.page_content}"
            )
        )
    return "\n\n".join(parts)


def build_citation_sources(documents):
    citation_sources = []

    for index, (doc, score) in enumerate(documents, start=1):
        citation_sources.append(
            {
                "number": index,
                "source": doc.metadata.get("file_name", "Unknown"),
                "page": doc.metadata.get("page"),
                "chunk_id": doc.metadata.get("chunk_id", "unknown"),
                "retrieval_score": doc.metadata.get("retrieval_score", score),
                "rerank_score": doc.metadata.get("rerank_score", score),
                "content": doc.page_content,
            }
        )

    return citation_sources


def check_grounding_evidence(
    retrieved_documents,
    expanded_documents,
    min_rerank_score=DEFAULT_MIN_GROUNDED_RERANK_SCORE,
    min_expanded_chunks=DEFAULT_MIN_GROUNDED_CHUNKS,
):
    if not retrieved_documents:
        return {
            "passed": False,
            "reason": "no_retrieved_chunks",
            "top_rerank_score": None,
            "retrieved_count": 0,
            "expanded_count": len(expanded_documents),
        }

    if len(expanded_documents) < min_expanded_chunks:
        return {
            "passed": False,
            "reason": "too_few_expanded_chunks",
            "top_rerank_score": _top_rerank_score(retrieved_documents),
            "retrieved_count": len(retrieved_documents),
            "expanded_count": len(expanded_documents),
        }

    top_rerank_score = _top_rerank_score(retrieved_documents)
    if top_rerank_score is None or top_rerank_score < min_rerank_score:
        return {
            "passed": False,
            "reason": "low_rerank_score",
            "top_rerank_score": top_rerank_score,
            "retrieved_count": len(retrieved_documents),
            "expanded_count": len(expanded_documents),
        }

    return {
        "passed": True,
        "reason": "passed",
        "top_rerank_score": top_rerank_score,
        "retrieved_count": len(retrieved_documents),
        "expanded_count": len(expanded_documents),
    }


def generate_answer(query, context, llm):
    """
    Generate an answer from the retrieved context.
    """

    if not context.strip():
        return "No relevant information found in the documents."

    if llm is None:
        return "Language model is not available."

    valid_sources = extract_valid_source_numbers(context)
    structured_llm = get_structured_answer_llm(llm)
    if structured_llm is not None:
        try:
            structured_response = structured_llm.invoke(build_rag_prompt(context, query))
        except Exception:
            structured_response = None
        structured_answer = extract_structured_answer(structured_response)
        if structured_answer:
            if answer_has_valid_citations(structured_answer, valid_sources):
                return structured_answer

            repaired_answer = repair_answer_with_citations(
                query=query,
                context=context,
                draft_answer=structured_answer,
                llm=llm,
                valid_sources=valid_sources,
            )
            if repaired_answer:
                return repaired_answer

    response = llm.invoke(build_rag_prompt(context, query))
    answer = extract_response_text(response)

    if not answer:
        return "No response from language model."

    if answer_has_valid_citations(answer, valid_sources):
        return answer

    repaired_answer = repair_answer_with_citations(
        query=query,
        context=context,
        draft_answer=answer,
        llm=llm,
        valid_sources=valid_sources,
    )
    if repaired_answer:
        return repaired_answer

    return answer


def repair_answer_with_citations(query, context, draft_answer, llm, valid_sources):
    if not valid_sources:
        return draft_answer

    repair_prompt = build_citation_repair_prompt(
        context=context,
        question=query,
        answer=draft_answer,
        valid_sources=valid_sources,
    )
    repaired_response = llm.invoke(repair_prompt)
    repaired_answer = extract_response_text(repaired_response)

    if not repaired_answer:
        return None

    if answer_has_valid_citations(repaired_answer, valid_sources):
        return repaired_answer

    return None


def get_structured_answer_llm(llm):
    with_structured_output = getattr(llm, "with_structured_output", None)
    if with_structured_output is None:
        return None

    try:
        return with_structured_output(AnswerDraftResult)
    except Exception:
        return None


def extract_structured_answer(response):
    if response is None:
        return ""

    if isinstance(response, AnswerDraftResult):
        return response.answer.strip()

    if isinstance(response, dict):
        return str(response.get("answer", "")).strip()

    response_answer = getattr(response, "answer", None)
    if response_answer is not None:
        return str(response_answer).strip()

    return extract_response_text(response)


def extract_valid_source_numbers(context):
    numbers = re.findall(r"Source \[(\d+)\]", context)
    return sorted({int(number) for number in numbers})


def extract_citation_numbers(answer):
    numbers = re.findall(r"\[(\d+)\]", answer)
    return [int(number) for number in numbers]


def answer_has_valid_citations(answer, valid_sources):
    citation_numbers = extract_citation_numbers(answer)
    if not citation_numbers:
        return False

    valid_source_set = set(valid_sources)
    return all(number in valid_source_set for number in citation_numbers)


def _top_rerank_score(documents):
    doc, score = documents[0]
    rerank_score = doc.metadata.get("rerank_score")
    if rerank_score is None:
        return None
    return float(rerank_score if rerank_score is not None else score)
