import json
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

from core.config import (
    DB_DIR,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_BM25_CANDIDATE_K,
    DEFAULT_ENABLE_QUERY_TRANSFORM,
    DEFAULT_GROUNDED_FALLBACK_MESSAGE,
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_EXPANDED_CHUNKS,
    DEFAULT_MIN_GROUNDED_CHUNKS,
    DEFAULT_MIN_GROUNDED_RERANK_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RETRIEVAL_K,
    REGISTRY_PATH,
    STATIC_DIR,
    SUPPORTED_EXTENSIONS,
    TEMPLATES_DIR,
    UPLOAD_DIR,
    get_google_api_key,
)
from rag.ingest import add_documents_to_vectorstore, add_loaded_documents_to_vectorstore
from rag.notes import delete_note, load_notes, save_note
from rag.registry import delete_document_from_registry, list_documents, load_chunk_registry
from rag.retrieve import (
    answer_has_valid_citations,
    build_hit_debug,
    build_citation_sources,
    check_grounding_evidence,
    expand_with_context_window,
    extract_valid_source_numbers,
    extract_citation_numbers,
    format_context,
    generate_answer,
    load_bm25_index,
    load_reranker,
    load_vectorstore,
    retrieve_documents_with_query_transform,
)
from rag.web_ingest import load_url_document


app = FastAPI(title="Rabbook")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
load_dotenv()  # Load environment variables from .env file at startup

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_llm() -> ChatGoogleGenerativeAI:
    api_key = get_google_api_key()
    if not api_key:
        raise RuntimeError("GEMINI_KEY is missing. Add it to .env before starting the app.")

    return ChatGroq(
        model=DEFAULT_LLM_MODEL,
        temperature=0.3,
    )


@lru_cache(maxsize=1)
def get_reranker():
    return load_reranker(DEFAULT_RERANK_MODEL)


def get_bm25_index():
    bm25_index = getattr(app.state, "bm25_index", None)
    if bm25_index is None:
        bm25_index = load_bm25_index(
            chunk_registry=get_chunk_registry(),
            vectorstore=get_vectorstore(),
        )
        app.state.bm25_index = bm25_index
    return bm25_index


def get_vectorstore():
    vectorstore = getattr(app.state, "vectorstore", None)
    if vectorstore is None:
        vectorstore = load_vectorstore(str(DB_DIR), get_embeddings())
        app.state.vectorstore = vectorstore
    return vectorstore


def get_chunk_registry():
    chunk_registry = getattr(app.state, "chunk_registry", None)
    if chunk_registry is None:
        chunk_registry = load_chunk_registry(str(REGISTRY_PATH))
        app.state.chunk_registry = chunk_registry
    return chunk_registry


def refresh_runtime_state():
    app.state.vectorstore = load_vectorstore(str(DB_DIR), get_embeddings())
    app.state.chunk_registry = load_chunk_registry(str(REGISTRY_PATH))
    app.state.bm25_index = load_bm25_index(
        chunk_registry=app.state.chunk_registry,
        vectorstore=app.state.vectorstore,
    )


def render_home(request: Request, **context):
    page_context = {
        "request": request,
        "answer": None,
        "query": "",
        "selected_file": "",
        "selected_file_type": "",
        "page_start": "",
        "page_end": "",
        "available_files": get_available_files(),
        "available_file_types": get_available_file_types(),
        "library_documents": get_library_documents(),
        "saved_notes": get_saved_notes(),
        "sources": [],
        "citations": [],
        "debug_mode": False,
        "debug_data": None,
        "message": None,
        "error": None,
    }
    page_context.update(context)
    return templates.TemplateResponse("index.html", page_context)


def answer_query(
    query,
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
    debug_mode=False,
):
    metadata_filter = build_metadata_filter(
        selected_file=selected_file,
        selected_file_type=selected_file_type,
        page_start=page_start,
        page_end=page_end,
    )
    retrieval_result = retrieve_documents_with_query_transform(
        get_vectorstore(),
        query,
        k=DEFAULT_RETRIEVAL_K,
        reranker=get_reranker(),
        bm25_index=get_bm25_index(),
        query_transformer=get_llm(),
        enable_query_transform=DEFAULT_ENABLE_QUERY_TRANSFORM,
        candidate_k=DEFAULT_RERANK_CANDIDATE_K,
        bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
        metadata_filter=metadata_filter,
        include_debug=debug_mode,
    )
    if debug_mode:
        retrieved_documents, debug_data = retrieval_result
        debug_data["metadata_filter"] = metadata_filter
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
        get_chunk_registry(),
        window_size=DEFAULT_CONTEXT_WINDOW,
        max_expanded_chunks=DEFAULT_MAX_EXPANDED_CHUNKS,
    )
    if debug_mode:
        debug_data["expanded_hits"] = build_hit_debug(expanded_documents)
        debug_data["stage_counts"]["expanded_context"] = len(expanded_documents)
    grounding = check_grounding_evidence(
        retrieved_documents,
        expanded_documents,
        min_rerank_score=DEFAULT_MIN_GROUNDED_RERANK_SCORE,
        min_expanded_chunks=DEFAULT_MIN_GROUNDED_CHUNKS,
    )
    if debug_mode:
        debug_data["grounding"].update(grounding)
        debug_data["grounding"]["stage"] = "retrieval"

    if not grounding["passed"]:
        answer = DEFAULT_GROUNDED_FALLBACK_MESSAGE
        citations = []
        return answer, build_sources(retrieved_documents), citations, debug_data

    context = format_context(expanded_documents)
    answer = generate_answer(query, context, get_llm())
    if not _answer_is_grounded(answer, context):
        if debug_mode:
            debug_data["grounding"] = {
                "stage": "answer",
                "passed": False,
                "reason": "citation_validation_failed",
                "top_rerank_score": grounding["top_rerank_score"],
                "retrieved_count": grounding["retrieved_count"],
                "expanded_count": grounding["expanded_count"],
            }
        answer = DEFAULT_GROUNDED_FALLBACK_MESSAGE
        citations = []
        return answer, build_sources(retrieved_documents), citations, debug_data

    if debug_mode:
        debug_data["grounding"] = {
            "stage": "answer",
            "passed": True,
            "reason": "answer_is_grounded",
            "top_rerank_score": grounding["top_rerank_score"],
            "retrieved_count": grounding["retrieved_count"],
            "expanded_count": grounding["expanded_count"],
        }

    citations = build_citations(expanded_documents, answer)
    return answer, build_sources(retrieved_documents), citations, debug_data


def build_sources(documents):
    return [
        {
            "source": doc.metadata.get("file_name", "Unknown"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "retrieval_score": _format_score(doc.metadata.get("retrieval_score", score)),
            "rerank_score": _format_score(doc.metadata.get("rerank_score", score)),
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
        item["retrieval_score"] = _format_score(item.get("retrieval_score"))
        item["rerank_score"] = _format_score(item.get("rerank_score"))
        filtered_citations.append(item)

    return filtered_citations


def _format_score(score):
    if score is None:
        return "n/a"
    return f"{float(score):.4f}"


def _answer_is_grounded(answer, context):
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


def get_available_files():
    records = get_chunk_registry().get("by_chunk_id", {})
    file_names = {
        record.get("metadata", {}).get("file_name", "")
        for record in records.values()
        if record.get("metadata", {}).get("file_name")
    }
    return sorted(file_names)


def get_available_file_types():
    records = get_chunk_registry().get("by_chunk_id", {})
    file_types = {
        record.get("metadata", {}).get("file_type", "")
        for record in records.values()
        if record.get("metadata", {}).get("file_type")
    }
    return sorted(file_types)


def get_library_documents():
    return list_documents(str(REGISTRY_PATH))


def get_saved_notes():
    return load_notes()


def delete_document(document_id):
    document = next(
        (item for item in get_library_documents() if item["document_id"] == document_id),
        None,
    )
    if document is None:
        raise ValueError("Document not found.")

    # Delete by document metadata so all chunks for this file are removed together.
    get_vectorstore()._collection.delete(where={"document_id": document_id})
    delete_document_from_registry(document_id, str(REGISTRY_PATH))

    source_path = Path(document.get("source", ""))
    if source_path.exists() and source_path.parent == UPLOAD_DIR:
        source_path.unlink()

    refresh_runtime_state()
    return document


def get_upload_target(document: UploadFile):
    if not document.filename:
        raise ValueError("Please choose a file to upload.")

    filename = Path(document.filename).name
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError("Only PDF and TXT files are supported.")

    return filename, UPLOAD_DIR / filename


async def save_uploaded_document(document: UploadFile, target_path: Path):
    target_path.write_bytes(await document.read())
    await document.close()


def ingest_uploaded_document(target_path: Path):
    add_documents_to_vectorstore(str(target_path), get_embeddings(), str(DB_DIR))
    refresh_runtime_state()


def ingest_url_document(url: str):
    document = load_url_document(url)
    add_loaded_documents_to_vectorstore([document], get_embeddings(), str(DB_DIR))
    refresh_runtime_state()
    return document


def parse_saved_citations(citations_json):
    if not citations_json:
        return []

    citations = json.loads(citations_json)
    return citations if isinstance(citations, list) else []


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_home(request)


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request):
    form = await request.form()
    query = str(form.get("query", "")).strip()
    selected_file = str(form.get("selected_file", "")).strip()
    selected_file_type = str(form.get("selected_file_type", "")).strip()
    page_start = str(form.get("page_start", "")).strip()
    page_end = str(form.get("page_end", "")).strip()
    debug_mode = str(form.get("debug_mode", "")).lower() in {"on", "true", "1"}

    if not query:
        return render_home(
            request,
            error="Please enter a question.",
            debug_mode=debug_mode,
            selected_file=selected_file,
            selected_file_type=selected_file_type,
            page_start=page_start,
            page_end=page_end,
        )

    try:
        answer, sources, citations, debug_data = answer_query(
            query,
            selected_file=selected_file,
            selected_file_type=selected_file_type,
            page_start=page_start,
            page_end=page_end,
            debug_mode=debug_mode,
        )
    except ValueError as exc:
        return render_home(
            request,
            query=query,
            error=str(exc),
            debug_mode=debug_mode,
            selected_file=selected_file,
            selected_file_type=selected_file_type,
            page_start=page_start,
            page_end=page_end,
        )
    except Exception as exc:
        return render_home(
            request,
            query=query,
            error=str(exc),
            debug_mode=debug_mode,
            selected_file=selected_file,
            selected_file_type=selected_file_type,
            page_start=page_start,
            page_end=page_end,
        )

    return render_home(
        request,
        answer=answer,
        query=query,
        selected_file=selected_file,
        selected_file_type=selected_file_type,
        page_start=page_start,
        page_end=page_end,
        sources=sources,
        citations=citations,
        debug_mode=debug_mode,
        debug_data=debug_data,
    )


@app.post("/documents", response_class=HTMLResponse)
async def upload_document(request: Request, document: UploadFile = File(...)):
    try:
        filename, target_path = get_upload_target(document)
        await save_uploaded_document(document, target_path)
        ingest_uploaded_document(target_path)
    except ValueError as exc:
        return render_home(request, error=str(exc))
    except Exception as exc:
        return render_home(request, error=str(exc))

    return render_home(request, message=f"Added {filename} to the vector store.")


@app.post("/urls", response_class=HTMLResponse)
async def import_url(request: Request, url: str = Form(...)):
    target_url = url.strip()
    if not target_url:
        return render_home(request, error="Please enter a URL to import.")

    try:
        document = ingest_url_document(target_url)
    except ValueError as exc:
        return render_home(request, error=str(exc))
    except Exception as exc:
        return render_home(request, error=str(exc))

    title = document.metadata.get("title") or document.metadata.get("file_name", "URL page")
    return render_home(request, message=f"Imported {title} from URL.")


@app.post("/documents/{document_id}/delete", response_class=HTMLResponse)
async def delete_document_route(request: Request, document_id: str):
    try:
        deleted_document = delete_document(document_id)
    except ValueError as exc:
        return render_home(request, error=str(exc))
    except Exception as exc:
        return render_home(request, error=str(exc))

    return render_home(
        request,
        message=f"Deleted {deleted_document['file_name']} from the library.",
    )


@app.post("/notes", response_class=HTMLResponse)
async def save_note_route(request: Request):
    form = await request.form()
    query = str(form.get("query", "")).strip()
    answer = str(form.get("answer", "")).strip()
    citations_json = str(form.get("citations_json", "")).strip()

    if not query or not answer:
        return render_home(request, error="Only completed answers can be saved as notes.")

    try:
        save_note(
            query=query,
            answer=answer,
            citations=parse_saved_citations(citations_json),
        )
    except json.JSONDecodeError:
        return render_home(request, error="Could not save note because citation data was invalid.")

    return render_home(request, message="Saved note.")


@app.post("/notes/{note_id}/delete", response_class=HTMLResponse)
async def delete_note_route(request: Request, note_id: str):
    try:
        delete_note(note_id)
    except ValueError as exc:
        return render_home(request, error=str(exc))

    return render_home(request, message="Deleted note.")
