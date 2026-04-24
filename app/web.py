import json
from functools import lru_cache
from typing import Any

from app.actions import (
    delete_document as run_delete_document,
    get_upload_target,
    ingest_uploaded_document,
    ingest_url_document,
    parse_saved_citations,
    save_history_item,
    save_uploaded_document,
)
from app.exports import (
    export_answer_json_response,
    export_answer_markdown_response,
    export_history_json_response,
    export_history_markdown_response,
    export_notes_json_response,
    export_notes_markdown_response,
)
from app.runtime import (
    get_bm25_index as load_runtime_bm25_index,
    get_chunk_registry as load_runtime_chunk_registry,
    get_vectorstore as load_runtime_vectorstore,
    refresh_runtime_state as refresh_cached_runtime_state,
)
from app.view_data import (
    get_available_file_types as load_available_file_types,
    get_available_files as load_available_files,
    get_history_items as load_history_items,
    get_library_documents as load_library_documents,
    get_saved_notes as load_saved_notes,
)
from agents.services import answer_query as run_answer_query
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_ollama import ChatOllama

from core.config import (
    DB_DIR,
    DEFAULT_CONTEXT_WINDOW,
    DEFAULT_BM25_CANDIDATE_K,
    DEFAULT_ENABLE_QUERY_TRANSFORM,
    ENABLE_LANGGRAPH_AGENT,
    DEFAULT_ENABLE_RESEARCH_FALLBACK,
    DEFAULT_GROUNDED_FALLBACK_MESSAGE,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_GPU,
    OLLAMA_THINKING_MODE,
    DEFAULT_MAX_EXPANDED_CHUNKS,
    DEFAULT_MIN_GROUNDED_CHUNKS,
    DEFAULT_MIN_GROUNDED_RERANK_SCORE,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RETRIEVAL_K,
    REGISTRY_PATH,
    STATIC_DIR,
    TEMPLATES_DIR,
    get_google_api_key,
)
from rag.ingest import (
    add_documents_to_vectorstore,
    reingest_directory,
)
from rag.history import clear_history, delete_history_entry, get_history_entry
from rag.notes import clear_notes, delete_note, save_note
from rag.registry import (
    rebuild_chunk_registry_from_vectorstore,
)
from rag.retrieve import (
    load_reranker,
)
app = FastAPI(title="Rabbook")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))
load_dotenv()  # Load environment variables from .env file at startup


class PromptTransformRunnable:
    def __init__(self, runnable, transform_prompt):
        self._runnable = runnable
        self._transform_prompt = transform_prompt

    def invoke(self, input_value, *args, **kwargs):
        return self._runnable.invoke(self._transform_prompt(input_value), *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._runnable, name)


class GemmaPromptWrapper:
    def __init__(self, llm, model_name):
        self._llm = llm
        self._model_name = model_name or ""

    def _transform_prompt(self, input_value: Any):
        if not isinstance(input_value, str):
            return input_value
        if "gemma" not in self._model_name.lower():
            return input_value
        if "<thought off>" in input_value.lower():
            return input_value
        return f"<thought off>\n{input_value}"

    def invoke(self, input_value, *args, **kwargs):
        return self._llm.invoke(self._transform_prompt(input_value), *args, **kwargs)

    def with_structured_output(self, *args, **kwargs):
        runnable = self._llm.with_structured_output(*args, **kwargs)
        return PromptTransformRunnable(runnable, self._transform_prompt)

    def __getattr__(self, name):
        return getattr(self._llm, name)

@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def get_llm():
    if DEFAULT_LLM_PROVIDER == "groq":
        llm = ChatGroq(
            model=DEFAULT_LLM_MODEL,
            temperature=0.3,
        )
        return GemmaPromptWrapper(llm, DEFAULT_LLM_MODEL)
    elif DEFAULT_LLM_PROVIDER == "gemini":
        api_key = get_google_api_key()
        if not api_key:
            raise RuntimeError("GEMINI_KEY is missing in .env.")
        llm = ChatGoogleGenerativeAI(
            model=DEFAULT_LLM_MODEL,
            google_api_key=api_key,
            temperature=0.3,
        )
        return GemmaPromptWrapper(llm, DEFAULT_LLM_MODEL)
    elif DEFAULT_LLM_PROVIDER == "ollama":
        # num_gpu: 0 forces CPU, -1 (default) uses GPU if available
        llm = ChatOllama(
            model=DEFAULT_LLM_MODEL,
            base_url=OLLAMA_BASE_URL,
            num_gpu=OLLAMA_NUM_GPU,
            temperature=0.3,
        )
        return GemmaPromptWrapper(llm, DEFAULT_LLM_MODEL)
    else:
        raise ValueError(f"Unsupported LLM provider: {DEFAULT_LLM_PROVIDER}")


def strip_thinking(text: str) -> str:
    """Removes <think>...</think> blocks from LLM responses."""
    import re
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


def get_bm25_index():
    return load_runtime_bm25_index(
        app,
        get_chunk_registry=get_chunk_registry,
        get_vectorstore=get_vectorstore,
    )


def get_vectorstore():
    return load_runtime_vectorstore(app, get_embeddings=get_embeddings)


def get_chunk_registry():
    return load_runtime_chunk_registry(app)


def refresh_runtime_state():
    refresh_cached_runtime_state(app, get_embeddings=get_embeddings)


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
        "history_items": get_history_items(),
        "saved_notes": get_saved_notes(),
        "sources": [],
        "citations": [],
        "debug_mode": False,
        "debug_data": None,
        "message": None,
        "error": None,
    }
    page_context.update(context)
    return templates.TemplateResponse(request, "index.html", page_context)


def answer_query(
    query,
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
    debug_mode=False,
):
    result = run_answer_query(
        query,
        vectorstore=get_vectorstore(),
        chunk_registry=get_chunk_registry(),
        reranker=get_reranker(),
        bm25_index=get_bm25_index(),
        llm=get_llm(),
        retrieval_k=DEFAULT_RETRIEVAL_K,
        rerank_candidate_k=DEFAULT_RERANK_CANDIDATE_K,
        bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
        context_window=DEFAULT_CONTEXT_WINDOW,
        max_expanded_chunks=DEFAULT_MAX_EXPANDED_CHUNKS,
        min_grounded_rerank_score=DEFAULT_MIN_GROUNDED_RERANK_SCORE,
        min_grounded_chunks=DEFAULT_MIN_GROUNDED_CHUNKS,
        grounded_fallback_message=DEFAULT_GROUNDED_FALLBACK_MESSAGE,
        enable_query_transform=DEFAULT_ENABLE_QUERY_TRANSFORM,
        selected_file=selected_file,
        selected_file_type=selected_file_type,
        page_start=page_start,
        page_end=page_end,
        debug_mode=debug_mode,
        use_langgraph=ENABLE_LANGGRAPH_AGENT,
        enable_research=DEFAULT_ENABLE_RESEARCH_FALLBACK,
    )
    
    answer = result.answer
    if DEFAULT_LLM_PROVIDER == "ollama" and not OLLAMA_THINKING_MODE:
        answer = strip_thinking(answer)
        
    return answer, result.sources, result.citations, result.debug_data


def get_available_files():
    return load_available_files(get_chunk_registry())


def get_available_file_types():
    return load_available_file_types(get_chunk_registry())


def get_library_documents():
    return load_library_documents(REGISTRY_PATH)


def get_saved_notes():
    return load_saved_notes()


def get_history_items():
    return load_history_items()


@lru_cache(maxsize=1)
def get_reranker():
    return load_reranker(DEFAULT_RERANK_MODEL)


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

    save_history_item(
        query=query,
        answer=answer,
        citations=citations,
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
        ingest_uploaded_document(
            target_path,
            add_documents_to_vectorstore=add_documents_to_vectorstore,
            get_embeddings=get_embeddings,
            refresh_runtime_state=refresh_runtime_state,
        )
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
        payload = ingest_url_document(
            target_url,
            add_documents_to_vectorstore=add_documents_to_vectorstore,
            get_embeddings=get_embeddings,
            refresh_runtime_state=refresh_runtime_state,
        )
    except ValueError as exc:
        return render_home(request, error=str(exc))
    except Exception as exc:
        return render_home(request, error=str(exc))

    title = payload.get("title") or payload.get("file_name", "URL page")
    return render_home(request, message=f"Imported {title} from URL.")


@app.post("/documents/{document_id}/delete", response_class=HTMLResponse)
async def delete_document_route(request: Request, document_id: str):
    try:
        deleted_document = run_delete_document(
            document_id,
            get_library_documents=get_library_documents,
            get_vectorstore=get_vectorstore,
            refresh_runtime_state=refresh_runtime_state,
        )
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


@app.get("/export/notes.md")
async def export_notes_markdown():
    return export_notes_markdown_response(get_saved_notes())


@app.get("/export/notes.json")
async def export_notes_json():
    return export_notes_json_response(get_saved_notes())


@app.post("/history/{history_id}/delete", response_class=HTMLResponse)
async def delete_history_route(request: Request, history_id: str):
    try:
        delete_history_entry(history_id)
    except ValueError as exc:
        return render_home(request, error=str(exc))

    return render_home(request, message="Deleted history item.")


@app.post("/history/{history_id}/notes", response_class=HTMLResponse)
async def save_history_to_notes_route(request: Request, history_id: str):
    try:
        item = get_history_entry(history_id)
        save_note(
            query=item.get("query", ""),
            answer=item.get("answer", ""),
            citations=item.get("citations", []),
        )
    except ValueError as exc:
        return render_home(request, error=str(exc))

    return render_home(request, message="Saved history item as note.")


@app.get("/export/history.md")
async def export_history_markdown():
    return export_history_markdown_response(get_history_items())


@app.get("/export/history.json")
async def export_history_json():
    return export_history_json_response(get_history_items())


@app.post("/export/answer.md")
async def export_answer_markdown(
    query: str = Form(...),
    answer: str = Form(...),
    citations_json: str = Form(""),
):
    return export_answer_markdown_response(
        query,
        answer,
        parse_saved_citations(citations_json.strip()),
    )


@app.post("/export/answer.json")
async def export_answer_json(
    query: str = Form(...),
    answer: str = Form(...),
    citations_json: str = Form(""),
):
    return export_answer_json_response(
        query,
        answer,
        parse_saved_citations(citations_json.strip()),
    )


@app.post("/maintenance/refresh", response_class=HTMLResponse)
async def refresh_runtime_route(request: Request):
    try:
        refresh_runtime_state()
    except Exception as exc:
        return render_home(request, error=str(exc))

    return render_home(request, message="Refreshed runtime state.")


@app.post("/maintenance/registry/rebuild", response_class=HTMLResponse)
async def rebuild_registry_route(request: Request):
    try:
        rebuilt_count = rebuild_chunk_registry_from_vectorstore(
            get_vectorstore(),
            str(REGISTRY_PATH),
        )
        refresh_runtime_state()
    except Exception as exc:
        return render_home(request, error=str(exc))

    return render_home(request, message=f"Rebuilt chunk registry from {rebuilt_count} chunks.")


@app.post("/maintenance/uploads/reingest", response_class=HTMLResponse)
async def reingest_uploads_route(request: Request):
    try:
        result = reingest_directory(
            str(UPLOAD_DIR),
            get_embeddings(),
            str(DB_DIR),
            str(REGISTRY_PATH),
        )
        refresh_runtime_state()
    except Exception as exc:
        return render_home(request, error=str(exc))

    return render_home(
        request,
        message=(
            f"Re-ingested uploads: {result['document_count']} documents, "
            f"{result['chunk_count']} chunks."
        ),
    )


@app.post("/maintenance/history/clear", response_class=HTMLResponse)
async def clear_history_route(request: Request):
    clear_history()
    return render_home(request, message="Cleared chat history.")


@app.post("/maintenance/notes/clear", response_class=HTMLResponse)
async def clear_notes_route(request: Request):
    clear_notes()
    return render_home(request, message="Cleared saved notes.")
