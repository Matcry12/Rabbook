from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, File, Request, UploadFile
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
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_EXPANDED_CHUNKS,
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
from rag.ingest import add_documents_to_vectorstore
from rag.registry import load_chunk_registry
from rag.retrieve import (
    build_hit_debug,
    expand_with_context_window,
    format_context,
    generate_answer,
    load_bm25_index,
    load_reranker,
    load_vectorstore,
    retrieve_documents_with_query_transform,
)


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
        "sources": [],
        "debug_mode": False,
        "debug_data": None,
        "message": None,
        "error": None,
    }
    page_context.update(context)
    return templates.TemplateResponse("index.html", page_context)


def answer_query(query, debug_mode=False):
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
        include_debug=debug_mode,
    )
    if debug_mode:
        retrieved_documents, debug_data = retrieval_result
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
    context = format_context(expanded_documents)
    answer = generate_answer(query, context, get_llm())
    return answer, build_sources(retrieved_documents), debug_data


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


def _format_score(score):
    if score is None:
        return "n/a"
    return f"{float(score):.4f}"


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


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return render_home(request)


@app.post("/ask", response_class=HTMLResponse)
async def ask(request: Request):
    form = await request.form()
    query = str(form.get("query", "")).strip()
    debug_mode = str(form.get("debug_mode", "")).lower() in {"on", "true", "1"}

    if not query:
        return render_home(request, error="Please enter a question.", debug_mode=debug_mode)

    try:
        answer, sources, debug_data = answer_query(query, debug_mode=debug_mode)
    except Exception as exc:
        return render_home(request, query=query, error=str(exc), debug_mode=debug_mode)

    return render_home(
        request,
        answer=answer,
        query=query,
        sources=sources,
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
