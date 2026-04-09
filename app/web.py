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
    DEFAULT_LLM_MODEL,
    DEFAULT_MAX_EXPANDED_CHUNKS,
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
    expand_with_context_window,
    format_context,
    generate_answer,
    load_vectorstore,
    retrieve_documents,
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


def render_home(request: Request, **context):
    page_context = {
        "request": request,
        "answer": None,
        "query": "",
        "sources": [],
        "message": None,
        "error": None,
    }
    page_context.update(context)
    return templates.TemplateResponse("index.html", page_context)


def answer_query(query):
    documents = retrieve_documents(get_vectorstore(), query, k=DEFAULT_RETRIEVAL_K)
    documents = expand_with_context_window(
        documents,
        get_chunk_registry(),
        window_size=DEFAULT_CONTEXT_WINDOW,
        max_expanded_chunks=DEFAULT_MAX_EXPANDED_CHUNKS,
    )
    context = format_context(documents)
    answer = generate_answer(query, context, get_llm())
    return answer, build_sources(documents)


def build_sources(documents):
    return [
        {
            "source": doc.metadata.get("file_name", "Unknown"),
            "page": doc.metadata.get("page"),
            "chunk_id": doc.metadata.get("chunk_id", "unknown"),
            "score": f"{score:.4f}",
            "content": doc.page_content,
        }
        for doc, score in documents
    ]


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

    if not query:
        return render_home(request, error="Please enter a question.")

    try:
        answer, sources = answer_query(query)
    except Exception as exc:
        return render_home(request, query=query, error=str(exc))

    return render_home(request, answer=answer, query=query, sources=sources)


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
