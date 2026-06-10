"""
Shared helpers for the three evaluation scripts.

Centralises model construction, dataset loading, and retrieval so that
evaluate_retrieval_metrics.py, evaluate_ragas.py, and evaluate_agent.py
stay short and focused on their own logic.
"""
import json
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)

from core.config import (
    DB_DIR,
    DEFAULT_BM25_CANDIDATE_K,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RETRIEVAL_K,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_GPU,
    OLLAMA_THINKING_MODE,
    get_google_api_key,
)
from rag.registry import load_chunk_registry
from rag.retrieve import (
    load_bm25_index,
    load_vectorstore,
    retrieve_documents_with_query_transform,
)

# evaluation/ package directory and project root
PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_DIR.parent

DATASET_PATH = PACKAGE_DIR / "data" / "eval_dataset.json"


def load_dataset() -> list[dict]:
    """Load the golden evaluation dataset from tests/eval_dataset.json."""
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def build_llm():
    """Instantiate the RAG LLM based on RABBOOK_LLM_PROVIDER."""
    if DEFAULT_LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=DEFAULT_LLM_MODEL, temperature=0.0)
    if DEFAULT_LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.0)
    from langchain_ollama import ChatOllama
    # thinking=False suppresses <think> blocks (e.g. Gemma) — matches app/web.py
    ollama_kwargs = dict(
        model=DEFAULT_LLM_MODEL,
        base_url=OLLAMA_BASE_URL,
        num_gpu=OLLAMA_NUM_GPU,
        temperature=0.0,
    )
    if not OLLAMA_THINKING_MODE:
        ollama_kwargs["thinking"] = False
    return ChatOllama(**ollama_kwargs)


def build_embeddings():
    """Return a HuggingFace all-MiniLM-L6-v2 embeddings instance."""
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def build_reranker():
    """Return a CrossEncoder reranker using DEFAULT_RERANK_MODEL."""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(DEFAULT_RERANK_MODEL)


def build_evaluator_llm():
    """
    Return the LLM used for RAGAS judging.

    Prefers Gemini gemini-3.1-flash-lite for structured-output quality.
    Falls back to the RAG LLM when GEMINI_KEY is not set.
    Note: if RABBOOK_LLM_PROVIDER is ollama, set GEMINI_KEY for better results.
    """
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = get_google_api_key()
    if not api_key:
        print("Warning: GEMINI_KEY not set — falling back to RAG LLM for evaluation.")
        return build_llm()
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0, google_api_key=api_key)


def load_retrieval_bundle(embeddings) -> tuple:
    """
    Load and return (vectorstore, bm25_index) from disk.

    Both are built from the same chunk registry so BM25 and dense
    retrieval are always in sync with the current ingested documents.
    """
    vectorstore = load_vectorstore(str(DB_DIR), embeddings)
    chunk_registry = load_chunk_registry()
    bm25_index = load_bm25_index(chunk_registry=chunk_registry, vectorstore=vectorstore)
    return vectorstore, bm25_index


def retrieve_chunk_ids(
    question: str,
    vectorstore,
    bm25_index,
    reranker,
    llm,
    k: int = DEFAULT_RETRIEVAL_K,
) -> list[str]:
    """
    Run the retrieval pipeline for a single question and return chunk ids in rank order.

    Query transformation is disabled so results are deterministic across runs.
    """
    results = retrieve_documents_with_query_transform(
        vectorstore,
        question,
        k=k,
        reranker=reranker,
        bm25_index=bm25_index,
        query_transformer=llm,
        enable_query_transform=False,
        candidate_k=DEFAULT_RERANK_CANDIDATE_K,
        bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
        metadata_filter=None,
        include_debug=False,
    )
    return [doc.metadata.get("chunk_id") for doc, _score in results]
