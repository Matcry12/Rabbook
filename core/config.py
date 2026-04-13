import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = BASE_DIR / "chroma_db"
REGISTRY_PATH = BASE_DIR / "chunk_registry.json"
NOTES_PATH = BASE_DIR / "notes.json"
HISTORY_PATH = BASE_DIR / "history.json"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR / ".env")

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
DEFAULT_LLM_MODEL = os.getenv("RABBOOK_LLM_MODEL", "llama-3.1-8b-instant")
DEFAULT_HOST = os.getenv("RABBOOK_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("RABBOOK_PORT", "6001"))
DEFAULT_RETRIEVAL_K = int(os.getenv("RABBOOK_RETRIEVAL_K", "4"))
DEFAULT_RERANK_CANDIDATE_K = int(os.getenv("RABBOOK_RERANK_CANDIDATE_K", "8"))
DEFAULT_BM25_CANDIDATE_K = int(os.getenv("RABBOOK_BM25_CANDIDATE_K", "8"))
DEFAULT_RRF_K = int(os.getenv("RABBOOK_RRF_K", "60"))
DEFAULT_ENABLE_QUERY_TRANSFORM = os.getenv("RABBOOK_ENABLE_QUERY_TRANSFORM", "true").lower() == "true"
DEFAULT_SUBQUERY_COUNT = int(os.getenv("RABBOOK_SUBQUERY_COUNT", "3"))
DEFAULT_CONTEXT_WINDOW = int(os.getenv("RABBOOK_CONTEXT_WINDOW", "1"))
DEFAULT_MAX_EXPANDED_CHUNKS = int(os.getenv("RABBOOK_MAX_EXPANDED_CHUNKS", "12"))
DEFAULT_MIN_GROUNDED_RERANK_SCORE = float(
    os.getenv("RABBOOK_MIN_GROUNDED_RERANK_SCORE", "1.0")
)
DEFAULT_MIN_GROUNDED_CHUNKS = int(os.getenv("RABBOOK_MIN_GROUNDED_CHUNKS", "1"))
DEFAULT_GROUNDED_FALLBACK_MESSAGE = os.getenv(
    "RABBOOK_GROUNDED_FALLBACK_MESSAGE",
    "I don't have enough support in the current documents to answer that confidently.",
)
DEFAULT_CHUNK_SIZE = int(os.getenv("RABBOOK_CHUNK_SIZE", "1200"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("RABBOOK_CHUNK_OVERLAP", "150"))
DEFAULT_SEMANTIC_MIN_CHUNK_SIZE = int(os.getenv("RABBOOK_SEMANTIC_MIN_CHUNK_SIZE", "350"))
DEFAULT_SEMANTIC_PERCENTILE = int(os.getenv("RABBOOK_SEMANTIC_PERCENTILE", "85"))
DEFAULT_RERANK_MODEL = os.getenv(
    "RABBOOK_RERANK_MODEL",
    "cross-encoder/ms-marco-MiniLM-L-6-v2",
)

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_google_api_key() -> str | None:
    return os.getenv("GEMINI_KEY")
