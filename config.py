import os
from pathlib import Path

from dotenv import load_dotenv


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
DB_DIR = BASE_DIR / "chroma_db"
REGISTRY_PATH = BASE_DIR / "chunk_registry.json"
TEMPLATES_DIR = BASE_DIR / "templates"
STATIC_DIR = BASE_DIR / "static"

load_dotenv(BASE_DIR / ".env")

SUPPORTED_EXTENSIONS = {".pdf", ".txt"}
DEFAULT_LLM_MODEL = os.getenv("RABBOOK_LLM_MODEL", "gemma-3-27b-it")
DEFAULT_HOST = os.getenv("RABBOOK_HOST", "0.0.0.0")
DEFAULT_PORT = int(os.getenv("RABBOOK_PORT", "6001"))
DEFAULT_RETRIEVAL_K = int(os.getenv("RABBOOK_RETRIEVAL_K", "4"))
DEFAULT_CONTEXT_WINDOW = int(os.getenv("RABBOOK_CONTEXT_WINDOW", "1"))
DEFAULT_MAX_EXPANDED_CHUNKS = int(os.getenv("RABBOOK_MAX_EXPANDED_CHUNKS", "12"))

UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def get_google_api_key() -> str | None:
    return os.getenv("GEMINI_KEY")
