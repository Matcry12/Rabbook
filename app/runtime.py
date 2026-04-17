from core.config import DB_DIR, REGISTRY_PATH
from rag.registry import load_chunk_registry
from rag.retrieve import load_bm25_index, load_vectorstore


def get_bm25_index(app, *, get_chunk_registry, get_vectorstore):
    bm25_index = getattr(app.state, "bm25_index", None)
    if bm25_index is None:
        bm25_index = load_bm25_index(
            chunk_registry=get_chunk_registry(),
            vectorstore=get_vectorstore(),
        )
        app.state.bm25_index = bm25_index
    return bm25_index


def get_vectorstore(app, *, get_embeddings):
    vectorstore = getattr(app.state, "vectorstore", None)
    if vectorstore is None:
        vectorstore = load_vectorstore(str(DB_DIR), get_embeddings())
        app.state.vectorstore = vectorstore
    return vectorstore


def get_chunk_registry(app):
    chunk_registry = getattr(app.state, "chunk_registry", None)
    if chunk_registry is None:
        chunk_registry = load_chunk_registry(str(REGISTRY_PATH))
        app.state.chunk_registry = chunk_registry
    return chunk_registry


def refresh_runtime_state(app, *, get_embeddings):
    app.state.vectorstore = load_vectorstore(str(DB_DIR), get_embeddings())
    app.state.chunk_registry = load_chunk_registry(str(REGISTRY_PATH))
    app.state.bm25_index = load_bm25_index(
        chunk_registry=app.state.chunk_registry,
        vectorstore=app.state.vectorstore,
    )
