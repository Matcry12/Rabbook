# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
# Run the app (http://127.0.0.1:6001)
python main.py

# Ingest documents from data/uploads/
python ingest_docs.py

# Run all tests
../venv/bin/python -m pytest tests/

# Run a single test file
../venv/bin/python -m pytest tests/test_phase2_rag_graph.py

# Run a single test case
../venv/bin/python -m pytest tests/test_phase2_rag_graph.py::Phase2RagGraphTests::test_build_initial_graph_state_sets_expected_defaults

# Evaluate retrieval and answer quality
../venv/bin/python evaluate_retrieval.py
```

## Environment

Copy `.env.example` to `.env`. Key variables:

| Variable | Default | Purpose |
|---|---|---|
| `GROQ_API_KEY` | — | LLM inference (Groq) |
| `GEMINI_KEY` | — | Alternative LLM (Google Gemini) |
| `RABBOOK_LLM_PROVIDER` | `groq` | Provider select: `groq`, `gemini`, or `ollama` |
| `RABBOOK_LLM_MODEL` | `llama-3.1-8b-instant` | Model name |
| `RABBOOK_OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (local provider) |
| `RABBOOK_ENABLE_LANGGRAPH_AGENT` | `true` | Route queries through the LangGraph RAG graph (the only maintained path) |
| `RABBOOK_ENABLE_RESEARCH_FALLBACK` | `false` | Allow the RAG graph to fall back to the web research agent |
| `RABBOOK_MIN_GROUNDED_RERANK_SCORE` | `1.0` | Grounding gate threshold |

All retrieval parameters (`RETRIEVAL_K`, `CONTEXT_WINDOW`, `MIN_GROUNDED_CHUNKS`, etc.) are in `core/config.py` and can be overridden via env vars.

## Architecture

### Layer separation

```
rag/        — retrieval primitives (no app state, no web concerns)
agents/     — LangGraph graphs and orchestration
app/        — FastAPI routes, runtime state, view helpers
learning/   — flashcard and review logic (future)
core/       — config and shared constants
```

`agents/services.py` is the public API between `app/` and the agent/retrieval layers. All query routing goes through `answer_query()` there.

`answer_query()` has two branches keyed on `use_langgraph`. In practice queries always take the **LangGraph path** (`run_rag_graph_answer`); the `else` branch (the inline "Direct RAG" pipeline) is **legacy and no longer used** — kept for reference, not maintained. Don't extend it or worry about keeping it in sync. The research fallback (`enable_research`) is wired only through the LangGraph path.

### Retrieval pipeline (`rag/retrieve.py`)

The multi-stage pipeline runs in this order:

1. **Query transform** — LLM generates 2–4 sub-queries (`generate_sub_queries`)
2. **Candidate collection** — dense (Chroma) + BM25 results per query, deduplicated
3. **RRF fusion** — `fuse_ranked_documents` merges ranked lists
4. **Reranking** — `CrossEncoder` reorders the fused set against the original query
5. **Context window expansion** — `expand_with_context_window` adds neighboring chunks from the same document using the chunk registry
6. **Grounding check** — `check_grounding_evidence` gates on rerank score and chunk count
7. **Answer generation** — `generate_answer` tries structured output first, falls back to plain invoke, then runs citation repair if needed

### LangGraph RAG graph (`agents/rag_graph.py`)

Activated when `RABBOOK_ENABLE_LANGGRAPH_AGENT=true`. The graph wraps the same retrieval pipeline as nodes:

```
prepare_input → retrieve → expand_context → check_grounding
  → decide_next_action
      → generate_answer → finalize_response → END
      → fallback_answer → finalize_response → END
```

`RagGraphState` is a `TypedDict` carrying all intermediate state. `decide_next_action_node` sets `next_action` to `"answer"`, `"retry_retrieval"`, or `"fallback"` — but `"retry_retrieval"` routing is not yet wired (known gap, addressed in Phase 3 of `PROJECT_UPDATE_PLAN.md`).

### Research graph (`agents/research_graph.py`)

A separate LangGraph agent for open-web research, distinct from the RAG graph (which answers from ingested documents). Reached via the `enable_research` flag through `answer_query()`; off by default (`RABBOOK_ENABLE_RESEARCH_FALLBACK`). Nodes:

```
plan_search → execute_search → finalize → END
                            ↘ save_note (when save_to_notes)
```

- `plan_search` reuses `generate_sub_queries` to turn the topic into search queries.
- `execute_search` runs `web_search` (DuckDuckGo via `ddgs`) per query, dedupes by URL, then `fetch_url_content` (crawl4ai) on the top 3 for full text, falling back to the snippet on fetch failure.
- `finalize` synthesizes; `save_note` optionally persists via `rag/notes.py`.

Entry point is `run_research_agent(topic, llm=..., save_to_notes=..., debug_mode=...)`, returning a `ResearchResult` (defined in `agents/services.py` alongside `AnswerResult`).

### Chunk registry (`chunk_registry.json`)

A flat JSON index of every ingested chunk, structured as:

```json
{
  "by_document": { "<document_id>": { "<chunk_index>": { "page_content": "...", "metadata": {} } } },
  "by_chunk_id":  { "<chunk_id>":    { "page_content": "...", "metadata": {} } }
}
```

The registry enables O(1) neighbor lookup for context window expansion and powers the BM25 index. It is rebuilt by `ingest_docs.py` or via the UI maintenance actions.

### App runtime (`app/runtime.py`)

Vectorstore, chunk registry, and BM25 index are lazy-loaded and cached on `app.state`. Call `refresh_runtime_state` after ingestion to invalidate the cache.

### LLM selection (`app/web.py`)

The app instantiates `ChatGroq`, `ChatGoogleGenerativeAI`, or `ChatOllama` based on `RABBOOK_LLM_PROVIDER` and available API keys. All support `.with_structured_output()`, which the retrieval layer uses for structured query rewriting and answer drafting. Ollama-specific tuning (`RABBOOK_OLLAMA_NUM_GPU`, `RABBOOK_OLLAMA_THINKING`) lives in `core/config.py`.

## Ingestion flow

`ingest_docs.py` → `rag/ingest.py` (file loading, metadata) → `rag/chunking.py` (semantic chunking) → Chroma + `chunk_registry.json`

URL imports: `rag/web_ingest.py` fetches and stores pages under `data/uploads/urls/` so they survive re-ingestion as regular documents. It also exposes `web_search` (`ddgs`) and `fetch_url_content` (`crawl4ai`), shared with the research graph.

## Tests

Tests use `unittest` with mocks — no real LLM or vectorstore calls. Phase-named files track the build-out order:

- `test_phase1_architecture.py` — module boundaries and import structure
- `test_phase2_rag_graph.py` — LangGraph RAG-graph node behavior and routing
- `test_phase3_rag_agent.py` — RAG agent integration
- `test_phase4_research_agent.py` / `test_research_graph.py` — research graph nodes and `run_research_agent`
- `test_query_transform.py` — sub-query generation and parsing
- `test_answer_structured_output.py` — structured answer extraction and citation repair
- `test_web_ingest.py` — `web_search` / `fetch_url_content`
- `test_actions.py`, `test_gemma_prompt_wrapper.py` — app actions and prompt wrapping

## Working style

This project is a learning environment. The user reviews every change to understand it, so:

- **Keep each implementation step small to medium.** One focused change per session: one new node, one new function, one new test file. Do not bundle multiple features into a single change.
- **Explain the why before writing code.** Before implementing, briefly state what the change does and why it fits the architecture. One or two sentences is enough.
- **Prefer readable over clever.** Step-by-step control flow, explicit variable names (`retrieved_documents` not `docs`), no unnecessary abstractions. Code should be easy to read on first review.
- **Do not add helpers, error handling, or abstractions beyond what the immediate task requires.** The user will ask for the next step when ready.
- **After each change, point out what to look at.** Name the specific function or file the user should read to understand what changed and why.
