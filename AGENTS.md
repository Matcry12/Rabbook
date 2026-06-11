# Repository Guidelines

## Project Structure & Module Organization
`main.py` is the FastAPI entrypoint and serves the web UI from `templates/index.html` and `static/style.css`. `ingest.py` handles document loading, chunking, metadata, and Chroma ingestion. `retrieve.py` handles retrieval, context expansion, and query-time helpers. `config.py` centralizes paths and environment-driven settings. `prompt.py` contains prompt-building logic. Test data and evaluation inputs live in `tests/`, currently with `tests/eval_question.json`.

## Build, Test, and Development Commands
- `cd rabbook && ../venv/bin/python main.py`: run the FastAPI app locally.
- `cd rabbook && ../venv/bin/python ingest.py`: rebuild the vector store and chunk registry from `data/`.
- `cd rabbook && ../venv/bin/python -m py_compile main.py ingest.py retrieve.py config.py prompt.py`: quick syntax validation.
- `cd rabbook && ../venv/bin/python -m pip install -r requirements.txt`: install dependencies into the shared virtual environment.

Use `RABBOOK_PORT=6001 ../venv/bin/python main.py` if you need to override the default port.

## Coding Style & Naming Conventions
Follow PEP 8 with 4-space indentation. Keep functions small and linear; favor obvious control flow over heavy abstraction. Use `snake_case` for functions, variables, and module-level settings. Keep metadata keys stable and explicit, for example `document_id`, `chunk_index`, and `chunk_id`. Prefer direct helper names such as `load_chunk_registry()` over generic names like `process_data()`.

Code in this repository should optimize for fast review, not maximum cleverness. Keep the main flow simple and easy to explain. Prefer readable step-by-step logic over layered abstraction, and only add helper functions when they remove real confusion. Use explicit names like `paragraphs`, `current_chunk_text`, and `chunk_registry`; avoid vague names like `data`, `items`, or `parts`. If a function needs a long explanation, simplify it.

Keep complexity concentrated in one well-named place instead of spreading it across many helpers. Choose the simpler implementation unless extra complexity provides a clear benefit. Strong defaults in `config.py` are preferred over long parameter lists. Comments should be rare and only clarify non-obvious logic; clear naming should do most of the work.

When comments are needed, explain the intent or tradeoff, not the syntax. Good comments describe why a block exists, for example why context windows use the chunk registry or why semantic overlap keeps trailing units. Avoid comments that only restate obvious code like variable assignment or loop mechanics.

## Testing Guidelines
There is no full automated test suite yet. Before opening a PR, run the `py_compile` check above and smoke-test the main flows:
- ingest a sample `.pdf` or `.txt`
- ask at least one question through the UI
- verify `chunk_registry.json` updates when documents are added

When adding tests, place them under `tests/` and name files `test_<feature>.py`.

## Commit & Pull Request Guidelines
Recent commits use concise, imperative summaries, for example: `Enhance configuration and retrieval functionality by adding chunk registry management`. Keep that style. PRs should include:
- a short problem/solution summary
- affected files or modules
- manual verification steps
- screenshots for UI changes

## Security & Configuration Tips
Store secrets in `.env`, never in source files. Do not commit `.env`, `chroma_db/`, uploaded documents, or local cache files. Rotate API keys immediately if they are exposed in logs, screenshots, or chat transcripts.
