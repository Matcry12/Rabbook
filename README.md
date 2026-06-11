---
title: Rabbook Agentic RAG
emoji: 📚
colorFrom: indigo
colorTo: purple
sdk: docker
app_port: 6001
pinned: false
---

# Rabbook — Agentic RAG System

> A production-quality Retrieval-Augmented Generation application built from scratch, featuring a real tool-use agent loop, hybrid retrieval, and a self-expanding knowledge base.

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-green?logo=fastapi)
![LangGraph](https://img.shields.io/badge/LangGraph-Agentic-orange)
![Tests](https://img.shields.io/badge/Tests-57%20passing-brightgreen)
![Docker](https://img.shields.io/badge/Docker-ready-2496ED?logo=docker)
![Accuracy](https://img.shields.io/badge/Answer%20Accuracy-71%25-success)
![Benchmark](https://img.shields.io/badge/Benchmark-100%20cases-blue)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

## What Makes This Different

Most RAG projects embed documents and call an LLM. Rabbook is built the way production systems are built:

| What | Why It Matters |
|------|----------------|
| **Real tool-use agent loop** | The LLM decides which tool to call each turn — not a hardcoded pipeline. Mirrors how Claude, Codex, and Gemini work. |
| **7-stage retrieval pipeline** | Dense + sparse fusion → RRF → cross-encoder reranking → context expansion → grounding gate. Each stage is measurable and independently testable. |
| **Self-expanding knowledge base** | When the agent fetches a web page, it auto-embeds it. Future queries over that content go through the full RAG pipeline — not raw text. |
| **Multi-provider LLM support** | Groq (Llama), Google Gemini, and local Ollama models (including thinking-mode toggle). Swap providers with a single env var. |
| **57 unit tests, zero LLM calls** | Full mock coverage across retrieval, agent loop, research graph, and structured output. |

---

## Results & Impact

I treated this as a real engineering project: build it, **measure it on a hard public benchmark, find the bottlenecks, and prove the fix** — all on a **free local 4.6B model** (Ollama `gemma`) at **$0 inference cost**.

**Benchmark:** 100 cases — 80 multi-hop **HotpotQA** (distractor setting) + 20 unanswerable **SQuAD v2** — scored by an LLM-as-judge **calibrated to 95% agreement with human labels** before use.

| Metric | Before | After | Lever |
|--------|:------:|:-----:|-------|
| **Answer accuracy** (multi-hop QA) | 64% | **71%** | Evidence-based prompt rework |
| **Hallucination** (unanswerable Qs) | ~20% | **~10%** | Grounding-discipline prompt rules |
| **Retrieval — both gold chunks found** | 54% | **89%** | Widened hybrid candidate pool |
| **Retrieval — Hit@k** | 0.99 | **1.00** | (same) |
| **Tool escalation** (snippet → full page) | 2 / 100 | **8 / 100** | Resolved "escalate vs. refuse" prompt conflict |

Each gain was **diagnosed before it was fixed** — e.g. the retrieval jump came from proving the second multi-hop chunk was missing from the candidate *pool* (not just mis-ranked), then widening it. Full methodology, per-case verdicts, and an **honest regression analysis** live in the white paper: **[`docs/EVALUATION.md`](docs/EVALUATION.md)**.

> **Why this matters:** diagnosing *why* a RAG system fails and proving the improvement with numbers is the skill that separates real AI engineering from a "ChatGPT wrapper."

---

## Two Agent Modes

### Mode 1 — Tool-Use Agent Loop (`agents/tool_agent.py`)

A real agentic loop where the LLM autonomously picks tools until it has enough information to answer. No hardcoded routing.

```
User query
    └─▶ LLM decides tool call
            ├─▶ query_documents  →  hybrid RAG search over local library
            ├─▶ web_search       →  DuckDuckGo, returns URLs + snippets
            └─▶ fetch_url        →  crawls page, auto-embeds into Chroma,
                                    returns "indexed — use query_documents"
    └─▶ LLM calls query_documents again → hits newly embedded content
    └─▶ LLM produces final grounded answer
```

The `fetch_url → embed → query_documents` pattern means every fetched page permanently enriches the local library for future queries.

### Mode 2 — LangGraph RAG Graph (`agents/rag_graph.py`)

A deterministic graph for structured, auditable retrieval with explicit grounding gates.

```mermaid
flowchart TD
    Start((User Query)) --> Prepare[Prepare Input & Metadata Filters]
    Prepare --> Retrieve[Hybrid Retrieval: Dense + Sparse]
    Retrieve --> Expand[Context Window Expansion]
    Expand --> Ground[Grounding Gate]

    Ground --> Decide{Decide Next Action}

    Decide -- "Grounded" --> Generate[Generate Answer with Citations]
    Decide -- "Weak Evidence" --> Refine[Refine Query]
    Decide -- "No Evidence" --> Research[Web Research Agent]

    Refine -->|retry| Retrieve
    Research -->|ingest & retry| Retrieve

    Generate --> Finalize[Finalize & Save History]
    Finalize --> End((Response))

    style Research fill:#f96,stroke:#333,stroke-width:2px
    style Refine fill:#bbf,stroke:#333,stroke-width:2px
    style Ground fill:#dfd,stroke:#333,stroke-width:2px
```

Switch modes with `RABBOOK_ENABLE_TOOL_AGENT=true/false`.

---

## Retrieval Pipeline

Seven stages run in sequence on every query:

```
1. Query Transform     LLM generates 2–4 sub-queries for broader coverage
2. Candidate Collection Dense (Chroma) + BM25 results per sub-query, deduplicated
3. RRF Fusion          Reciprocal Rank Fusion merges the ranked lists
4. Cross-Encoder Reranking  ms-marco-MiniLM re-scores against the original query
5. Context Window Expansion  Neighboring chunks added for full document context
6. Grounding Gate      Rerank score + chunk count gate; blocks hallucination-prone answers
7. Answer Generation   Structured output with citation repair fallback
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.13 |
| Agent Orchestration | LangGraph, LangChain tool-use (`bind_tools`) |
| Vector Store | ChromaDB |
| Embeddings | `all-MiniLM-L6-v2` (HuggingFace, local) |
| Sparse Retrieval | Rank-BM25 |
| Reranking | `ms-marco-MiniLM-L-6-v2` Cross-Encoder |
| LLM Providers | Groq (Llama 3.x), Google Gemini, Ollama (local, thinking-mode aware) |
| Web Crawling | crawl4ai + DuckDuckGo (`ddgs`) |
| Frontend | Jinja2, Vanilla CSS |
| Testing | `unittest` + mocks, 57 tests, no real LLM calls |

---

## Project Structure

```
agents/
  tool_agent.py       — real tool-use agent loop (the main path)
  rag_graph.py        — LangGraph deterministic graph
  research_graph.py   — standalone web research agent
  services.py         — public API: answer_query(), AnswerResult
rag/
  retrieve.py         — full 7-stage retrieval pipeline
  chunking.py         — semantic chunking (embedding-based split points)
  ingest.py           — document loading → Chroma + chunk registry
  web_ingest.py       — URL fetch, crawl, save, web_search
  registry.py         — chunk registry (O(1) neighbor lookup for context expansion)
app/
  web.py              — FastAPI routes, LLM instantiation, provider switching
  runtime.py          — lazy-load & cache: vectorstore, BM25, registry
core/
  config.py           — all env vars with defaults
evaluation/           — the 3-layer evaluation suite (see "Evaluation" below)
  evaluate_retrieval_metrics.py  — Layer 1: Hit@k / Recall@k / MRR
  evaluate_agent.py              — Layer 3: routing & refusal checks
  evaluate_ragas.py             — RAGAS faithfulness / answer relevancy
  time_agent.py                 — full agent run harness (timing, tools, answers)
  build_eval_corpus.py          — builds the 100-case HotpotQA + SQuAD v2 benchmark
  data/                         — golden dataset + per-case results & verdicts
docs/
  EVALUATION.md       — evaluation white paper (methodology, results, analysis)
```

> 📊 **Evaluation lives in [`evaluation/`](evaluation/) with a full write-up in
> [`docs/EVALUATION.md`](docs/EVALUATION.md).** See the [Results & Impact](#results--impact)
> and [Evaluation](#evaluation) sections.

---

## Setup

### Docker (recommended)

```bash
cp .env.example .env
# Add GROQ_API_KEY or GEMINI_KEY

docker compose up --build
# → http://localhost:6001
```

The image pre-downloads embedding and reranking models at build time so the first query is instant. The `data/` directory is mounted as a volume — documents and chat history persist across restarts.

### Local

```bash
git clone <repo>
cd rabbook

python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

cp .env.example .env
# Add GROQ_API_KEY or GEMINI_KEY
```

Key `.env` options:

```bash
RABBOOK_LLM_PROVIDER=groq          # groq | gemini | ollama
RABBOOK_LLM_MODEL=llama-3.1-8b-instant
RABBOOK_ENABLE_TOOL_AGENT=true     # real agent loop (recommended)
RABBOOK_ENABLE_LANGGRAPH_AGENT=true
RABBOOK_ENABLE_RESEARCH_FALLBACK=false
RABBOOK_OLLAMA_THINKING=false      # suppress <think> blocks for gemma/deepseek
```

```bash
python main.py          # → http://127.0.0.1:6001
python ingest_docs.py   # embed files from data/uploads/
```

---

## Running Tests

```bash
python -m pytest tests/ -q
# 57 passed
```

All tests use mocks — no API keys, no network, no vectorstore required.

---

## Evaluation

Rabbook ships with a **three-layer evaluation suite** — because retrieval can look perfect while generation fabricates, and generation can look fine while agent routing is broken. Each layer isolates one failure mode:

| Layer | Measures | Judge |
|-------|----------|-------|
| **Retrieval** | Does the retriever fetch the gold chunks? (Hit@k, Recall@k, MRR) | Deterministic IR metrics |
| **Answer quality** | Is the final answer correct / non-fabricated? | LLM-as-judge (95% human-calibrated) |
| **Agent behaviour** | Does it route locally first and refuse unanswerable questions? | Heuristic |

**Benchmark:** 100 cases from two public datasets — **80 multi-hop HotpotQA** (distractor setting: 2 gold + 8 distractor paragraphs per question) and **20 unanswerable SQuAD v2** questions (to test refusal vs. hallucination).

| Layer | Headline (tuned, gemma4:e2b) |
|-------|------------------------------|
| Retrieval | Hit@k **1.00** · Recall@k **0.83** · MRR **0.95** |
| Answer quality | **57 / 80 ≈ 71%** correct on multi-hop QA |
| Hallucination | No-fabrication rate **90%** · ~10% fabricated on unanswerable Qs |

```bash
# Layer 1 — retrieval IR metrics (fast, no API cost)
python -m evaluation.evaluate_retrieval_metrics
# Layer 3 — agent behaviour checks
python -m evaluation.evaluate_agent
```

📄 **Full write-up:** [`docs/EVALUATION.md`](docs/EVALUATION.md) — a white paper covering the methodology, the recall diagnostic, the prompt-failure taxonomy, the before/after results, a per-case verdict table, and limitations.

> RAGAS metrics (Faithfulness, Answer Relevancy) are also wired in via `evaluation/evaluate_ragas.py` as an industry-standard cross-check.

---

## Notes

- Port defaults to `6001` (browsers commonly block `6000`)
- Uploaded files: `data/uploads/`
- URL imports: `data/uploads/urls/` (persisted, re-ingested on restart)
- Chunk registry: `data/chunk_registry.json` (flat index for O(1) neighbor lookup)
