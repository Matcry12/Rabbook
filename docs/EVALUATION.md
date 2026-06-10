# Rabbook Evaluation — A White Paper on an Agentic RAG System

**System:** Rabbook — a local-first agentic Retrieval-Augmented Generation (RAG) assistant.
**Model under test:** Ollama `gemma4:e2b-it-qat` (≈4.6 B params), thinking mode ON, run locally.
**Date:** 2026-06-10
**Scope:** one end-to-end run of the full system over a 100-case public benchmark, judged for retrieval quality, answer correctness, and hallucination behaviour.

---

## Abstract

Rabbook answers questions from an ingested document corpus and, when the corpus is
insufficient, can reach the open web through tools. The hard questions for any RAG
system are not "does it retrieve?" but "does it *use* what it retrieved, and does it
*refuse* when it knows nothing?" This paper measures all three layers — retrieval,
answer quality, and agent behaviour — on a deliberately hard public benchmark
(multi-hop HotpotQA + unanswerable SQuAD v2).

**Headline results (gemma4:e2b-it-qat)** — a *baseline* run, then a *tuned* run after
the retrieval fix (§4.1) and prompt rework (§5.1):

| Layer | Metric | Baseline | Tuned |
|---|---|---|---|
| Retrieval | Hit@k / Recall@k / MRR | 0.99 / 0.78 / 0.95 | **1.00 / 0.83 / 0.95** |
| Answer quality | Correct on multi-hop answer cases | 51 / 80 ≈ 64 % | **57 / 80 ≈ 71 %** |
| Hallucination | No-fabrication rate (lenient) | 80 % | **90 %** |
| Hallucination | Confidently fabricated on unanswerable cases | 4 / 20 | **1–2 / 20** |
| Routing | `fetch_url` used (snippet → full page) | 2 / 100 | **8 / 100** |
| Latency | Mean / median per question | 15.0 s / 13.6 s | 16.7 s / 13.3 s |

The retrieval layer was already strong; the binding ceiling is the **4.6 B model's
multi-hop reasoning** and its **grounding discipline**, not the pipeline. RAG works
"properly" in that retrieval almost always surfaces the right evidence; the losses
happen downstream in reasoning. Two targeted changes — widening the candidate pool and
restructuring the prompt — lifted answers **64 % → 71 %** and cut fabrications, at the
cost of a few regressions (documented honestly in §5.1).

---

## 1. The system under evaluation (the flow)

```
                          ┌─────────────────────────────┐
   user query ──────────► │      Tool agent loop        │
                          │   (agents/tool_agent.py)    │
                          │  LLM picks a tool each step │
                          └──────────────┬──────────────┘
                                         │ chooses among
              ┌──────────────────────────┼──────────────────────────┐
              ▼                          ▼                          ▼
     query_documents               web_search                   fetch_url
   (local RAG pipeline)        (DuckDuckGo snippets)      (crawl4ai full page,
                                                            embeds into corpus)
              │
              ▼
   ┌───────────────────────────────────────────────────────────────┐
   │  RAG retrieval pipeline (rag/retrieve.py)                       │
   │  1. query → 2–4 sub-queries (LLM)                              │
   │  2. dense (Chroma) + BM25 candidates per sub-query             │
   │  3. RRF fusion                                                  │
   │  4. CrossEncoder rerank                                         │
   │  5. context-window expansion (neighbour chunks)                │
   │  6. grounding gate (rerank score + chunk count)               │
   └───────────────────────────────────────────────────────────────┘
```

Two layers are tested together in the headline run: the **deterministic retrieval
pipeline** and the **LLM-driven tool agent** that decides when to read locally, when to
search the web, and when to refuse.

The agent's behaviour is shaped by a hardened system prompt
(`agents/tool_agent.py`) with four rules added in response to observed failures:

1. **Use the facts in the retrieved context** — stop concluding "no answer" after a
   successful retrieval.
2. **For multi-part questions, resolve each part then combine** — targets multi-hop.
3. **`web_search` returns only snippets; if they are insufficient you MUST `fetch_url`**
   the page — the snippet and the full page are *different tools*.
4. **Never repeat a search; never return an empty answer** — broke an infinite
   re-search loop and an empty-answer mode.

Rule 4 had the biggest effect: forbidding repeat searches forced the agent back to
`query_documents`, where the answer frequently already lived, and cut average latency
~27 %.

---

## 2. Why a three-layer evaluation

A single number hides where a RAG system actually fails. Each layer isolates one
failure mode:

| Layer | Script | Question it answers | Judge |
|---|---|---|---|
| **Retrieval** | `evaluate_retrieval_metrics.py` | Did the retriever fetch the labelled chunks? | None — deterministic IR metrics |
| **Answer quality** | `evaluation/time_agent.py` + LLM judge | Is the final answer correct / non-fabricated? | LLM-as-judge, human-calibrated |
| **Agent behaviour** | `evaluate_agent.py` | Does it route correctly and refuse unanswerable questions? | Heuristic |

Retrieval can look perfect while generation fabricates; generation can look fine while
routing is broken. Only the combination tells you which.

---

## 3. The benchmark dataset

**File:** `evaluation/data/eval_dataset.json` — 100 cases from two public benchmarks.

- **80 answer cases — HotpotQA, distractor setting.** Multi-hop questions that require
  chaining two facts across two gold paragraphs, mixed with 8 distractor paragraphs
  each. All paragraphs are ingested verbatim, so the retriever must find 2 gold chunks
  among 10 paragraphs' worth of text.
- **20 fallback cases — SQuAD v2 unanswerable.** Questions whose answer is *not* in the
  corpus. Questions overlapping any corpus title are excluded to avoid accidental
  answerability. These exist purely to test **refusal vs. hallucination**.

Each item carries a `ground_truth`, the `relevant_chunk_ids` the retriever should
surface, and an `expected_behavior` of `"answer"` or `"fallback"`.

> **Why this dataset is hard on purpose.** Multi-hop + distractors stresses retrieval
> recall; unanswerable questions stress grounding discipline. A small local model has
> nowhere to hide.

### Rebuilding the corpus

```bash
../venv/bin/python -m evaluation.download_eval_sources   # 1. fetch raw data (idempotent)
../venv/bin/python -m evaluation.build_eval_corpus        # 2. build corpus + eval spec
# 3. clean rebuild of the Chroma index from the new corpus
../venv/bin/python -c "from rag.ingest import reingest_directory; from langchain_huggingface import HuggingFaceEmbeddings; from core.config import DB_DIR, REGISTRY_PATH; e=HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2'); print(reingest_directory('data/eval_corpus', e, str(DB_DIR), REGISTRY_PATH))"
../venv/bin/python -m evaluation.label_eval_dataset       # 4. map gold sentences → chunk ids
```

---

## 4. Layer 1 — Retrieval: is the RAG "proper"?

Measured over the 80 answerable cases at k = 4. Corpus: 797 documents / 1481 chunks.

| Metric | Value | Reading |
|---|---|---|
| **Hit@4** | **0.988** | At least one gold chunk in the top-4 almost every time. |
| **Recall@4** | **0.780** | Of ~2 gold chunks per question, the retriever finds ~78 %. |
| **Precision@4** | **0.422** | ≈1.7 of 4 retrieved chunks are gold (ceiling ≈0.55 at ~2.2 gold/k=4). |
| **MRR** | **0.952** | The *first* gold chunk is almost always rank 1. |

**Is the RAG proper? Yes — this is the strong layer.** Hit@4 of 0.99 and MRR of 0.95
mean the hybrid pipeline (dense + BM25 → RRF → CrossEncoder rerank) reliably puts the
*first* relevant chunk at the top. The hybrid design is doing its job.

**Where it leaks:** Recall@4 = 0.78 is the honest difficulty signal. Each multi-hop
question needs *both* gold chunks, and the **second hop** is what slips out of the top-4
among 8 distractors. MRR ≫ Recall confirms it: finding the first chunk is easy, finding
the second is the hard part. This is an inherent property of multi-hop retrieval, not a
pipeline defect — but it directly caps downstream answer accuracy (you cannot reason
over a chunk you never retrieved). §4.1 diagnoses exactly *where* the second chunk is
lost and tunes the pipeline to recover most of it.

### 4.1 Diagnosing and fixing the recall gap

A missing second chunk has three possible causes, each needing a different fix:

- **(A) ranking** — the chunk is retrieved into the candidate pool but reranked below
  the cut → fix by returning more.
- **(B) pool too narrow** — the chunk never enters the candidate pool because
  `candidate_k` is too small → fix by widening the pool.
- **(C) unretrievable** — dense+BM25 genuinely cannot find it → needs a stronger
  embedding model or iterative retrieval.

To tell them apart we ran each answer case twice (a one-off diagnostic with
query transform off, deterministic): once at the **production pool** (`candidate_k=8`)
and once at a **wide pool** (`candidate_k=40`), measuring both partial recall and the
true multi-hop signal — **were *both* gold chunks found**.

| Stage | Partial recall | **Both gold found** |
|---|---|---|
| recall@4 (production top-4) | 0.780 | 0.537 |
| recall@8 (production pool) | 0.822 | 0.625 |
| recallPool (production pool, ~16 cand) | 0.838 | 0.662 |
| **recallPool (WIDE pool, 40 cand)** | **0.946** | **0.887** |

**Finding: it is overwhelmingly cause (B), not (A).** Looking deeper into the *current*
pool lifts "both gold found" only 0.537 → 0.662, but widening the *candidate pool*
8→40 lifts it 0.662 → **0.887 (+22 pts)**. The second chunk was usually **not in the
candidate pool at all** — `candidate_k=8` (dense-top-8 + BM25-top-8) simply did not reach
the bridge-entity chunk. A residual **9/80 cases (~11 %)** miss a gold chunk even in the
wide pool — these are cause (C) and set the retrieval ceiling at ~0.887 with the current
embeddings.

**Fix applied** (`core/config.py`, env-overridable):

| Parameter | Old | New | Rationale |
|---|---|---|---|
| `RABBOOK_RERANK_CANDIDATE_K` | 8 | **40** | The big lever — gets the second chunk into the pool. |
| `RABBOOK_BM25_CANDIDATE_K` | 8 | **40** | Same, for the lexical side. |
| `RABBOOK_RETRIEVAL_K` | 4 | **6** | Return enough that the reranker can surface both gold chunks. |

**Result — production retrieval after tuning:**

| Metric | Before (k=4, cand=8) | After (k=6, cand=40) |
|---|---|---|
| Hit@k | 0.988 | **1.000** |
| Recall@k | 0.780 | **0.834** |
| MRR | 0.952 | 0.954 |
| Precision@k | 0.422 | 0.302 |

Hit@k reaching 1.000 means at least one gold chunk is now returned for *every* case.
Recall rose +5.4 pts; the gap to the 0.946 pool ceiling is return-k headroom (returning
6 of a reranked wide pool doesn't always place *both* gold chunks in the top-6) and could
be closed further by raising k at the cost of feeding the small LLM more context.

**Cost:** negligible. Widening `candidate_k` only enlarges the CrossEncoder's input;
160 retrievals including 80-candidate reranks ran in 31 s on **CPU** (~0.2 s each). No
GPU required — the project's GTX 1660 SUPER (6 GB) is reserved for the LLM via Ollama.

**Why we optimised recall over precision.** For multi-hop QA the two are asymmetric: a
**missing** gold chunk is fatal (the fact isn't in context, so the model can only guess
or refuse), whereas an **extra** non-gold chunk is just ignored by the LLM. So Precision
dropping 0.42 → 0.30 is expected and benign — we widened the *candidate pool* (free
recall; the reranker still trims the returned set to 6, so junk never reaches the LLM)
and only nudged the *return count* 4→6, protecting recall while keeping the returned
context small and clean. Precision still matters as a *limit* — too many distractors
confuse a 4.6 B model — but recall is what decides whether a correct answer is even
*possible*.

> **Not yet re-validated end-to-end.** These are retrieval-layer numbers. The answer eval
> (§5) was run on the *old* config; whether the recall lift converts to higher answer
> accuracy requires re-running the full agent eval, which is the next step.

---

## 5. Layer 2 — Answer quality: 64 % → 71 %

> **Headline: the tuned agent answers ~71 % of multi-hop cases correctly** (57/80), up
> from a 64 % baseline (51/80). This section first explains the baseline and *why* it sat
> at 64 %, then §5.1 documents the two fixes that lifted it to 71 % and the regressions
> they cost. If you only want the final number, it is **71 %**.

### How the run was performed

**Harness:** `evaluation/time_agent.py`. Each case runs through `run_tool_agent` with a
**fresh conversation** (no state bleeds between cases). Per case it records wall-clock
start, elapsed seconds, the full tool sequence, and the final answer, written to
`evaluation/data/time_agent_results.json`.

**Run health:** 100/100 completed, **0 hard errors**, **1 empty answer**, mean
**15.0 s/case** (median 13.6 s, range 3.8–45.1 s).

> **Reproducibility caveat — the corpus mutates during the run.** `fetch_url` embeds
> the pages it crawls into the live vectorstore, so a case running after a fetch sees a
> marginally larger corpus. Impact here is negligible (crawled pages are off-topic for
> other questions), and in practice `fetch_url` fired only **twice** in 100 cases — but
> a headline CV number should be produced from a frozen corpus snapshot.

### How the answers were judged

We used an **LLM-as-judge** (Haiku) rather than string matching, because HotpotQA
answers are short facts phrased many ways: "CMS" ≡ "Centers for Medicare and Medicaid
Services", "UK" ≡ "British", "31 July 1975" ≡ "1975-07-31". A judge must accept
semantic equivalents, more-specific-but-correct answers, and matching yes/no.

**The judge was calibrated before it was trusted.** We hand-scored 20 cases, then had
the judge score the same 20: **agreement 19/20 (95 %)**, the lone miss a borderline
hedge. Every batch was then reconciled against a manual pass; where the judge's
self-tallied total disagreed with its own per-case table, **the per-case table is
authoritative** (one batch self-miscounted 24 vs. an actual 22). The full per-case
verdicts — question, answer, tools, time, verdict, reason — live in
[`evaluation/data/judge_verdicts.md`](../evaluation/data/judge_verdicts.md).

### Baseline result: 51 / 80 ≈ 64 % correct (before tuning)

| Batch | Correct |
|---|---|
| 1–20 | 15/20 |
| 21–40 | 14/20 |
| 41–60 | 10/20 |
| 61–80 | 12/20 |
| **Total** | **51 / 80 ≈ 64 %** |

**Why not higher? The four recurring causes (from the per-case table):**

1. **Multi-hop reasoning is the ceiling (the dominant cause).** When the fact is local
   and the chain is short, the agent is accurate. Nearly every miss is a failure to
   *chain two facts* — compounded by Recall@4 = 0.78, since some second-hop chunks were
   never retrieved. This is a 4.6 B-model limitation, and it is exactly what the
   benchmark is designed to expose.
2. **Won't commit on comparisons.** On "who has more / which is better", the agent
   sometimes hedges instead of picking one — scored wrong (e.g. case 51 declines to name
   the Brothers Quay; case 71 won't give a yes/no on layer count).
3. **Granularity mismatch (partly a judging artifact).** The agent named the
   *institution* when the reference wanted the *city* — "Fordham University" vs.
   "New York City" (case 43), "Northern Kentucky University" vs. "Highland Heights"
   (case 52). A more lenient rule would recover ~2 points.
4. **Gives up with evidence in hand.** A handful of cases return "could not find" or an
   empty answer (case 6) even though retrieval succeeded — the residue of the
   retrieved-but-ignored failure that prompt rule 1 mostly, but not fully, fixed.

**Is this good?** For a 4.6 B model running locally and free, on *multi-hop* HotpotQA
with distractors, yes — even the 64 % baseline is a respectable, honest result, and the
tuned **71 %** more so. Published HotpotQA numbers that look higher generally use far
larger models and/or gold-paragraph (non-distractor) settings. The number is a property
of the model size and the task difficulty, not a broken pipeline.

### 5.1 Prompt iteration and the tuned run

The baseline above used the first system prompt. We then **diagnosed every failure** and
mapped each to a missing prompt rule, rather than tweaking wording at random:

| Failure pattern | Example cases | Prompt rule added |
|---|---|---|
| Wrong granularity (institution, not city) | 43, 52, 64 | "Answer at the level the question asks for" |
| Won't commit (explains instead of answering) | 6, 51, 71, 73 | "Begin yes/no with Yes/No; on *which-has-more*, you MUST pick one" |
| Premature refusal (gives up after local only) | 16, 72, 79 | "If local fails you MUST web_search before refusing" |
| Snippet starvation (never reads full page) | 8, 42, 45 | sharpened "if a snippet is insufficient, fetch_url" |
| Fabrication on unanswerable questions | 82, 88, 89, 93 | "Only state tool-supported facts; refuse rather than guess" |

The key insight was a **contradiction the first prompt never resolved**: "give up less"
(escalate to the web) versus "make up less" (refuse). The fix sequences them —
*exhaust tools → then refuse cleanly, never fabricate* — so refusal is a last resort and
fabrication is never licensed. The reworked prompt is in `agents/tool_agent.py`
(structured into FINDING / ANSWERING / WHEN TO STOP blocks).

**Validation method.** To avoid *overfitting the prompt to the eval set*, we ran the full
100 cases again (new prompt **and** the §4.1 retrieval config together) — not just the
known failures — so the previously-passing cases act as a regression check.

**Result (tuned run, `evaluation/data/rerun_full100.json`):**

| Metric | Baseline | Tuned | Δ |
|---|---|---|---|
| Answer accuracy (1–80) | 51/80 (64 %) | **57/80 (71 %)** | +9 fixed, −3 regressed |
| No-fabrication rate | 80 % | **90 %** | fabrications 4 → 1–2 |
| `fetch_url` used | 2/100 | **8/100** | rule actually fired |
| Mean latency | 15.0 s | 16.7 s | +1.7 s (more web/fetch) |

**Honest accounting — it is not a free lunch.** Nine answer failures flipped to correct
(yes/no commits, multi-hop chains, recall-driven recoveries), but **three previously-correct
cases regressed:**

- **#74** Clinchfield Railroad → now wrongly "Union Pacific" (a factual flip).
- **#28** "who had more influence, Sartre or Shaw?" → now *hedges* — the new commit rule
  was ignored on this one.
- **#44** Greenwood neighborhood → now answers "Tulsa / Black Wall Street", dropping the
  finer-grained name (the granularity rule again).

Two fallback wrinkles also appeared: **#92** now confidently dates "Prussia" to 1701
(borderline new fabrication) and **#94** returned an *empty* answer (violating the
"never empty" rule). The **granularity rule (#43, #52 still fail) did not reliably take**,
which points to a model-capability ceiling rather than a wording problem.

**Net:** clearly positive on both axes (+6 net answers, −3 fabrications, 4× more
`fetch_url`), with a small, disclosed regression cost. These verdicts use the same
human-judgment rubric as the baseline; locking them behind a reproducible scripted judge
remains the open item (§8).

---

## 6. Layer 3 — Does the agent hallucinate?

This is the most CV-relevant question: on the 20 **unanswerable** cases, does the agent
*refuse*, or does it confidently make something up? Each answer is classified into one
of three buckets (the agent has web tools, so a correct web-sourced answer is legitimate,
not a failure):

| Bucket | Baseline | Tuned | Meaning |
|---|---|---|---|
| **Refused** | 9 / 20 | 8 / 20 | Said it couldn't find it / asked for clarification — correct grounding. |
| **Correct from web** | 7 / 20 | 10 / 20 | Gave a factually correct answer it sourced online — legitimate. |
| **Fabricated** | 4 / 20 | **1–2 / 20** | Confidently asserted a wrong/unsupported claim — the real failure. |

The tuned prompt (§5.1) turned fabrications 82, 88, 89 into clean refusals / correct-web
answers; only 93 (and borderline 92) remain. Some former refusals became legitimate
correct-from-web answers, so the *no-fabrication* rate rose **80 % → 90 %** even as strict
refusals dipped slightly.

From these:

- **Strict refusal rate** — baseline 9/20 = 45 %, tuned 8/20 = 40 % (a couple of refusals
  became correct-from-web answers, which is an improvement, not a loss).
- **No-fabrication rate** — baseline 16/20 = 80 % → **tuned 18/20 = 90 %** — how often it
  *avoids* a confident lie (refused OR correct-from-web).

**Yes, the agent still hallucinates — but the tuned prompt cut it from ~20 % to ~5–10 %
of unanswerable questions.** The baseline's four fabrications were concrete and checkable:

- **Case 82** — mischaracterises "Sonderungsverbot" (actually a no-segregation rule).
- **Case 88** — claims Philip Honywood governed New France; he did not.
- **Case 89** — invents an HDTV "launch failure" for Sky Digital (it launched fine in 1998).
- **Case 93** — asserts unsupported "Indigenous Australian HLA alleles".

These are the genuine weakness, and they are subjective to score — distinguishing
"fabricated" from "correct from web" requires *fact-checking niche claims*, which a
single judge does imperfectly. That subjectivity is the strongest argument for the
next-step judge in §8.

> The deterministic heuristic in `evaluate_agent.py` reports only **7/20** "fallback
> handled" because it matches literal refusal phrases and cannot credit clarification
> requests or hedged non-answers. The LLM-judge 9/20 is the more accurate refusal count.

---

## 7. Routing behaviour — what the agent actually did

Deterministic checks over the 100-case run (`evaluate_agent.py`):

| Check | Value |
|---|---|
| Called a tool (not answering from memory) | 99 / 100 |
| Used local RAG (`query_documents`) first | 75 / 83 of cases that called it |
| Finished within `MAX_ITERATIONS` | 99 / 100 |
| Fallback handled (refusal-phrase heuristic) | 7 / 20 |

Tool usage across the run:

- **First tool:** `query_documents` 75×, `web_search` 24×, none 1×. **Local-first
  routing works** — the agent reaches for the corpus before the web most of the time.
- **`web_search` used in 58 cases, but `fetch_url` in only 2** *(baseline)*. **This was
  the key behavioural finding.** Despite an explicit prompt rule to escalate from snippet
  to full page, the small model almost never did — it tried to answer from DuckDuckGo
  snippets alone. Several web-dependent misses traced back to this: the answer was on the
  page, but the agent never opened it.
- **After the §5.1 prompt rework, `fetch_url` usage rose 2 → 8 / 100** — the sharpened
  rule measurably changed behaviour. Still low in absolute terms, so the residual gap is
  likely a tool-design question (e.g. have `web_search` auto-fetch its top result) rather
  than more prompt words.

---

## 8. Issues, limitations, and next steps

**Confirmed issues (ranked by impact, after the §4.1 + §5.1 tuning):**

1. **Multi-hop reasoning ceiling** — the 4.6 B model is the binding constraint, now on
   the ~71 %. A stronger agent model (Groq Llama, Gemini) is the highest-leverage fix.
2. **Granularity ceiling** — the agent still names the institution when asked for the
   city (#43, #52); the explicit prompt rule did not reliably take, suggesting a model
   limit, not wording.
3. **Snippet-only web answering** — improved (`fetch_url` 2 → 8 / 100) but still low; the
   real fix is likely tool design (auto-fetch the top web result), not more prompt words.
4. **Residual fabrication (1–2 / 20)** on unanswerable questions — much reduced, not zero.
5. **Prompt-change regressions** — the §5.1 rework broke 3 previously-correct cases
   (#28, #44, #74) while fixing 9; net positive, but it shows prompt edits carry risk.

**Methodological limitations (stated honestly for a CV artifact):**

- **Judge reproducibility.** Scoring used an LLM judge + manual reconciliation, not yet
  a committed, re-runnable script. **Next step:** a reproducible judge (Gemini, already
  wired into the project) that emits structured JSON, so the headline numbers regenerate
  on demand and the contested fabrication calls (82/88/89/93) are fact-checked with web
  access.
- **Corpus mutation** during the run (§5) — freeze a snapshot for the final number.
- **Single model.** Only `gemma4:e2b-it-qat` was measured. A prior spot-check showed the
  larger `e4b` variant fixes specific hallucinations and recovers hard answers — *model
  size*, not thinking mode, is the lever.

---

## 9. How to reproduce

```bash
# Layer 1 — deterministic retrieval metrics (fast, no API cost)
../venv/bin/python -m evaluation.evaluate_retrieval_metrics

# Layer 2 — full agent run with per-case timing, tools, answers → JSON
../venv/bin/python evaluation/time_agent.py 100 0

# Layer 3 — deterministic agent-behaviour checks
../venv/bin/python -m evaluation.evaluate_agent
```

Ensure the corpus is ingested first (`../venv/bin/python ingest_docs.py`).

### Optional: LangSmith tracing

Set in `.env` to trace every tool call and LLM invocation:

```
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=<your key>
LANGCHAIN_PROJECT=rabbook-eval
```

The LangSmith UI shows which tools were called, in what order, with what arguments, and
the latency/token cost of each step — the fastest way to debug routing.

---

## 10. Artifacts

| File | Role |
|---|---|
| `evaluation/data/eval_dataset.json` | The 100-case golden dataset (80 answer + 20 fallback) |
| `evaluation/evaluate_retrieval_metrics.py` | Deterministic retrieval metrics (Hit/Recall/Precision/MRR) |
| `evaluation/time_agent.py` | Run harness — per-case timing, tools, answers → JSON |
| `evaluation/data/time_agent_results.json` | Raw per-case results — **baseline** run |
| `evaluation/data/rerun_full100.json` | Raw per-case results — **tuned** run (new prompt + §4.1 config) |
| `evaluation/data/judge_verdicts.md` | **Full per-case table** — question, answer, tools, time, verdict, reason |
| `evaluation/AGENT_ANSWER_QUALITY_REPORT.md` | Companion report on the answer-quality run |
| `agents/tool_agent.py` | The agent under test (system prompt iterated here) |
| `rag/retrieve.py` | The retrieval pipeline (hybrid + rerank + grounding gate) |

---

## Appendix — Metric definitions

**Retrieval** (at k = 4): **Hit@k** = 1 if any gold chunk in top-k. **Recall@k** =
|retrieved ∩ gold| / |gold|. **Precision@k** = |retrieved ∩ gold| / k. **MRR** = mean of
1/rank of the first gold chunk.

**Answer cases:** CORRECT if the answer states the same fact as the reference (accepting
abbreviations, more-specific answers, date formats, matching yes/no); INCORRECT if it
states a wrong fact or fails to answer (empty, "couldn't find", refusal on an answerable
question).

**Fallback cases:** Refused / Correct-from-web / Fabricated, as defined in §6. Strict
refusal rate = Refused / 20; no-fabrication rate = (Refused + Correct-from-web) / 20.
