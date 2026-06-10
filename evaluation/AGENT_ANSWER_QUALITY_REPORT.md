# Agent Answer-Quality Report

**Date:** 2026-06-10
**System under test:** the tool agent (`agents/tool_agent.py`) — the LLM-driven loop that chooses between `query_documents` (local RAG), `web_search`, and `fetch_url`.

This report documents an end-to-end run of the agent over the full 100-case
benchmark, how the answers were judged, and the results. It complements
`docs/EVALUATION.md` (which defines the three eval layers and the dataset) by
recording one concrete agent run and its answer-quality scoring.

---

## 1. What we evaluated, and why

We measured **answer quality** — does the agent produce the *correct final
answer*, and on unanswerable questions does it *refuse instead of fabricating*.

This is distinct from the other two eval layers:

| Layer | Question it answers | Why it isn't enough alone |
|---|---|---|
| Retrieval metrics | Did the retriever fetch the right chunks? | Good retrieval can still be followed by a wrong answer. |
| **Agent answer quality (this report)** | **Is the final answer correct / non-fabricated?** | **This is what an end user actually experiences.** |

Retrieval already scored well (Hit@4 = 0.99, Recall@4 = 0.78). The open question
was whether the agent *uses* that retrieval to answer correctly — so we ran the
full agent and judged its answers.

---

## 2. System configuration

| Setting | Value | Why |
|---|---|---|
| Provider / model | Ollama `gemma4:e2b-it-qat` (4.6B) | Local, free, no API cost per case; the model the project runs by default. |
| Thinking mode | ON (`RABBOOK_OLLAMA_THINKING=true`) | Tested both; ON did not hurt and is the more capable setting. |
| Tools | `query_documents`, `web_search`, `fetch_url` | The full agent toolset. |
| `MAX_ITERATIONS` | 8 | Existing default. |

### System prompt (iterated during this work)

The agent's system prompt was hardened twice based on observed failures. The
final version (in `agents/tool_agent.py`) adds four rules that each fixed a
specific failure seen in earlier runs:

1. **"Use facts found in the retrieved context"** — fixed cases where the agent
   retrieved the answer but then concluded "no answer found" (e.g. the
   Sam Mendes / Ruby Yang multi-hop case went from wrong → correct).
2. **"For multi-part questions, resolve each part then combine"** — targets the
   multi-hop structure of HotpotQA.
3. **"web_search returns only short snippets; if they're not enough you MUST
   `fetch_url`"** — `web_search` returns DuckDuckGo snippets only; the full page
   requires the separate `fetch_url` (crawl4ai) tool.
4. **"Never repeat a search; never return an empty answer"** — fixed an
   infinite re-search loop (one case ran 8 identical web searches and timed out)
   and an empty-answer failure mode.

Rule 4 had the biggest impact: banning repeat searches forced the agent to fall
back to `query_documents`, where the answer often already lived.

---

## 3. The benchmark dataset

100 cases from two public benchmarks (full build pipeline in `docs/EVALUATION.md`):

- **80 answer cases** — HotpotQA distractor setting. Multi-hop questions needing
  reasoning across 2 gold paragraphs, mixed with 8 distractor paragraphs each.
  Hard for a small model: it must chain two facts.
- **20 fallback cases** — SQuAD v2 unanswerable questions. The answer is **not**
  in the ingested corpus. These test whether the agent refuses rather than
  hallucinates.

---

## 4. How we ran it

**Harness:** `evaluation/time_agent.py`

- Runs each case through `run_tool_agent` with a **fresh conversation** (no state
  carries between cases).
- Records per case: wall-clock start, elapsed seconds, the tool sequence, and the
  full final answer.
- Writes everything to `evaluation/data/time_agent_results.json`.
- Supports an offset so the run can be done in slices and appended (we ran 20
  first as a bug-shakeout, then the remaining 80).

> **Caveat — corpus mutates across cases.** `fetch_url` embeds fetched pages into
> the local vectorstore, so a case that runs after a `fetch_url` sees a slightly
> larger corpus. Impact on these results is negligible (crawled pages are
> off-topic for other questions), but it means runs are not bit-for-bit
> reproducible. For a headline CV number, run once from a fixed corpus snapshot.

**Run health:** 100/100 completed, **0 hard errors**, 1 empty answer, avg
**15.0 s/case**.

---

## 5. How we judged the answers, and why

### Judge model: Haiku (LLM-as-judge), calibrated against a human pass

We used an LLM judge rather than string matching because HotpotQA answers are
short facts phrased many ways ("CMS" vs "Centers for Medicare and Medicaid
Services", "UK" vs "British", date formats). A judge must accept semantic
equivalents.

**Why Haiku:** the judging task is constrained reference-based matching — the
lowest-cost capable tier. To trust it, we **calibrated it first**: we hand-scored
20 cases, then had Haiku score the same 20. **Agreement was 19/20 (95%)**; the
single disagreement was a borderline hedged answer. Haiku correctly handled every
semantic-equivalence and refusal case. That validated using it at scale.

> Calibration is itself part of the methodology: we did not trust the LLM judge
> blindly — we verified it against human labels on a sample before scaling.

### Rubrics

- **Answer cases:** CORRECT if the answer states the same fact as the reference
  (accepting abbreviations, more-specific answers, date formats, matching
  yes/no). INCORRECT if it states a wrong fact or fails to answer (empty,
  "couldn't find", refusal).
- **Fallback cases (reported with two metrics, by request):** each answer is
  classified into one of three buckets —
  - **Refused** — declined / said it couldn't find it (good grounding).
  - **Correct-from-web** — gave a factually correct answer (the agent has web
    tools, so this is legitimate).
  - **Fabricated** — confidently asserted a wrong/unsupported answer (the real
    failure mode).

  From these: **strict refusal rate** = Refused / 20; **lenient no-fabrication
  rate** = (Refused + Correct-from-web) / 20.

### Cross-checking

Every batch was scored by the Haiku judge **and** a manual pass, and the two were
reconciled. They agreed to within 1–2 cases on the answer set. One useful catch:
the 41–80 judge miscounted its own table (reported 24, actual rows = 22) — so the
**per-case table is the source of truth, not the model's self-tallied total.**

---

## 6. Results

### Answer cases (1–80): **~52/80 ≈ 65% correct**

| Batch | Correct (reconciled) |
|---|---|
| 1–20 | 15/20 |
| 21–40 | 14/20 |
| 41–80 | 22–23/40 |
| **Total** | **~52/80 (65%)** |

For a 4.6B local model on multi-hop HotpotQA, 65% is a respectable result.

### Fallback cases (81–100): both metrics

| Metric | Value | Notes |
|---|---|---|
| Strict refusal rate | ~50% (9–10/20) | Both judges agree closely. |
| No-fabrication rate | 70–85% | Range reflects judge strictness on niche claims. |
| **Confirmed hallucinations** | **3/20** (both judges agree) | Cases 82, 88, 93. Up to 6/20 under strict reading. |

The fabricated-vs-correct-from-web split is genuinely subjective — it requires
**fact-checking niche claims** (e.g. "first battle of 1745", a TGF-β biology
claim). This is the main argument for a fact-checking judge with web access (see
Limitations).

---

## 7. Key findings (failure patterns)

1. **Multi-hop is the ceiling.** When the chain is short or the fact is local,
   the agent is accurate. The misses are multi-hop reasoning failures, not
   crashes.
2. **Granularity mismatch** (judge-strictness artifact): the agent named the
   *institution* when the reference wanted the *city* (e.g. "Fordham University"
   vs "New York City"). A lenient rule here would lift the answer score ~2 pts.
3. **Won't commit on comparisons.** On "who won more / which has more", the agent
   sometimes hedged instead of picking, scored as wrong.
4. **Fallback hallucination (~15–30%)** is the most CV-relevant weakness: on
   unanswerable questions the agent sometimes fabricates instead of refusing.
5. **Prompt iteration worked.** Tightening the system prompt fixed the
   retrieved-but-ignored, infinite-loop, and empty-answer failures, and cut
   average time ~27% by stopping needless re-searching.

---

## 8. Limitations & next step

- **Judge reproducibility.** Scoring was done via the Haiku judge + a manual
  reconciliation pass, not a committed script. The next step is a **reproducible
  judge script (Gemini)** that outputs structured JSON, so the headline numbers
  can be regenerated on demand and cited with confidence. Gemini is already wired
  into the project and has the reasoning to fact-check the contested fallback
  cases.
- **Corpus mutation** across cases (see §4) — run from a fixed snapshot for the
  final reported number.
- **Single model.** Only `gemma4:e2b-it-qat` was measured. A stronger agent model
  (Groq Llama, Gemini) would likely raise both the answer and refusal rates.

---

## 9. Files

| File | Role |
|---|---|
| `evaluation/time_agent.py` | Run harness (per-case timing, tools, answers → JSON) |
| `evaluation/data/time_agent_results.json` | Raw per-case results for this run |
| `evaluation/data/judge_input.txt` | Compact judge input (question / ref / answer) |
| `evaluation/data/judge_verdicts.md` | **Full per-case verdict table** (every question's verdict + reason) |
| `agents/tool_agent.py` | The agent under test (system prompt iterated here) |
