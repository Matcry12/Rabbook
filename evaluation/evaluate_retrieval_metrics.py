"""
Deterministic IR metrics for the Rabbook retrieval pipeline.

No LLM judging. Measures whether the retriever fetches the labelled chunks
from the golden dataset. Run this first — it is fast and free.

Metrics reported at k=DEFAULT_RETRIEVAL_K:
  Hit@k       — was at least one relevant chunk in the top-k results?
  Recall@k    — what fraction of relevant chunks appeared in the top-k?
  Precision@k — what fraction of the top-k results were relevant?
  MRR         — mean reciprocal rank of the first relevant chunk hit
"""
import warnings

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)

from core.config import DEFAULT_RETRIEVAL_K
from .eval_common import (
    build_embeddings,
    build_llm,
    build_reranker,
    load_dataset,
    load_retrieval_bundle,
    retrieve_chunk_ids,
)


# ---------------------------------------------------------------------------
# Pure metric functions
# ---------------------------------------------------------------------------

def hit_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """1.0 if any of the top-k predicted ids appear in the relevant set."""
    return 1.0 if any(chunk_id in relevant_ids for chunk_id in predicted_ids[:k]) else 0.0


def recall_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of relevant chunks that appear in the top-k results."""
    if not relevant_ids:
        return 0.0
    hits = len(set(predicted_ids[:k]) & relevant_ids)
    return hits / len(relevant_ids)


def precision_at_k(predicted_ids: list[str], relevant_ids: set[str], k: int) -> float:
    """Fraction of the top-k results that are relevant."""
    if k == 0:
        return 0.0
    hits = len(set(predicted_ids[:k]) & relevant_ids)
    return hits / k


def reciprocal_rank(predicted_ids: list[str], relevant_ids: set[str]) -> float:
    """1/rank of the first relevant chunk in the ranked list (1-indexed). 0 if none found."""
    for rank, chunk_id in enumerate(predicted_ids, start=1):
        if chunk_id in relevant_ids:
            return 1.0 / rank
    return 0.0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Initializing models...")
    embeddings = build_embeddings()
    reranker = build_reranker()
    llm = build_llm()

    print("Loading retrieval bundle...")
    vectorstore, bm25_index = load_retrieval_bundle(embeddings)

    dataset = load_dataset()

    # Only evaluate cases that have labelled relevant chunks ("answer" cases).
    answer_cases = [c for c in dataset if c.get("relevant_chunk_ids")]
    fallback_cases = [c for c in dataset if not c.get("relevant_chunk_ids")]

    k = DEFAULT_RETRIEVAL_K
    print(f"\nEvaluating {len(answer_cases)} answer cases at k={k}...")
    print(f"(Skipping {len(fallback_cases)} fallback case(s) — no relevant_chunk_ids labelled)\n")

    col_q = 62
    print(
        f"{'Question':<{col_q}}  {'Hit':>4}  {'Rec':>5}  {'Pre':>5}  {'RR':>5}"
        f"  Predicted  /  Relevant"
    )
    print("-" * 120)

    hit_scores: list[float] = []
    recall_scores: list[float] = []
    precision_scores: list[float] = []
    rr_scores: list[float] = []

    for case in answer_cases:
        question = case["question"]
        relevant_ids = set(case["relevant_chunk_ids"])

        predicted_ids = retrieve_chunk_ids(
            question, vectorstore, bm25_index, reranker, llm, k=k
        )

        hit = hit_at_k(predicted_ids, relevant_ids, k)
        rec = recall_at_k(predicted_ids, relevant_ids, k)
        pre = precision_at_k(predicted_ids, relevant_ids, k)
        rr = reciprocal_rank(predicted_ids, relevant_ids)

        hit_scores.append(hit)
        recall_scores.append(rec)
        precision_scores.append(pre)
        rr_scores.append(rr)

        q_label = question[:col_q]
        predicted_label = str(predicted_ids)
        relevant_label = str(sorted(relevant_ids))
        print(
            f"{q_label:<{col_q}}  {hit:>4.2f}  {rec:>5.3f}  {pre:>5.3f}  {rr:>5.3f}"
            f"  {predicted_label}  /  {relevant_label}"
        )

    if not hit_scores:
        print("No answer cases to evaluate.")
        return

    n = len(hit_scores)
    print()
    print("=" * 60)
    print(f"Macro-averages over {n} evaluated cases (k={k})")
    print("=" * 60)
    print(f"  Hit@{k}:       {sum(hit_scores) / n:.3f}  (fraction of cases with ≥1 relevant chunk in top-k)")
    print(f"  Recall@{k}:    {sum(recall_scores) / n:.3f}  (fraction of labelled chunks retrieved)")
    print(f"  Precision@{k}: {sum(precision_scores) / n:.3f}  (fraction of retrieved chunks that are relevant)")
    print(f"  MRR:          {sum(rr_scores) / n:.3f}  (mean reciprocal rank of first relevant chunk)")
    print(f"\n  Note: {len(fallback_cases)} fallback case(s) skipped — they have no labelled relevant_chunk_ids.")


if __name__ == "__main__":
    main()
