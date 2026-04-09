import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "tests" / "eval_question.json"

from core.config import DB_DIR, DEFAULT_CONTEXT_WINDOW, DEFAULT_RETRIEVAL_K
from rag.registry import load_chunk_registry
from rag.retrieve import expand_with_context_window, load_vectorstore, retrieve_documents


def load_eval_cases():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def build_embeddings():
    try:
        return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    except Exception as exc:
        raise RuntimeError(
            "Failed to load embedding model 'sentence-transformers/all-MiniLM-L6-v2'. "
            "Make sure the model is cached locally or that this machine has network access "
            "to Hugging Face before running retrieval evaluation."
        ) from exc


def evaluate_case(case, vectorstore, chunk_registry, k=DEFAULT_RETRIEVAL_K):
    question = case["question"]
    relevant_chunk_ids = set(case.get("relevant_chunk_ids", []))
    is_labeled = bool(relevant_chunk_ids)

    retrieved = retrieve_documents(vectorstore, question, k=k)
    expanded = expand_with_context_window(
        retrieved,
        chunk_registry,
        window_size=DEFAULT_CONTEXT_WINDOW,
    )

    retrieved_chunk_ids = [
        doc.metadata.get("chunk_id")
        for doc, _ in expanded
        if doc.metadata.get("chunk_id")
    ]

    matched_chunk_ids = [
        chunk_id for chunk_id in retrieved_chunk_ids if chunk_id in relevant_chunk_ids
    ]

    recall = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled)
    hit_rate = hit_rate_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled)

    return {
        "question": question,
        "is_labeled": is_labeled,
        "notes": case.get("notes", ""),
        "relevant_chunk_ids": sorted(relevant_chunk_ids),
        "retrieved_chunk_ids": retrieved_chunk_ids,
        "matched_chunk_ids": matched_chunk_ids,
        "precision_at_k": precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled),
        "recall_at_k": recall,
        "f1_at_k": f1_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled),
        "hit_rate_at_k": hit_rate,
        "mrr": mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids, is_labeled),
        "verdict": build_verdict(is_labeled, hit_rate, recall),
    }


def precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled):
    if not is_labeled:
        return None

    if not retrieved_chunk_ids:
        return 0.0

    hits = sum(1 for chunk_id in retrieved_chunk_ids if chunk_id in relevant_chunk_ids)
    return hits / len(retrieved_chunk_ids)


def recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled):
    if not is_labeled:
        return None

    hits = sum(1 for chunk_id in retrieved_chunk_ids if chunk_id in relevant_chunk_ids)
    return hits / len(relevant_chunk_ids)


def f1_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled):
    if not is_labeled:
        return None

    precision = precision_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled)
    recall = recall_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)


def hit_rate_at_k(retrieved_chunk_ids, relevant_chunk_ids, is_labeled):
    if not is_labeled:
        return None

    return 1.0 if any(chunk_id in relevant_chunk_ids for chunk_id in retrieved_chunk_ids) else 0.0


def mean_reciprocal_rank(retrieved_chunk_ids, relevant_chunk_ids, is_labeled):
    if not is_labeled:
        return None

    for rank, chunk_id in enumerate(retrieved_chunk_ids, start=1):
        if chunk_id in relevant_chunk_ids:
            return 1.0 / rank
    return 0.0


def build_summary(results):
    labeled_results = [result for result in results if result["is_labeled"]]
    passed_results = [result for result in labeled_results if result["verdict"] == "PASS"]

    return {
        "cases": len(results),
        "labeled_cases": len(labeled_results),
        "passed_cases": len(passed_results),
        "avg_precision_at_k": average(result["precision_at_k"] for result in labeled_results),
        "avg_recall_at_k": average(result["recall_at_k"] for result in labeled_results),
        "avg_f1_at_k": average(result["f1_at_k"] for result in labeled_results),
        "avg_hit_rate_at_k": average(result["hit_rate_at_k"] for result in labeled_results),
        "avg_mrr": average(result["mrr"] for result in labeled_results),
    }


def average(values):
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_verdict(is_labeled, hit_rate, recall):
    if not is_labeled:
        return "SKIP"
    if hit_rate == 1.0 and recall == 1.0:
        return "PASS"
    if hit_rate == 1.0:
        return "PARTIAL"
    return "FAIL"


def print_report(summary, results):
    print("RAG Retrieval Report")
    print("=" * 30)
    print(f"Cases: {summary['cases']}")
    print(f"Labeled Cases: {summary['labeled_cases']}")
    print(f"Passed Cases: {summary['passed_cases']}/{summary['labeled_cases']}")
    print(f"Hit Rate@k: {summary['avg_hit_rate_at_k']:.3f}")
    print(f"Recall@k: {summary['avg_recall_at_k']:.3f}")
    print(f"MRR: {summary['avg_mrr']:.3f}")
    print()

    print("Question Results")
    print("-" * 30)
    for result in results:
        print(f"[{result['verdict']}] {result['question']}")
        if not result["is_labeled"]:
            print("Reason: no labeled relevant chunks yet")
            if result["notes"]:
                print(f"Notes: {result['notes']}")
            print()
            continue
        print(f"Recall@k: {result['recall_at_k']:.3f}")
        print(f"Hit Rate@k: {result['hit_rate_at_k']:.3f}")
        print(f"MRR: {result['mrr']:.3f}")
        print(f"Relevant Chunks: {result['relevant_chunk_ids']}")
        print(f"Matched Chunks: {result['matched_chunk_ids']}")
        print(f"Retrieved Chunks: {result['retrieved_chunk_ids']}")
        print()


def main():
    load_dotenv()

    embeddings = build_embeddings()
    vectorstore = load_vectorstore(str(DB_DIR), embeddings)
    chunk_registry = load_chunk_registry()

    cases = load_eval_cases()
    results = [
        evaluate_case(case, vectorstore, chunk_registry)
        for case in cases
    ]
    summary = build_summary(results)
    print_report(summary, results)


if __name__ == "__main__":
    raise SystemExit(
        "Run this from the project root with `python evaluate_retrieval.py`, not `python tests/evaluate_retrieval.py`."
    )
