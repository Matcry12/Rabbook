"""
RAGAS evaluation for the Rabbook retrieval pipeline.

Metrics computed (requires ground_truth from eval_dataset.json):
  Faithfulness                    — is the answer faithful to the retrieved context?
  AnswerRelevancy                 — does the answer address the question?
  AnswerCorrectness               — does the answer match the ground truth?
  LLMContextRecall                — does the context cover the ground truth?
  LLMContextPrecisionWithReference — are the retrieved chunks relevant to the ground truth?

Only "answer" cases (expected_behavior == "answer") are evaluated.
Fallback cases have no ground-truth context, so RAGAS metrics do not apply.
"""
import json
import warnings
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)

from ragas import EvaluationDataset, SingleTurnSample, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    AnswerCorrectness,
    AnswerRelevancy,
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.run_config import RunConfig

from .eval_common import (
    build_embeddings,
    build_evaluator_llm,
    build_llm,
    build_reranker,
    load_dataset,
    load_retrieval_bundle,
)
from core.config import DEFAULT_RETRIEVAL_K
from rag.retrieve import retrieve_documents_with_query_transform
from core.config import DEFAULT_BM25_CANDIDATE_K, DEFAULT_RERANK_CANDIDATE_K

# Cache lives alongside the other eval data artifacts
CACHE_PATH = Path(__file__).resolve().parent / "data" / "ragas_cache.json"


def retrieve_contexts(question: str, vectorstore, bm25_index, reranker, llm) -> list[str]:
    """Run retrieval for a question and return a list of page-content strings."""
    documents = retrieve_documents_with_query_transform(
        vectorstore,
        question,
        k=DEFAULT_RETRIEVAL_K,
        reranker=reranker,
        bm25_index=bm25_index,
        query_transformer=llm,
        enable_query_transform=False,
        candidate_k=DEFAULT_RERANK_CANDIDATE_K,
        bm25_candidate_k=DEFAULT_BM25_CANDIDATE_K,
        metadata_filter=None,
        include_debug=False,
    )
    return [doc.page_content for doc, _score in documents]


def generate_answer(question: str, contexts: list[str], llm) -> str:
    """Generate a concise answer from the retrieved contexts."""
    from langchain_core.messages import HumanMessage
    context_block = "\n\n".join(contexts)
    prompt = (
        "Answer the question using only the context below. "
        "Be concise and specific.\n\n"
        f"Context:\n{context_block}\n\n"
        f"Question: {question}"
    )
    return llm.invoke([HumanMessage(content=prompt)]).content


def main():
    print("Initializing models...")
    llm = build_llm()
    embeddings = build_embeddings()
    reranker = build_reranker()

    print("Loading retrieval bundle...")
    vectorstore, bm25_index = load_retrieval_bundle(embeddings)

    dataset = load_dataset()

    # Only RAGAS-evaluate cases that have a ground truth and are expected to answer.
    answer_cases = [c for c in dataset if c.get("expected_behavior") == "answer"]
    fallback_cases = [c for c in dataset if c.get("expected_behavior") != "answer"]
    print(f"Dataset: {len(answer_cases)} answer cases, {len(fallback_cases)} fallback case(s) skipped.")

    # Load cache so reruns skip already-collected questions.
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached answers from {CACHE_PATH.name}")

    print(f"\nCollecting answers for {len(answer_cases)} questions...")
    samples = []
    skipped = 0

    for i, case in enumerate(answer_cases, 1):
        question = case["question"]
        ground_truth = case.get("ground_truth", "")

        if question in cache:
            print(f"  [{i}/{len(answer_cases)}] (cached) {question[:65]}...")
            entry = cache[question]
            samples.append(SingleTurnSample(
                user_input=question,
                response=entry["answer"],
                retrieved_contexts=entry["contexts"],
                reference=ground_truth,
            ))
            continue

        print(f"  [{i}/{len(answer_cases)}] {question[:70]}...")
        contexts = retrieve_contexts(question, vectorstore, bm25_index, reranker, llm)
        if not contexts:
            print("    skipped — no context retrieved")
            skipped += 1
            continue

        answer = generate_answer(question, contexts, llm)

        cache[question] = {"answer": answer, "contexts": contexts}
        CACHE_PATH.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")

        samples.append(SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        ))

    if not samples:
        print("\nNo samples to evaluate.")
        print("Ensure documents are ingested: python ingest_docs.py")
        return

    dataset_obj = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(build_evaluator_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    answer_correctness_metric = AnswerCorrectness(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )
    context_recall_metric = LLMContextRecall(llm=evaluator_llm)
    context_precision_metric = LLMContextPrecisionWithReference(llm=evaluator_llm)

    run_config = RunConfig(timeout=120, max_workers=2, max_wait=120)

    print(f"\nRunning RAGAS on {len(samples)} samples (2 workers)...")
    result = evaluate(
        dataset_obj,
        metrics=[
            faithfulness_metric,
            answer_relevancy_metric,
            answer_correctness_metric,
            context_recall_metric,
            context_precision_metric,
        ],
        run_config=run_config,
    )

    scores = result.to_pandas()
    print("\nRAGAS Results")
    print("=" * 60)
    print(f"  Faithfulness:              {scores['faithfulness'].mean():.3f}  (answer stays within retrieved context)")
    print(f"  Answer Relevancy:          {scores['answer_relevancy'].mean():.3f}  (answer addresses the question)")
    print(f"  Answer Correctness:        {scores['answer_correctness'].mean():.3f}  (answer matches the ground truth)")
    print(f"  Context Recall:            {scores['context_recall'].mean():.3f}  (context covers the ground truth)")
    print(f"  Context Precision:         {scores['llm_context_precision_with_reference'].mean():.3f}  (retrieved chunks are relevant to ground truth)")
    print(f"\n  Evaluated: {len(samples)}/{len(answer_cases)} answer cases", end="")
    if skipped:
        print(f"  ({skipped} skipped — empty retrieval)", end="")
    print()
    print(f"  Skipped:   {len(fallback_cases)} fallback case(s) — no ground-truth context to judge against")


if __name__ == "__main__":
    main()
