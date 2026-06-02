"""
RAGAS evaluation for the Rabbook retrieval pipeline.

Metrics computed (no ground_truth required):
  Faithfulness      — is the answer faithful to the retrieved context?
  Answer Relevancy  — does the answer address the question?

To also measure Context Precision and Context Recall, add a "ground_truth"
field to each case in tests/eval_question.json.
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
from ragas.metrics import Faithfulness, AnswerRelevancy
from ragas.run_config import RunConfig

from core.config import (
    DB_DIR,
    DEFAULT_LLM_MODEL,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_BM25_CANDIDATE_K,
    DEFAULT_RERANK_CANDIDATE_K,
    DEFAULT_RERANK_MODEL,
    DEFAULT_RETRIEVAL_K,
    OLLAMA_BASE_URL,
    OLLAMA_NUM_GPU,
)
from rag.registry import load_chunk_registry
from rag.retrieve import (
    load_bm25_index,
    load_vectorstore,
    retrieve_documents_with_query_transform,
)

BASE_DIR = Path(__file__).resolve().parent
EVAL_PATH = BASE_DIR / "tests" / "eval_question.json"
CACHE_PATH = BASE_DIR / "tests" / "ragas_cache.json"


def build_llm():
    if DEFAULT_LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(model=DEFAULT_LLM_MODEL, temperature=0.0)
    if DEFAULT_LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(model=DEFAULT_LLM_MODEL, temperature=0.0)
    from langchain_ollama import ChatOllama
    return ChatOllama(model=DEFAULT_LLM_MODEL, base_url=OLLAMA_BASE_URL, num_gpu=OLLAMA_NUM_GPU)


def build_evaluator_llm():
    """Gemini for RAGAS evaluation — better structured output than small local models."""
    from core.config import get_google_api_key
    from langchain_google_genai import ChatGoogleGenerativeAI
    api_key = get_google_api_key()
    if not api_key:
        print("Warning: GEMINI_KEY not set — falling back to RAG LLM for evaluation.")
        return build_llm()
    return ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite", temperature=0.0, google_api_key=api_key)


def build_embeddings():
    from langchain_huggingface import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def retrieve_contexts(question: str, vectorstore, bm25_index, reranker, llm) -> list[str]:
    docs = retrieve_documents_with_query_transform(
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
    return [doc.page_content for doc, _score in docs]


def generate_answer(question: str, contexts: list[str], llm) -> str:
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

    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(DEFAULT_RERANK_MODEL)

    vectorstore = load_vectorstore(str(DB_DIR), embeddings)
    chunk_registry = load_chunk_registry()
    bm25_index = load_bm25_index(chunk_registry=chunk_registry, vectorstore=vectorstore)

    cases = json.loads(EVAL_PATH.read_text(encoding="utf-8"))
    questions = [c["question"] for c in cases]

    # Load cache so reruns skip already-collected questions
    cache: dict[str, dict] = {}
    if CACHE_PATH.exists():
        cache = json.loads(CACHE_PATH.read_text(encoding="utf-8"))
        print(f"Loaded {len(cache)} cached answers from {CACHE_PATH.name}")

    print(f"\nCollecting answers for {len(questions)} questions...")
    samples = []
    skipped = 0
    for i, question in enumerate(questions, 1):
        if question in cache:
            print(f"  [{i}/{len(questions)}] (cached) {question[:65]}...")
            entry = cache[question]
            samples.append(SingleTurnSample(
                user_input=question,
                response=entry["answer"],
                retrieved_contexts=entry["contexts"],
            ))
            continue

        print(f"  [{i}/{len(questions)}] {question[:70]}...")
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
        ))

    if not samples:
        print("\nNo samples to evaluate.")
        print("Ensure documents are ingested: python ingest_docs.py")
        return

    dataset = EvaluationDataset(samples=samples)

    evaluator_llm = LangchainLLMWrapper(build_evaluator_llm())
    evaluator_embeddings = LangchainEmbeddingsWrapper(embeddings)

    faithfulness_metric = Faithfulness(llm=evaluator_llm)
    answer_relevancy_metric = AnswerRelevancy(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    run_config = RunConfig(timeout=120, max_workers=2, max_wait=120)

    print(f"\nRunning RAGAS on {len(samples)} samples (2 workers)...")
    result = evaluate(
        dataset,
        metrics=[faithfulness_metric, answer_relevancy_metric],
        run_config=run_config,
    )

    print("\nRAGAS Results")
    print("=" * 44)
    scores = result.to_pandas()
    print(f"  Faithfulness:     {scores['faithfulness'].mean():.3f}  (answer stays within retrieved context)")
    print(f"  Answer Relevancy: {scores['answer_relevancy'].mean():.3f}  (answer addresses the question)")
    print(f"\n  Evaluated: {len(samples)}/{len(questions)} questions", end="")
    if skipped:
        print(f"  ({skipped} skipped — empty retrieval)", end="")
    print()
    print("\nTip: add 'ground_truth' to eval_question.json to unlock")
    print("     ContextPrecision and ContextRecall metrics.")


if __name__ == "__main__":
    main()
