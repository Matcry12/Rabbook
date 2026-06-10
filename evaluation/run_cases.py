"""
Run specific dataset cases by index (1-based) through the tool agent.

Usage: python -m evaluation.run_cases 6 24 30 51 88 98
Prints, per case: time, tools, the ground truth, and the full answer — so the
answers can be judged by hand. Model is whatever RABBOOK_LLM_MODEL is set to.
"""
import sys
import time
import warnings

from dotenv import load_dotenv

load_dotenv()
warnings.filterwarnings("ignore", category=DeprecationWarning)

from core.config import DEFAULT_LLM_MODEL
from agents.tool_agent import run_tool_agent
from .eval_common import build_embeddings, build_llm, build_reranker, load_dataset


def main():
    indices = [int(a) for a in sys.argv[1:]]
    if not indices:
        print("Give 1-based case indices, e.g. python -m evaluation.run_cases 6 24 30")
        return

    print(f"Model: {DEFAULT_LLM_MODEL}")
    print("Initializing models...")
    llm = build_llm()
    embeddings = build_embeddings()
    reranker = build_reranker()

    dataset = load_dataset()

    for idx in indices:
        case = dataset[idx - 1]
        question = case["question"]
        reference = case.get("ground_truth", "")
        expected = case.get("expected_behavior", "answer")

        start = time.perf_counter()
        trace: list = []
        try:
            answer = run_tool_agent(
                question, llm=llm, embeddings=embeddings, reranker=reranker, trace=trace
            )
        except Exception as exc:
            answer = f"(ERROR) {type(exc).__name__}: {exc}"
        elapsed = time.perf_counter() - start
        tools = [s["tool"] for s in trace if "tool" in s]

        print("\n" + "=" * 80)
        print(f"CASE {idx}  ({expected})  |  {elapsed:.1f}s  |  tools={tools}")
        print(f"Q:   {question}")
        print(f"REF: {reference}")
        print(f"ANS: {answer if answer.strip() else '(empty)'}")


if __name__ == "__main__":
    main()
