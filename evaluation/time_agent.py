"""
Quick timing probe for the tool agent.

For each case it prints:
  - the wall-clock time the case STARTED (separate column)
  - how long the agent took (seconds)
  - the FIRST tool the agent called
  - the full tool sequence

Runs a small slice of the dataset (default 5) so it's fast to eyeball.
Override the count with:  python -m evaluation.time_agent 10
"""
import json
import sys
import time
import warnings
from datetime import datetime

from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", category=DeprecationWarning)

from agents.tool_agent import run_tool_agent
from .eval_common import (
    build_embeddings,
    build_llm,
    build_reranker,
    load_dataset,
)

RESULTS_PATH = "evaluation/data/time_agent_results.json"


def main():
    # Usage: python -m evaluation.time_agent [count] [offset]
    # offset > 0 runs a later slice and APPENDS to the existing results file.
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0

    print("Initializing models...")
    llm = build_llm()
    embeddings = build_embeddings()
    reranker = build_reranker()

    dataset = load_dataset()[offset:offset + n]
    print(f"\nTiming {len(dataset)} cases (offset {offset})...\n")

    col_q = 50
    print(f"{'Started at':<12}  {'Secs':>6}  {'First tool':<16}  {'Question':<{col_q}}  Tools")
    print("-" * 130)

    results = []

    for case in dataset:
        question = case["question"]
        reference = case.get("ground_truth", "")
        expected = case.get("expected_behavior", "answer")

        start_clock = datetime.now().strftime("%H:%M:%S")
        start = time.perf_counter()

        trace: list = []
        error = None
        answer = ""
        try:
            answer = run_tool_agent(
                question,
                llm=llm,
                embeddings=embeddings,
                reranker=reranker,
                trace=trace,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        elapsed = time.perf_counter() - start
        tool_sequence = [step["tool"] for step in trace if "tool" in step]
        first_tool = tool_sequence[0] if tool_sequence else "(none)"

        results.append({
            "question": question,
            "ground_truth": reference,
            "expected_behavior": expected,
            "answer": answer,
            "error": error,
            "elapsed_sec": round(elapsed, 1),
            "tools": tool_sequence,
        })

        tail = error if error else ""
        print(
            f"{start_clock:<12}  {elapsed:6.1f}  {first_tool:<16}  "
            f"{question[:col_q]:<{col_q}}  {tool_sequence} {tail}"
        )

    # When running a later slice, append to whatever is already on disk.
    if offset > 0:
        try:
            with open(RESULTS_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except FileNotFoundError:
            existing = []
        results = existing + results

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print("-" * 130)
    print(f"Wrote {len(results)} total results (with answers) to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
