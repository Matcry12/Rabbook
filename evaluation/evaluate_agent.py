"""
End-to-end evaluation of the tool agent.

Checks whether the agent:
  - routes tool calls at all (rather than answering from memory)
  - calls query_documents before web_search (local-RAG-first routing)
  - finishes within the iteration limit
  - refuses to answer unanswerable questions rather than fabricating a response

No LLM judge is used. All checks are deterministic heuristics so results
are reproducible without API credits.
"""
import warnings

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

ITERATION_LIMIT_MESSAGE = "Agent reached the iteration limit without a final answer."

# Phrases that indicate the agent acknowledged it could not find the answer.
_REFUSAL_PHRASES = [
    "don't have", "do not have", "cannot find", "not available",
    "no information", "not mentioned", "not provided", "not contain",
    "does not contain", "doesn't contain", "doesn't provide",
    "does not provide", "unable to find", "i don't know",
]


def _fallback_handled(answer: str) -> bool:
    """
    Return True when the agent appears to have refused rather than fabricated.

    Heuristic: the answer contains at least one refusal/can't-find signal
    phrase. Domain-agnostic — works for any question type, not just salary
    or compensation queries.
    """
    answer_lower = answer.lower()
    return any(phrase in answer_lower for phrase in _REFUSAL_PHRASES)


def main():
    print("Initializing models...")
    llm = build_llm()
    embeddings = build_embeddings()
    reranker = build_reranker()

    dataset = load_dataset()

    print(f"\nRunning tool agent on {len(dataset)} cases...\n")

    col_q = 60
    print(f"{'Question':<{col_q}}  {'Tools':<32}  Fin  Fallback?")
    print("-" * 110)

    total_called_a_tool = 0
    total_used_local_first = 0
    local_first_applicable = 0  # cases where query_documents appeared at all
    total_finished = 0
    total_errors = 0
    fallback_cases_total = 0
    fallback_cases_handled = 0

    for case in dataset:
        question = case["question"]
        expected_behavior = case.get("expected_behavior", "answer")

        # Run the agent. A tool may raise (e.g. a failed web fetch); we record
        # the case as an error and keep going rather than aborting the whole run.
        trace: list = []
        try:
            answer = run_tool_agent(
                question,
                llm=llm,
                embeddings=embeddings,
                reranker=reranker,
                trace=trace,
            )
        except Exception as exc:
            total_errors += 1
            tool_sequence = [step["tool"] for step in trace if "tool" in step]
            print(
                f"{question[:col_q]:<{col_q}}  {str(tool_sequence)[:32]:<32}  "
                f"ERR  {type(exc).__name__}: {str(exc)[:60]}"
            )
            continue

        tool_sequence = [step["tool"] for step in trace if "tool" in step]
        called_a_tool = len(tool_sequence) > 0
        finished = answer != ITERATION_LIMIT_MESSAGE

        # Local-first: only meaningful when query_documents appears in the sequence.
        used_local_first = False
        if "query_documents" in tool_sequence:
            local_first_applicable += 1
            used_local_first = tool_sequence[0] == "query_documents"
            if used_local_first:
                total_used_local_first += 1

        if called_a_tool:
            total_called_a_tool += 1
        if finished:
            total_finished += 1

        fallback_label = ""
        if expected_behavior == "fallback":
            fallback_cases_total += 1
            handled = _fallback_handled(answer)
            if handled:
                fallback_cases_handled += 1
            fallback_label = "OK" if handled else "FAIL"

        tool_str = str(tool_sequence)[:32]
        fin_label = "yes" if finished else "NO"
        print(
            f"{question[:col_q]:<{col_q}}  {tool_str:<32}  {fin_label:<3}  {fallback_label}"
        )

    n = len(dataset)
    print()
    print("=" * 60)
    print(f"Agent evaluation summary ({n} cases)")
    print("=" * 60)
    print(f"  Called a tool:          {total_called_a_tool}/{n}")
    print(f"  Used local RAG first:   {total_used_local_first}/{local_first_applicable}"
          f"  (of cases that called query_documents)")
    print(f"  Finished within limit:  {total_finished}/{n}")
    if fallback_cases_total:
        print(f"  Fallback handled:       {fallback_cases_handled}/{fallback_cases_total}"
              f"  (refused with a can't-find signal phrase)")


if __name__ == "__main__":
    main()
