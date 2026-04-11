import json
import re
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent
EVAL_PATH = BASE_DIR / "tests" / "eval_question.json"

from app.web import answer_query
from core.config import DEFAULT_GROUNDED_FALLBACK_MESSAGE


def load_eval_cases():
    return json.loads(EVAL_PATH.read_text(encoding="utf-8"))


def evaluate_case(case):
    answer, sources, citations, debug_data = answer_query(
        case["question"],
        debug_mode=True,
    )

    answer_verdict, matched_concepts = evaluate_answer(case, answer)
    grounding_verdict = evaluate_grounding(answer, citations, debug_data)

    return {
        "question": case["question"],
        "expected_behavior": case.get("expected_behavior", "answer"),
        "answer": answer,
        "answer_verdict": answer_verdict,
        "grounding_verdict": grounding_verdict,
        "matched_concepts": matched_concepts,
        "total_concepts": len(case.get("expected_concepts", [])),
        "sources": [source["chunk_id"] for source in sources],
        "citations": [citation["number"] for citation in citations],
        "grounding_reason": ((debug_data or {}).get("grounding") or {}).get("reason", "unknown"),
        "notes": case.get("notes", ""),
    }


def evaluate_answer(case, answer):
    expected_behavior = case.get("expected_behavior", "answer")
    answer_text = normalize_text(answer)
    is_fallback = answer_text == normalize_text(DEFAULT_GROUNDED_FALLBACK_MESSAGE)

    if expected_behavior == "fallback":
        return ("correct" if is_fallback else "wrong"), 0

    expected_concepts = case.get("expected_concepts", [])
    forbidden_concepts = case.get("forbidden_concepts", [])
    if not expected_concepts:
        return ("partially_correct", 0)

    for concept_group in forbidden_concepts:
        if any(normalize_text(phrase) in answer_text for phrase in concept_group):
            return "wrong", 0

    matched_concepts = 0
    for concept_group in expected_concepts:
        if any(normalize_text(phrase) in answer_text for phrase in concept_group):
            matched_concepts += 1

    if matched_concepts == len(expected_concepts):
        return "correct", matched_concepts
    if matched_concepts > 0:
        return "partially_correct", matched_concepts
    return "wrong", matched_concepts


def evaluate_grounding(answer, citations, debug_data):
    grounding = (debug_data or {}).get("grounding") or {}
    answer_text = normalize_text(answer)
    is_fallback = answer_text == normalize_text(DEFAULT_GROUNDED_FALLBACK_MESSAGE)

    if grounding.get("passed") is True and citations:
        return "grounded_answer"
    if is_fallback and grounding.get("passed") is False:
        return "safe_fallback"
    if citations:
        return "partially_grounded"
    return "not_grounded"


def build_summary(results):
    correct_answers = [result for result in results if result["answer_verdict"] == "correct"]
    partially_correct_answers = [
        result for result in results if result["answer_verdict"] == "partially_correct"
    ]
    grounded_answers = [
        result for result in results if result["grounding_verdict"] == "grounded_answer"
    ]
    safe_fallbacks = [
        result for result in results if result["grounding_verdict"] == "safe_fallback"
    ]
    partially_grounded_answers = [
        result for result in results if result["grounding_verdict"] == "partially_grounded"
    ]

    return {
        "cases": len(results),
        "correct_answers": len(correct_answers),
        "partially_correct_answers": len(partially_correct_answers),
        "grounded_answers": len(grounded_answers),
        "safe_fallbacks": len(safe_fallbacks),
        "partially_grounded_answers": len(partially_grounded_answers),
    }


def print_report(summary, results):
    print("RAG Answer Evaluation Report")
    print("=" * 32)
    print(f"Cases: {summary['cases']}")
    print(f"Correct Answers: {summary['correct_answers']}/{summary['cases']}")
    print(f"Partially Correct Answers: {summary['partially_correct_answers']}/{summary['cases']}")
    print(f"Grounded Answers: {summary['grounded_answers']}/{summary['cases']}")
    print(f"Safe Fallbacks: {summary['safe_fallbacks']}/{summary['cases']}")
    print(
        f"Partially Grounded Answers: "
        f"{summary['partially_grounded_answers']}/{summary['cases']}"
    )
    print()

    print("Question Results")
    print("-" * 32)
    for result in results:
        print(result["question"])
        print(f"Answer Verdict: {result['answer_verdict']}")
        print(f"Grounding Verdict: {result['grounding_verdict']}")
        print(f"Grounding Reason: {result['grounding_reason']}")
        if result["total_concepts"]:
            print(
                f"Matched Concepts: {result['matched_concepts']}/{result['total_concepts']}"
            )
        print(f"Cited Sources: {result['citations']}")
        print(f"Retrieved Chunks: {result['sources']}")
        print(f"Answer: {result['answer']}")
        if result["notes"]:
            print(f"Notes: {result['notes']}")
        print()


def normalize_text(text):
    return re.sub(r"\s+", " ", text.lower()).strip()


def main():
    load_dotenv()

    cases = load_eval_cases()
    results = [evaluate_case(case) for case in cases]
    summary = build_summary(results)
    print_report(summary, results)


if __name__ == "__main__":
    raise SystemExit(
        "Run this from the project root with `python evaluate_retrieval.py`, not `python tests/evaluate_retrieval.py`."
    )
