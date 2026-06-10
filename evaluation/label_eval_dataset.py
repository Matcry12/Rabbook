"""
Map gold sentences to chunk ids produced by semantic chunking.

Run AFTER ingestion (reingest_directory on data/eval_corpus/) so the chunk
registry reflects the current index.

Reads:  evaluation/data/eval_raw.json, data/chunk_registry.json
Writes: evaluation/data/eval_dataset.json
"""
import json
import re
from pathlib import Path

from core.config import REGISTRY_PATH

# Anchored paths — work from any working directory when run as a module
_PACKAGE_DIR = Path(__file__).resolve().parent   # evaluation/

RAW_EVAL_PATH = _PACKAGE_DIR / "data" / "eval_raw.json"
FINAL_EVAL_PATH = _PACKAGE_DIR / "data" / "eval_dataset.json"

SHORT_SENTENCE_THRESHOLD = 40   # chars; short sentences require exact match
MATCH_PREFIX_LENGTH = 80        # chars; prefix used for longer sentences

FALLBACK_GROUND_TRUTH = (
    "The ingested documents do not contain information to answer this question."
)


def normalize(text: str) -> str:
    """Lowercase and collapse all whitespace runs to single spaces."""
    return re.sub(r"\s+", " ", text.lower()).strip()


def build_chunk_index(registry: dict) -> list[tuple[str, str]]:
    """
    Return a list of (chunk_id, normalized_page_content) from the registry.

    Uses the by_chunk_id index for direct access.
    """
    index = []
    for chunk_id, chunk in registry["by_chunk_id"].items():
        normalized_content = normalize(chunk["page_content"])
        index.append((chunk_id, normalized_content))
    return index


def find_matching_chunks(sentence: str, chunk_index: list[tuple[str, str]]) -> list[str]:
    """
    Return chunk_ids whose normalized content contains the gold sentence.

    Short sentences (<40 chars normalized) require an exact full-sentence match.
    Longer sentences match on the first 80 normalized chars to tolerate
    chunk-boundary splits.
    """
    normalized_sentence = normalize(sentence)
    is_short = len(normalized_sentence) < SHORT_SENTENCE_THRESHOLD

    if is_short:
        search_token = normalized_sentence
    else:
        search_token = normalized_sentence[:MATCH_PREFIX_LENGTH]

    matched_ids = []
    for chunk_id, normalized_content in chunk_index:
        if search_token in normalized_content:
            matched_ids.append(chunk_id)
    return matched_ids


def build_notes(record: dict, matched_titles: list[str]) -> str:
    """Build the notes string for a final eval record."""
    source = record.get("source", "")
    record_type = record.get("type", "")

    if source == "hotpotqa":
        titles_str = " | ".join(matched_titles) if matched_titles else "(none)"
        parts = [f"hotpotqa multi-hop; gold titles: {titles_str}"]
        if record_type:
            parts[0] += f"; type: {record_type}"
        return parts[0]
    else:
        return "squad_v2 unanswerable; topic absent from corpus"


def main() -> None:
    print(f"Loading raw eval spec from {RAW_EVAL_PATH} ...")
    raw_records = json.loads(RAW_EVAL_PATH.read_text(encoding="utf-8"))
    print(f"  Loaded {len(raw_records)} records.")

    print(f"Loading chunk registry from {REGISTRY_PATH} ...")
    registry = json.loads(Path(REGISTRY_PATH).read_text(encoding="utf-8"))
    chunk_index = build_chunk_index(registry)
    print(f"  Indexed {len(chunk_index)} chunks.")

    final_records = []
    zero_match_cases = []
    total_supporting_facts = 0
    total_unmatched_facts = 0

    for record in raw_records:
        expected_behavior = record["expected_behavior"]
        question = record["question"]

        if expected_behavior == "fallback":
            final_records.append({
                "question": question,
                "ground_truth": FALLBACK_GROUND_TRUTH,
                "relevant_chunk_ids": [],
                "expected_behavior": "fallback",
                "notes": build_notes(record, []),
            })
            continue

        # Answer case: find matching chunks for each supporting fact.
        supporting_facts = record.get("supporting", [])
        relevant_chunk_ids: list[str] = []
        matched_titles: list[str] = []
        unmatched_in_record = 0

        for fact in supporting_facts:
            total_supporting_facts += 1
            title = fact["title"]
            sentence = fact["sentence"]

            matched = find_matching_chunks(sentence, chunk_index)
            if matched:
                for chunk_id in matched:
                    if chunk_id not in relevant_chunk_ids:
                        relevant_chunk_ids.append(chunk_id)
                if title not in matched_titles:
                    matched_titles.append(title)
            else:
                unmatched_in_record += 1
                total_unmatched_facts += 1

        notes = build_notes(record, matched_titles)
        if unmatched_in_record > 0 and len(relevant_chunk_ids) == 0:
            notes += " (WARNING: gold sentences not found in any chunk)"

        final_records.append({
            "question": question,
            "ground_truth": record["answer"],
            "relevant_chunk_ids": relevant_chunk_ids,
            "expected_behavior": "answer",
            "notes": notes,
        })

        if len(relevant_chunk_ids) == 0:
            zero_match_cases.append(question)

    FINAL_EVAL_PATH.write_text(
        json.dumps(final_records, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(final_records)} records to {FINAL_EVAL_PATH}")

    # Summary statistics
    answer_cases = [r for r in final_records if r["expected_behavior"] == "answer"]
    fallback_cases = [r for r in final_records if r["expected_behavior"] == "fallback"]
    total_chunk_ids = sum(len(r["relevant_chunk_ids"]) for r in answer_cases)
    avg_chunks = total_chunk_ids / len(answer_cases) if answer_cases else 0.0

    print("\n--- Summary ---")
    print(f"  Total records:            {len(final_records)}")
    print(f"  Answer cases:             {len(answer_cases)}")
    print(f"  Fallback cases:           {len(fallback_cases)}")
    print(f"  Avg relevant chunks/case: {avg_chunks:.2f}")
    print(f"  Supporting facts total:   {total_supporting_facts}")
    print(f"  Supporting facts unmatched: {total_unmatched_facts}")
    print(f"  Answer cases with 0 matched chunks: {len(zero_match_cases)}")

    if zero_match_cases:
        print("\n  First 5 zero-match questions (label failures to investigate):")
        for q in zero_match_cases[:5]:
            print(f"    - {q[:100]}")


if __name__ == "__main__":
    main()
