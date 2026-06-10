"""
Build the evaluation corpus and raw eval spec from HotpotQA + SQuAD v2.

Outputs:
  - data/eval_corpus/*.txt          one file per unique paragraph title
  - evaluation/data/eval_raw.json   100 unannotated eval cases (no chunk ids yet)

Run download_eval_sources.py first.
"""
import hashlib
import json
import random
import re
from pathlib import Path

import pandas as pd

# ── constants ────────────────────────────────────────────────────────────────

N_HOTPOT = 80
N_FALLBACK = 20
RANDOM_SEED = 13

# Anchored paths — work from any working directory when run as a module
_PACKAGE_DIR = Path(__file__).resolve().parent   # evaluation/
_PROJECT_ROOT = _PACKAGE_DIR.parent              # project root

EVAL_SOURCES_DIR = _PROJECT_ROOT / "data" / "eval_sources"
HOTPOT_PATH = EVAL_SOURCES_DIR / "hotpot_validation.parquet"
SQUAD_PATH = EVAL_SOURCES_DIR / "squad_v2_dev.json"

CORPUS_DIR = _PROJECT_ROOT / "data" / "eval_corpus"
RAW_EVAL_PATH = _PACKAGE_DIR / "data" / "eval_raw.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def slug(text: str) -> str:
    """Convert a title to a filesystem-safe slug (lowercase, hyphens)."""
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    text = re.sub(r"[\s_]+", "-", text.strip())
    return text


def title_hash(title: str) -> str:
    """Return the first 8 hex chars of the sha256 of a title string."""
    return hashlib.sha256(title.encode("utf-8")).hexdigest()[:8]


def corpus_filename(title: str) -> str:
    """Return a safe, unique filename for a paragraph with this title."""
    return f"{slug(title)[:60]}-{title_hash(title)}.txt"


def inspect_hotpot_schema(df: pd.DataFrame) -> None:
    """Print the columns and one row to verify the parquet schema."""
    print("HotpotQA columns:", list(df.columns))
    first = df.iloc[0]
    print("  question:", first["question"])
    print("  answer:", first["answer"])
    print("  type:", first["type"])
    context = first["context"]
    print("  context type:", type(context))
    print("  context keys:", list(context.keys()) if hasattr(context, "keys") else "N/A")
    titles = context["title"]
    sentences = context["sentences"]
    print(f"  context paragraphs: {len(titles)}")
    print(f"  first title: {titles[0]}")
    print(f"  first para sentences[0]: {sentences[0][0][:80]!r}")
    sf = first["supporting_facts"]
    print("  supporting_facts keys:", list(sf.keys()) if hasattr(sf, "keys") else "N/A")
    print("  supporting_facts titles:", sf["title"])
    print("  supporting_facts sent_id:", sf["sent_id"])


# ── step 1: load and sample hotpot ───────────────────────────────────────────

def load_hotpot_sample() -> list:
    """Load the HotpotQA parquet and return N_HOTPOT deterministically sampled rows."""
    print(f"Loading {HOTPOT_PATH} ...")
    df = pd.read_parquet(HOTPOT_PATH)
    print(f"  Loaded {len(df)} rows.")

    print("\n--- HotpotQA schema inspection ---")
    inspect_hotpot_schema(df)
    print("----------------------------------\n")

    rng = random.Random(RANDOM_SEED)
    indices = list(range(len(df)))
    rng.shuffle(indices)
    selected = [df.iloc[i] for i in indices[:N_HOTPOT]]
    return selected


# ── step 2: collect unique paragraphs ────────────────────────────────────────

def collect_paragraphs(rows: list) -> dict[str, str]:
    """
    Return {title: paragraph_text} for every unique title across all rows.

    If the same title appears in multiple questions, its sentence list from
    the first occurrence is used (they are identical in the dataset).
    """
    paragraphs: dict[str, str] = {}
    for row in rows:
        context = row["context"]
        titles = context["title"]
        sentences_list = context["sentences"]
        for title, sentences in zip(titles, sentences_list):
            if title not in paragraphs:
                paragraphs[title] = " ".join(sentences)
    return paragraphs


# ── step 3: write corpus files ───────────────────────────────────────────────

def write_corpus(paragraphs: dict[str, str]) -> None:
    """Clear data/eval_corpus/ and write one .txt file per paragraph."""
    if CORPUS_DIR.exists():
        for f in CORPUS_DIR.iterdir():
            if f.suffix == ".txt":
                f.unlink()
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)

    for title, content in paragraphs.items():
        filename = corpus_filename(title)
        (CORPUS_DIR / filename).write_text(content, encoding="utf-8")

    print(f"Wrote {len(paragraphs)} corpus files to {CORPUS_DIR}/")


# ── step 4: build hotpot eval records ────────────────────────────────────────

def build_hotpot_records(rows: list) -> tuple[list[dict], int]:
    """
    Build one eval record per HotpotQA row.

    Returns (records, skipped_supporting_fact_count).
    """
    records = []
    skipped_facts = 0

    for row in rows:
        context = row["context"]
        titles = context["title"]
        sentences_list = context["sentences"]
        sf = row["supporting_facts"]
        sf_titles = sf["title"]
        sf_sent_ids = sf["sent_id"]

        # Build a title -> sentence list index for this row.
        title_to_sentences: dict[str, list[str]] = {}
        for title, sentences in zip(titles, sentences_list):
            title_to_sentences[title] = list(sentences)

        supporting = []
        for sf_title, sf_sent_id in zip(sf_titles, sf_sent_ids):
            sentence_list = title_to_sentences.get(sf_title, [])
            if sf_sent_id >= len(sentence_list):
                skipped_facts += 1
                continue
            sentence_text = sentence_list[sf_sent_id]
            supporting.append({"title": sf_title, "sentence": sentence_text})

        records.append({
            "question": str(row["question"]),
            "answer": str(row["answer"]),
            "expected_behavior": "answer",
            "supporting": supporting,
            "source": "hotpotqa",
            "type": str(row["type"]),
        })

    return records, skipped_facts


# ── step 5: load squad unanswerable questions ─────────────────────────────────

def load_squad_fallback_records(corpus_titles: set[str]) -> list[dict]:
    """
    Return N_FALLBACK unanswerable SQuAD v2 questions that don't overlap corpus titles.

    Skips any question whose text contains a corpus title (case-insensitive).
    """
    print(f"Loading {SQUAD_PATH} ...")
    squad_data = json.loads(SQUAD_PATH.read_text(encoding="utf-8"))

    corpus_titles_lower = {t.lower() for t in corpus_titles}

    impossible_questions: list[str] = []
    for article in squad_data["data"]:
        for paragraph in article["paragraphs"]:
            for qa in paragraph["qas"]:
                if not qa.get("is_impossible", False):
                    continue
                question_text = qa["question"]
                question_lower = question_text.lower()
                # Skip if any corpus title appears in the question text.
                if any(title in question_lower for title in corpus_titles_lower):
                    continue
                impossible_questions.append(question_text)

    print(f"  Found {len(impossible_questions)} unanswerable questions after title filter.")

    rng = random.Random(RANDOM_SEED)
    rng.shuffle(impossible_questions)
    selected = impossible_questions[:N_FALLBACK]

    records = []
    for question in selected:
        records.append({
            "question": question,
            "answer": "",
            "expected_behavior": "fallback",
            "supporting": [],
            "source": "squad_v2",
        })
    return records


# ── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    random.seed(RANDOM_SEED)
    RAW_EVAL_PATH.parent.mkdir(parents=True, exist_ok=True)

    # Step 1: load hotpot sample
    hotpot_rows = load_hotpot_sample()

    # Step 2: collect unique paragraphs
    paragraphs = collect_paragraphs(hotpot_rows)

    # Step 3: write corpus files
    write_corpus(paragraphs)

    # Step 4: build hotpot eval records
    hotpot_records, skipped_facts = build_hotpot_records(hotpot_rows)
    print(f"Built {len(hotpot_records)} HotpotQA eval records.")
    if skipped_facts:
        print(f"  Skipped {skipped_facts} supporting facts (sent_id out of range).")

    # Step 5: load squad fallback records
    squad_records = load_squad_fallback_records(set(paragraphs.keys()))

    # Step 6: combine and write
    combined = hotpot_records + squad_records
    RAW_EVAL_PATH.write_text(
        json.dumps(combined, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"\nWrote {len(combined)} records to {RAW_EVAL_PATH}")

    # Step 7: summary
    answer_count = sum(1 for r in combined if r["expected_behavior"] == "answer")
    fallback_count = sum(1 for r in combined if r["expected_behavior"] == "fallback")
    print("\n--- Summary ---")
    print(f"  Corpus files:   {len(paragraphs)}")
    print(f"  Answer cases:   {answer_count}")
    print(f"  Fallback cases: {fallback_count}")
    print(f"  Total records:  {len(combined)}")
    if skipped_facts:
        print(f"  Skipped supporting facts: {skipped_facts}")

    print("\nSample records (first 2):")
    for record in combined[:2]:
        print(json.dumps(record, indent=2, ensure_ascii=False)[:400])
        print("  ...")


if __name__ == "__main__":
    main()
