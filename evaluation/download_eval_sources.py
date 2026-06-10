"""
Download the raw data sources for the benchmark eval corpus.

Files are written to data/eval_sources/ and only downloaded if not already
present, so this script is safe to rerun.

Sources:
  - HotpotQA distractor validation set (parquet, ~27 MB)
  - SQuAD v2 dev set (json, ~4 MB)
"""
import urllib.request
from pathlib import Path

# Anchored to project root so the script works regardless of working directory
EVAL_SOURCES_DIR = Path(__file__).resolve().parent.parent / "data" / "eval_sources"

HOTPOT_URL = (
    "https://huggingface.co/datasets/hotpotqa/hotpot_qa/resolve/main/"
    "distractor/validation-00000-of-00001.parquet"
)
HOTPOT_PATH = EVAL_SOURCES_DIR / "hotpot_validation.parquet"

SQUAD_URL = "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json"
SQUAD_PATH = EVAL_SOURCES_DIR / "squad_v2_dev.json"


def download_if_missing(url: str, dest: Path) -> None:
    """Download url to dest, following redirects. Skip if dest already exists."""
    if dest.exists():
        print(f"  Already present: {dest}  ({dest.stat().st_size:,} bytes)")
        return

    print(f"  Downloading {url}")
    print(f"  -> {dest}")

    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(request) as response:
        data = response.read()

    dest.write_bytes(data)
    print(f"  Done: {dest.stat().st_size:,} bytes")


def main() -> None:
    EVAL_SOURCES_DIR.mkdir(parents=True, exist_ok=True)

    print("Downloading HotpotQA distractor validation set...")
    download_if_missing(HOTPOT_URL, HOTPOT_PATH)

    print("Downloading SQuAD v2 dev set...")
    download_if_missing(SQUAD_URL, SQUAD_PATH)

    print("\nAll sources present.")
    for path in [HOTPOT_PATH, SQUAD_PATH]:
        print(f"  {path}: {path.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
