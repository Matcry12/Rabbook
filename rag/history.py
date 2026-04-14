import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from core.config import HISTORY_PATH


def load_history(history_path=HISTORY_PATH):
    path = Path(history_path)
    if not path.exists():
        return []

    content = json.loads(path.read_text(encoding="utf-8"))
    records = content if isinstance(content, list) else []
    return sorted(records, key=lambda item: item.get("created_at", ""), reverse=True)


def save_history_entry(
    query,
    answer,
    citations=None,
    selected_file="",
    selected_file_type="",
    page_start="",
    page_end="",
    history_path=HISTORY_PATH,
):
    records = load_history(history_path)
    entry = {
        "history_id": uuid4().hex[:12],
        "query": query,
        "answer": answer,
        "citations": citations or [],
        "selected_file": selected_file,
        "selected_file_type": selected_file_type,
        "page_start": page_start,
        "page_end": page_end,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    records.insert(0, entry)
    Path(history_path).write_text(
        json.dumps(records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return entry


def delete_history_entry(history_id, history_path=HISTORY_PATH):
    records = load_history(history_path)
    remaining_records = [item for item in records if item.get("history_id") != history_id]
    if len(remaining_records) == len(records):
        raise ValueError("History item not found.")

    Path(history_path).write_text(
        json.dumps(remaining_records, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return True


def get_history_entry(history_id, history_path=HISTORY_PATH):
    for item in load_history(history_path):
        if item.get("history_id") == history_id:
            return item
    raise ValueError("History item not found.")


def clear_history(history_path=HISTORY_PATH):
    Path(history_path).write_text(
        json.dumps([], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return True
