import json
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from core.config import NOTES_PATH


def load_notes(notes_path=NOTES_PATH):
    path = Path(notes_path)
    if not path.exists():
        return []

    content = json.loads(path.read_text(encoding="utf-8"))
    notes = content if isinstance(content, list) else []
    return sorted(notes, key=lambda item: item.get("saved_at", ""), reverse=True)


def save_note(query, answer, citations=None, notes_path=NOTES_PATH):
    notes = load_notes(notes_path)
    note = {
        "note_id": uuid4().hex[:12],
        "query": query,
        "answer": answer,
        "citations": citations or [],
        "saved_at": datetime.now().isoformat(timespec="seconds"),
    }
    notes.insert(0, note)
    Path(notes_path).write_text(
        json.dumps(notes, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return note


def delete_note(note_id, notes_path=NOTES_PATH):
    notes = load_notes(notes_path)
    remaining_notes = [note for note in notes if note.get("note_id") != note_id]
    if len(remaining_notes) == len(notes):
        raise ValueError("Note not found.")

    Path(notes_path).write_text(
        json.dumps(remaining_notes, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return True


def clear_notes(notes_path=NOTES_PATH):
    Path(notes_path).write_text(
        json.dumps([], indent=2, ensure_ascii=True),
        encoding="utf-8",
    )
    return True
