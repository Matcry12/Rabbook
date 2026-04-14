import json


def export_records_as_json(records):
    return json.dumps(records, indent=2, ensure_ascii=False)


def export_notes_as_markdown(notes):
    lines = ["# Rabbook Notes", ""]

    for note in notes:
        lines.extend(
            [
                f"## {note.get('query', 'Untitled note')}",
                "",
                f"- Saved: {note.get('saved_at', 'unknown')}",
                "",
                note.get("answer", ""),
                "",
            ]
        )
        lines.extend(_build_citation_lines(note.get("citations", [])))

    return "\n".join(lines).strip() + "\n"


def export_history_as_markdown(history_items):
    lines = ["# Rabbook Chat History", ""]

    for item in history_items:
        lines.extend(
            [
                f"## {item.get('query', 'Untitled question')}",
                "",
                f"- Asked: {item.get('created_at', 'unknown')}",
                f"- File Filter: {item.get('selected_file') or 'all'}",
                f"- File Type Filter: {item.get('selected_file_type') or 'all'}",
                f"- Page Range: {_build_page_range_text(item)}",
                "",
                item.get("answer", ""),
                "",
            ]
        )
        lines.extend(_build_citation_lines(item.get("citations", [])))

    return "\n".join(lines).strip() + "\n"


def export_answer_as_markdown(query, answer, citations=None):
    lines = [
        "# Rabbook Answer",
        "",
        f"## Question",
        "",
        query,
        "",
        "## Answer",
        "",
        answer,
        "",
    ]
    lines.extend(_build_citation_lines(citations or []))
    return "\n".join(lines).strip() + "\n"


def _build_citation_lines(citations):
    if not citations:
        return []

    lines = ["### Citations", ""]
    for item in citations:
        source = item.get("source", "Unknown source")
        page = item.get("page")
        page_text = f", page {page}" if page is not None else ""
        lines.append(f"- [{item.get('number', '?')}] {source}{page_text}")
    lines.append("")
    return lines


def _build_page_range_text(item):
    start = item.get("page_start") or "start"
    end = item.get("page_end") or "end"
    return f"{start}-{end}"
