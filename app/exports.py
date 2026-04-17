from fastapi.responses import Response

from rag.exporters import (
    export_answer_as_markdown,
    export_history_as_markdown,
    export_notes_as_markdown,
    export_records_as_json,
)


def build_download_response(content, filename, media_type):
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )


def export_notes_markdown_response(saved_notes):
    return build_download_response(
        export_notes_as_markdown(saved_notes),
        "rabbook_notes.md",
        "text/markdown; charset=utf-8",
    )


def export_notes_json_response(saved_notes):
    return build_download_response(
        export_records_as_json(saved_notes),
        "rabbook_notes.json",
        "application/json; charset=utf-8",
    )


def export_history_markdown_response(history_items):
    return build_download_response(
        export_history_as_markdown(history_items),
        "rabbook_history.md",
        "text/markdown; charset=utf-8",
    )


def export_history_json_response(history_items):
    return build_download_response(
        export_records_as_json(history_items),
        "rabbook_history.json",
        "application/json; charset=utf-8",
    )


def export_answer_markdown_response(query, answer, citations):
    return build_download_response(
        export_answer_as_markdown(
            query=query.strip(),
            answer=answer.strip(),
            citations=citations,
        ),
        "rabbook_answer.md",
        "text/markdown; charset=utf-8",
    )


def export_answer_json_response(query, answer, citations):
    payload = {
        "query": query.strip(),
        "answer": answer.strip(),
        "citations": citations,
    }
    return build_download_response(
        export_records_as_json(payload),
        "rabbook_answer.json",
        "application/json; charset=utf-8",
    )
