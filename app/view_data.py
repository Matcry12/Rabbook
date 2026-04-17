from rag.history import load_history
from rag.notes import load_notes
from rag.registry import list_documents


def get_available_files(chunk_registry):
    records = chunk_registry.get("by_chunk_id", {})
    file_names = {
        record.get("metadata", {}).get("file_name", "")
        for record in records.values()
        if record.get("metadata", {}).get("file_name")
    }
    return sorted(file_names)


def get_available_file_types(chunk_registry):
    records = chunk_registry.get("by_chunk_id", {})
    file_types = {
        record.get("metadata", {}).get("file_type", "")
        for record in records.values()
        if record.get("metadata", {}).get("file_type")
    }
    return sorted(file_types)


def get_library_documents(registry_path):
    return list_documents(str(registry_path))


def get_saved_notes():
    return load_notes()


def get_history_items():
    return load_history()
