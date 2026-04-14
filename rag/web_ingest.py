import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


def fetch_url_content(url: str, timeout: int = 15) -> dict:
    """
    Fetch a single URL and return extracted content plus metadata.
    """

    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Please enter a valid http or https URL.")

    response = requests.get(
        url,
        timeout=timeout,
        headers={
            "User-Agent": (
                "Mozilla/5.0 (compatible; Rabbook/1.0; +https://example.local/rabbook)"
            )
        },
    )
    response.raise_for_status()

    page_text, title = extract_page_text(response.text)
    if len(page_text) < 200:
        raise ValueError("Could not extract enough readable text from this URL.")

    title_slug = slugify_text(title or parsed_url.netloc)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]
    file_name = f"url-{title_slug}-{url_hash}.txt"

    return {
        "page_text": page_text,
        "source_url": url,
        "title": title,
        "domain": parsed_url.netloc,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_name": file_name,
    }


def save_url_import(payload: dict, target_dir: Path) -> Path:
    """
    Persist the extracted URL content as a text file so it can be re-ingested later.
    """

    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / payload["file_name"]
    file_path.write_text(build_url_import_text(payload), encoding="utf-8")
    return file_path


def build_url_import_text(payload: dict) -> str:
    lines = [
        f"Title: {payload.get('title') or 'Untitled page'}",
        f"Source URL: {payload.get('source_url', '')}",
        f"Domain: {payload.get('domain', '')}",
        f"Fetched At: {payload.get('fetched_at', '')}",
        "",
        payload.get("page_text", ""),
    ]
    return "\n".join(lines).strip() + "\n"


def extract_page_text(html: str) -> tuple[str, str]:
    """
    Pull readable text from a simple HTML page without full article extraction logic.
    """

    soup = BeautifulSoup(html, "html.parser")

    for tag_name in ("script", "style", "noscript", "svg", "footer", "nav"):
        for tag in soup.find_all(tag_name):
            tag.decompose()

    title = soup.title.get_text(" ", strip=True) if soup.title else "Untitled page"
    content_root = soup.find("article") or soup.find("main") or soup.body or soup

    parts = []
    for tag in content_root.find_all(["h1", "h2", "h3", "p", "li"]):
        text = clean_text(tag.get_text(" ", strip=True))
        if not text:
            continue
        parts.append(text)

    if not parts:
        fallback_text = clean_text(content_root.get_text("\n", strip=True))
        if fallback_text:
            parts.append(fallback_text)

    full_text = "\n\n".join(parts)
    return full_text, title


def clean_text(text: str) -> str:
    return " ".join(text.split())


def slugify_text(text: str, max_length: int = 50) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact[:max_length] or "imported-page"
