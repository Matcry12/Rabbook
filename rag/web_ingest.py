import asyncio
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import trafilatura
from crawl4ai import AsyncWebCrawler, CrawlerRunConfig
from ddgs import DDGS


def web_search(query: str, max_results: int = 5) -> list[dict]:
    results = []
    with DDGS() as ddgs:
        ddgs_results = ddgs.text(query, max_results=max_results)
        for r in ddgs_results:
            results.append({
                "url": r["href"],
                "title": r["title"],
                "snippet": r["body"]
            })
    return results


def fetch_url_content(url: str, timeout: int = 15) -> dict:
    parsed_url = urlparse(url)
    if parsed_url.scheme not in {"http", "https"} or not parsed_url.netloc:
        raise ValueError("Please enter a valid http or https URL.")

    async def _crawl():
        config = CrawlerRunConfig(page_timeout=timeout * 1000)
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun(url=url, config=config)

    result = asyncio.run(_crawl())

    if not result.success:
        raise ValueError(f"Failed to fetch URL: {result.error_message}")

    page_text = trafilatura.extract(result.html) or ""
    if len(page_text) < 200:
        raise ValueError("Could not extract enough readable text from this URL.")

    metadata = trafilatura.extract_metadata(result.html)
    title = (metadata.title if metadata else None) or parsed_url.netloc

    title_slug = slugify_text(title)
    url_hash = hashlib.sha256(url.encode("utf-8")).hexdigest()[:8]

    return {
        "page_text": page_text,
        "source_url": url,
        "title": title,
        "domain": parsed_url.netloc,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_name": f"url-{title_slug}-{url_hash}.txt",
    }


def fetch_urls_parallel(urls: list[str], timeout: int = 15) -> list[dict]:
    valid_urls = [
        u for u in urls
        if urlparse(u).scheme in {"http", "https"} and urlparse(u).netloc
    ]

    async def _crawl_many():
        config = CrawlerRunConfig(page_timeout=timeout * 1000)
        async with AsyncWebCrawler() as crawler:
            return await crawler.arun_many(urls=valid_urls, config=config)

    results = asyncio.run(_crawl_many())

    payloads = []
    for result in results:
        if not result.success:
            print(f"[web_ingest] Fetch FAILED for {result.url}: {result.error_message}")
            continue
        page_text = trafilatura.extract(result.html) or ""
        if len(page_text) < 200:
            continue
        parsed_url = urlparse(result.url)
        metadata = trafilatura.extract_metadata(result.html)
        title = (metadata.title if metadata else None) or parsed_url.netloc
        url_hash = hashlib.sha256(result.url.encode("utf-8")).hexdigest()[:8]
        payloads.append({
            "page_text": page_text,
            "source_url": result.url,
            "title": title,
            "domain": parsed_url.netloc,
            "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "file_name": f"url-{slugify_text(title)}-{url_hash}.txt",
        })
    return payloads


def save_url_import(payload: dict, target_dir: Path) -> Path:
    target_dir.mkdir(parents=True, exist_ok=True)
    file_path = target_dir / payload["file_name"]
    file_path.write_text(build_url_import_text(payload), encoding="utf-8")
    return file_path


def build_research_import_payload(result: dict) -> dict:
    source_url = result.get("url", "")
    title = result.get("title") or "Web Source"
    domain = urlparse(source_url).netloc or "unknown"
    page_text = result.get("content") or result.get("snippet") or ""
    url_hash = hashlib.sha256(source_url.encode("utf-8")).hexdigest()[:8]
    title_slug = slugify_text(title or domain)

    return {
        "page_text": page_text,
        "source_url": source_url,
        "title": title,
        "domain": domain,
        "fetched_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "file_name": f"research-{title_slug}-{url_hash}.txt",
    }


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


def slugify_text(text: str, max_length: int = 50) -> str:
    cleaned = "".join(char.lower() if char.isalnum() else "-" for char in text)
    compact = "-".join(part for part in cleaned.split("-") if part)
    return compact[:max_length] or "imported-page"
