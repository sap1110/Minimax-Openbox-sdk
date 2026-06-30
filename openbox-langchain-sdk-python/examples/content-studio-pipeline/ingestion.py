"""
Ingestion — RSS/API feed fetching and deduplication.

Mirrors the n8n nodes:
  RSS — Hacker News / TLDR AI / VentureBeat AI / Dark Reading / Wired Security / The Hacker News
  API — HN Algolia / DEV.to / arXiv
  Merge RSS → Dedup and Bundle
"""

from __future__ import annotations

import logging
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import feedparser
import httpx

_logger = logging.getLogger("content_studio.ingestion")

RSS_FEEDS = [
    (
        "Hacker News",
        "https://hnrss.org/newest?q=AI+governance+OR+%22agentic+AI%22+OR+%22LLM+security%22"
        "+OR+%22AI+compliance%22+OR+%22AI+trust%22+OR+%22model+risk%22"
        "+OR+%22responsible+AI%22+OR+%22EU+AI+Act%22&count=20",
    ),
    ("TLDR AI", "https://tldr.tech/ai/rss"),
    ("VentureBeat", "https://venturebeat.com/category/ai/feed/"),
    ("Dark Reading", "https://www.darkreading.com/rss.xml"),
    ("Wired", "https://www.wired.com/feed/category/security/latest/rss"),
    ("The Hacker News", "https://feeds.feedburner.com/TheHackersNews"),
]

TIMEOUT = 15.0


def _source_from_url(url: str) -> str:
    """Map URL domain → human-readable source name (mirrors n8n JS function)."""
    u = (url or "").lower()
    if "venturebeat" in u:
        return "VentureBeat"
    if "ycombinator" in u:
        return "Hacker News"
    if "tldr.tech" in u:
        return "TLDR AI"
    if "theregister" in u:
        return "The Register"
    if "darkreading" in u:
        return "Dark Reading"
    if "technologyreview" in u:
        return "MIT Tech Review"
    if "wired.com" in u:
        return "Wired"
    if "thehackernews" in u:
        return "The Hacker News"
    if "securityweek" in u:
        return "SecurityWeek"
    if "helpnetsecurity" in u:
        return "Help Net Security"
    if "arxiv" in u:
        return "arXiv"
    return "RSS"


def _fetch_rss(source_name: str, url: str) -> list[dict[str, Any]]:
    """Fetch and parse a single RSS feed. Returns list of normalized items."""
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries:
            link = getattr(entry, "link", "") or ""
            title = getattr(entry, "title", "") or ""
            published = getattr(entry, "published", "") or getattr(entry, "updated", "") or ""
            if title:
                items.append({
                    "title": title,
                    "url": link,
                    "published": published,
                    "source": source_name,
                })
        _logger.info("RSS %s: %d items", source_name, len(items))
        return items
    except Exception as exc:
        _logger.warning("RSS %s failed: %s", source_name, exc)
        return []


def _fetch_hn_algolia() -> list[dict[str, Any]]:
    """Fetch HN Algolia search results for AI governance topics."""
    since = int(time.time()) - 86400
    url = (
        "https://hn.algolia.com/api/v1/search"
        "?query=AI+governance+OR+%22agentic+AI%22+OR+%22LLM+security%22"
        "+OR+%22AI+compliance%22+OR+%22model+risk%22"
        f"&tags=story&numericFilters=created_at_i%3E%3D{since}&hitsPerPage=20"
    )
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(url)
            resp.raise_for_status()
            hits = resp.json().get("hits", [])
            items = []
            for h in hits:
                title = h.get("title", "") or ""
                link = h.get("url", "") or f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
                created = h.get("created_at", "") or ""
                if title:
                    items.append({
                        "title": title,
                        "url": link,
                        "published": created,
                        "source": "Hacker News",
                    })
            _logger.info("HN Algolia: %d items", len(items))
            return items
    except Exception as exc:
        _logger.warning("HN Algolia failed: %s", exc)
        return []


def _fetch_devto() -> list[dict[str, Any]]:
    """Fetch DEV.to articles for AI governance tags."""
    url = "https://dev.to/api/articles?tags=aigovernance,llmsecurity,responsibleai,aiethics&per_page=20&top=1"
    try:
        with httpx.Client(timeout=TIMEOUT) as client:
            resp = client.get(url)
            resp.raise_for_status()
            articles = resp.json()
            items = []
            for a in articles:
                title = a.get("title", "") or ""
                link = a.get("url", "") or ""
                published = a.get("published_at", "") or ""
                if title:
                    items.append({
                        "title": title,
                        "url": link,
                        "published": published,
                        "source": "DEV.to",
                    })
            _logger.info("DEV.to: %d items", len(items))
            return items
    except Exception as exc:
        _logger.warning("DEV.to failed: %s", exc)
        return []


def _fetch_arxiv() -> list[dict[str, Any]]:
    """Fetch recent arXiv papers on AI governance topics."""
    url = (
        "https://export.arxiv.org/api/query"
        "?search_query=ti:%22AI+governance%22+OR+ti:%22LLM+security%22"
        "+OR+ti:%22agentic+AI%22+OR+ti:%22responsible+AI%22"
        "&sortBy=submittedDate&sortOrder=descending&max_results=8"
    )
    try:
        feed = feedparser.parse(url)
        items = []
        for entry in feed.entries:
            title = getattr(entry, "title", "").replace("\n", " ").strip()
            link = getattr(entry, "link", "") or ""
            published = getattr(entry, "published", "") or ""
            if title:
                items.append({
                    "title": title,
                    "url": link,
                    "published": published,
                    "source": "arXiv",
                })
        _logger.info("arXiv: %d items", len(items))
        return items
    except Exception as exc:
        _logger.warning("arXiv failed: %s", exc)
        return []


def _dedup(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Deduplicate by normalized title key (mirrors n8n Dedup and Bundle node)."""
    seen: set[str] = set()
    result: list[dict[str, Any]] = []
    for item in items:
        title = item.get("title", "")
        if not title:
            continue
        key = "".join(c for c in title.lower() if c.isalnum())[:60]
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def fetch_all_feeds() -> list[dict[str, Any]]:
    """Fetch all RSS/API sources, merge, and deduplicate.

    Returns a flat list of items: {title, url, published, source}
    """
    all_items: list[dict[str, Any]] = []

    for source_name, url in RSS_FEEDS:
        all_items.extend(_fetch_rss(source_name, url))

    all_items.extend(_fetch_hn_algolia())
    all_items.extend(_fetch_devto())
    all_items.extend(_fetch_arxiv())

    deduped = _dedup(all_items)
    _logger.info("Total after dedup: %d items", len(deduped))
    return deduped
