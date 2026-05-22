"""Web / internet ingestion architecture.

LLMs effectively have a frozen snapshot of the web from their training
cutoff. RCK can be LIVE: when configured with a fetcher provider, it
ingests web pages on demand, extracts triples via Open IE, and stores
them with URL + timestamp provenance.

Without network access (default), the module returns informative stubs
so the architecture is testable. Real providers plug in.

Providers:
  - WebFetcher: URL -> text content
  - SearchProvider: query -> list of URLs
  - RssFeed: URL -> stream of articles

Operations:
  ingest_url(kb, url)              -- fetch + extract + store
  ingest_search(kb, query, k=5)    -- search + ingest top-k
  refresh_known(kb)                -- re-fetch URLs we already ingested
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from typing import Protocol

from rck.knowledge_base import ShardedKnowledgeBase
from rck.open_ie import extract_triples_from_text
from rck.provenance import ProvenanceStore


# ---------------------------------------------------------------------------
#  Provider protocols
# ---------------------------------------------------------------------------

class WebFetcher(Protocol):
    def fetch(self, url: str) -> dict:
        """Return {'url', 'status', 'text', 'fetched_at'} or {'error': str}."""
        ...


class SearchProvider(Protocol):
    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        """Return [{'url', 'title', 'snippet'}, ...]."""
        ...


# ---------------------------------------------------------------------------
#  Stub providers (no network)
# ---------------------------------------------------------------------------

@dataclass
class StubWebFetcher:
    """Returns a clear 'not configured' message instead of fetching."""
    name: str = "stub-web-fetcher"

    def fetch(self, url: str) -> dict:
        return {
            "url": url, "status": "stub",
            "text": "",
            "error": (f"[stub] no real web fetcher configured. Install "
                      f"a real one (e.g. via httpx + readability-lxml) and "
                      f"call WebIngest.set_fetcher(...). URL would be: {url}"),
            "fetched_at": time.time(),
        }


@dataclass
class StubSearchProvider:
    name: str = "stub-search"

    def search(self, query: str, *, max_results: int = 5) -> list[dict]:
        return [{
            "url": "stub://no-search-configured",
            "title": "[stub]",
            "snippet": f"No real search provider; query was: {query!r}",
        }]


# ---------------------------------------------------------------------------
#  Simple HTML -> text helper for when a fetcher does provide HTML.
# ---------------------------------------------------------------------------

_HTML_TAG = re.compile(r"<[^>]+>")
_HTML_ENTITY = re.compile(r"&(?:[a-zA-Z]+|#\d+);")
_MULTI_WS = re.compile(r"\s+")


def html_to_text(html: str) -> str:
    """Naive HTML stripper. Good enough for plain pages; production
    setups should use a real readability extractor."""
    text = _HTML_TAG.sub(" ", html)
    text = _HTML_ENTITY.sub(" ", text)
    text = _MULTI_WS.sub(" ", text)
    return text.strip()


# ---------------------------------------------------------------------------
#  WebIngest -- the user-facing API
# ---------------------------------------------------------------------------

@dataclass
class WebIngest:
    """High-level web ingestion pipeline."""

    fetcher: WebFetcher = field(default_factory=StubWebFetcher)
    searcher: SearchProvider = field(default_factory=StubSearchProvider)
    known_urls: dict[str, float] = field(default_factory=dict)

    def set_fetcher(self, fetcher: WebFetcher) -> None:
        self.fetcher = fetcher

    def set_searcher(self, searcher: SearchProvider) -> None:
        self.searcher = searcher

    def ingest_url(
        self, kb: ShardedKnowledgeBase, url: str,
        provenance: ProvenanceStore | None = None,
    ) -> dict:
        """Fetch, html->text, extract triples, store with URL provenance."""
        result = self.fetcher.fetch(url)
        if "error" in result and not result.get("text"):
            return {"url": url, "ok": False, "error": result["error"],
                    "triples": 0}
        text = result.get("text", "")
        if not text:
            return {"url": url, "ok": False, "triples": 0,
                    "error": "no text returned"}
        # If it looks like HTML, strip it.
        if "<" in text[:1000] and ">" in text[:1000]:
            text = html_to_text(text)
        triples = extract_triples_from_text(text)
        for s, r, o in triples:
            kb.store({"S": s, "R": r, "O": o})
            if provenance is not None:
                provenance.store(s, r, o, source=url, tags={"web"})
        self.known_urls[url] = time.time()
        return {"url": url, "ok": True, "triples": len(triples),
                "text_length": len(text)}

    def ingest_search(
        self, kb: ShardedKnowledgeBase, query: str,
        *, max_results: int = 5,
        provenance: ProvenanceStore | None = None,
    ) -> dict:
        """Search, then ingest each result."""
        results = self.searcher.search(query, max_results=max_results)
        per_result = []
        for r in results:
            url = r.get("url")
            if not url or url.startswith("stub://"):
                per_result.append({"url": url, "ok": False, "triples": 0,
                                   "error": "stub search provider"})
                continue
            per_result.append(self.ingest_url(kb, url, provenance=provenance))
        total = sum(r.get("triples", 0) for r in per_result)
        return {"query": query, "results": per_result,
                "total_triples": total}

    def refresh_known(
        self, kb: ShardedKnowledgeBase,
        provenance: ProvenanceStore | None = None,
        *, max_urls: int = 10,
    ) -> dict:
        """Re-fetch the URLs we already ingested, in case content changed."""
        n_ok = 0; new_triples = 0
        for url in list(self.known_urls)[:max_urls]:
            res = self.ingest_url(kb, url, provenance=provenance)
            if res["ok"]:
                n_ok += 1; new_triples += res["triples"]
        return {"refreshed": n_ok, "new_triples": new_triples}
