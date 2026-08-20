"""Search client abstraction for ResearcherAgent.

Implements offline corpus search over the ai_agent_offline_research_corpus_v2 directory.
Optionally enriches results with Tavily if TAVILY_API_KEY is set.
"""

from __future__ import annotations

import json
import logging
import math
import re
from pathlib import Path

from multi_agent_research_lab.core.schemas import SourceDocument

logger = logging.getLogger(__name__)

def _get_corpus_dir() -> Path:
    """Locate the offline corpus directory from current working dir or package location."""
    candidates = [
        Path.cwd() / "ai_agent_offline_research_corpus_v2" / "topics",
        Path(__file__).parents[3] / "ai_agent_offline_research_corpus_v2" / "topics",
        Path(__file__).parents[2] / "ai_agent_offline_research_corpus_v2" / "topics",
    ]
    for c in candidates:
        if c.exists():
            return c
    return candidates[0]


def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer, lowercase."""
    return re.findall(r"[a-z0-9]+", text.lower())


def _tf_idf_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    """Compute a simple term-frequency overlap score."""
    if not doc_tokens:
        return 0.0
    doc_set = set(doc_tokens)
    matches = sum(1 for t in query_tokens if t in doc_set)
    return matches / math.sqrt(len(doc_tokens))


def _pick_corpus_file(query: str) -> Path | None:
    """Choose the most relevant corpus JSON file for the query."""
    corpus_dir = _get_corpus_dir()
    if not corpus_dir.exists():
        logger.warning("Corpus directory not found: %s", corpus_dir)
        return None

    files = sorted(corpus_dir.glob("*.json"))
    if not files:
        return None


    query_tokens = _tokenize(query)
    best_file: Path | None = None
    best_score = -1.0

    for fp in files:
        # Score against filename + title (cheap heuristic before full JSON load)
        stem_tokens = _tokenize(fp.stem)
        score = _tf_idf_score(query_tokens, stem_tokens)
        if score > best_score:
            best_score = score
            best_file = fp

    logger.info("Selected corpus file: %s (score=%.3f)", best_file, best_score)
    return best_file


def _load_corpus(filepath: Path) -> dict:
    with filepath.open(encoding="utf-8") as f:
        return json.load(f)


def _extract_sources(corpus: dict, query: str, max_results: int) -> list[SourceDocument]:
    """Extract SourceDocuments from corpus, ranked by relevance."""
    kb = corpus.get("knowledge_base", {})
    query_tokens = _tokenize(query)
    candidates: list[tuple[float, SourceDocument]] = []

    # --- knowledge_articles ---
    for art in kb.get("knowledge_articles", []):
        text = f"{art.get('title', '')} {art.get('content', '')}"
        score = _tf_idf_score(query_tokens, _tokenize(text))
        snippet = art.get("content", "")[:600]
        doc = SourceDocument(
            title=art.get("title", "Unknown"),
            url=None,
            snippet=snippet,
            metadata={
                "source_id": art.get("article_id", ""),
                "type": "knowledge_article",
                "relevance_score": score,
            },
        )
        candidates.append((score, doc))

    # --- source_documents ---
    for src in kb.get("source_documents", []):
        text = f"{src.get('title', '')} {src.get('full_text', '')} {' '.join(src.get('key_takeaways', []))}"
        score = _tf_idf_score(query_tokens, _tokenize(text))
        snippet = src.get("full_text", "")[:600] or "; ".join(src.get("key_takeaways", []))[:600]
        doc = SourceDocument(
            title=src.get("title", "Unknown"),
            url=src.get("provenance_url"),
            snippet=snippet,
            metadata={
                "source_id": src.get("document_id", ""),
                "type": "source_document",
                "is_synthetic": src.get("is_synthetic", False),
                "citation_label": src.get("citation_label", ""),
                "year": src.get("year"),
                "relevance_score": score,
            },
        )
        candidates.append((score, doc))

    # --- fact_bank (top facts as mini sources) ---
    facts = kb.get("fact_bank", [])
    for fact in facts:
        stmt = fact.get("statement", "")
        score = _tf_idf_score(query_tokens, _tokenize(stmt))
        if score > 0:
            doc = SourceDocument(
                title=f"Fact: {stmt[:80]}",
                url=None,
                snippet=stmt,
                metadata={
                    "source_id": fact.get("fact_id", ""),
                    "type": "fact",
                    "confidence": fact.get("confidence"),
                    "evidence_source_ids": fact.get("evidence_source_ids", []),
                    "relevance_score": score,
                },
            )
            candidates.append((score, doc))

    # Sort by descending score, deduplicate, return top N
    candidates.sort(key=lambda x: x[0], reverse=True)
    seen_ids: set[str] = set()
    results: list[SourceDocument] = []
    for score, doc in candidates:
        sid = doc.metadata.get("source_id", "")
        if sid not in seen_ids:
            seen_ids.add(sid)
            results.append(doc)
        if len(results) >= max_results:
            break

    return results


class SearchClient:
    """Offline corpus search client.

    Primary: searches the ai_agent_offline_research_corpus_v2 JSON files
    using TF-IDF keyword matching across articles, source_documents, and facts.

    Optional: if TAVILY_API_KEY is set, enriches with live web results.
    """

    def __init__(self) -> None:
        from multi_agent_research_lab.core.config import get_settings

        self._settings = get_settings()

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Return SourceDocuments relevant to query from the offline corpus."""
        logger.info("SearchClient.search query='%s' max=%d", query[:80], max_results)

        corpus_file = _pick_corpus_file(query)
        results: list[SourceDocument] = []

        if corpus_file:
            try:
                corpus = _load_corpus(corpus_file)
                results = _extract_sources(corpus, query, max_results)
                logger.info("Corpus search returned %d documents", len(results))
            except Exception as exc:
                logger.warning("Corpus search failed: %s", exc)

        # Optional Tavily enrichment
        if self._settings.tavily_api_key and len(results) < max_results:
            try:
                results = self._enrich_with_tavily(query, results, max_results)
            except Exception as exc:
                logger.warning("Tavily enrichment failed (non-fatal): %s", exc)

        if not results:
            logger.warning("Search returned no results for query: %s", query[:80])

        return results[:max_results]

    def _enrich_with_tavily(
        self,
        query: str,
        existing: list[SourceDocument],
        max_results: int,
    ) -> list[SourceDocument]:
        """Append Tavily web results to existing corpus results."""
        from tavily import TavilyClient  # type: ignore[import]

        client = TavilyClient(api_key=self._settings.tavily_api_key)
        needed = max_results - len(existing)
        response = client.search(query, max_results=needed)

        for item in response.get("results", []):
            existing.append(
                SourceDocument(
                    title=item.get("title", "Web Result"),
                    url=item.get("url"),
                    snippet=item.get("content", "")[:600],
                    metadata={"source_id": f"tavily_{item.get('url', '')}", "type": "web"},
                )
            )
        return existing

