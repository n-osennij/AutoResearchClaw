"""Unified literature search with deduplication.

Combines results from the academic backends in ``_SOURCES`` — OpenAlex,
Semantic Scholar, arXiv, OpenReview, Crossref, PubMed and CORE —
deduplicates by DOI → arXiv ID → fuzzy title match, and returns a merged
list sorted by citation count (descending).

Source priority: OpenAlex (most generous limits) → Semantic Scholar → arXiv;
the remaining four are additive and opt-in via ``literature_search.sources``.
If any source hits rate limits, remaining sources compensate automatically.

Public API
----------
- ``search_papers(query, *, limit, sources, year_min, deduplicate, keys)``
  → ``list[Paper]``
- ``search_papers_multi_query(queries, *, keys, inter_query_delay, ...)``
  → ``list[Paper]``
- ``SourceKeys`` — every backend credential in one object
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
import importlib
import logging
import re
import time
import urllib.error
from typing import cast

from researchclaw.literature.arxiv_client import search_arxiv
from researchclaw.literature.core_client import search_core
from researchclaw.literature.crossref_client import search_crossref
from researchclaw.literature.models import Author, Paper
from researchclaw.literature.openalex_client import search_openalex
from researchclaw.literature.openreview_client import search_openreview
from researchclaw.literature.pubmed_client import search_pubmed
from researchclaw.literature.semantic_scholar import search_semantic_scholar

logger = logging.getLogger(__name__)

# OpenAlex first (10K/day), then S2 (1K/5min), then arXiv (1/3s) — least
# pressure on the most restrictive API.  OpenReview / Crossref / PubMed / CORE
# are additive sources appended by the caller when the domain needs them.
_DEFAULT_SOURCES = ("openalex", "semantic_scholar", "arxiv")


@dataclass(frozen=True)
class SourceKeys:
    """Credentials for the literature backends.

    Bundled into one object so ``search_papers`` does not grow a keyword
    argument per new source. Empty values mean "not configured" — each
    adapter then falls back to that source's unauthenticated behaviour
    (OpenAlex and arXiv work without a key, CORE does not).
    """

    s2_api_key: str = ""
    openalex_email: str = ""
    openalex_api_key: str = ""
    pubmed_api_key: str = ""
    core_api_key: str = ""


_Adapter = Callable[[str, int, int, SourceKeys], list[Paper]]


def _openalex_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    kwargs: dict[str, object] = {"limit": limit, "year_min": year_min}
    if keys.openalex_email:
        kwargs["email"] = keys.openalex_email
    if keys.openalex_api_key:
        kwargs["api_key"] = keys.openalex_api_key
    return search_openalex(query, **kwargs)


def _semantic_scholar_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_semantic_scholar(
        query, limit=limit, year_min=year_min, api_key=keys.s2_api_key
    )


def _arxiv_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_arxiv(query, limit=limit, year_min=year_min)


def _openreview_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_openreview(query, limit=limit, year_min=year_min)


def _crossref_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_crossref(
        query, limit=limit, year_min=year_min, email=keys.openalex_email
    )


def _pubmed_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_pubmed(
        query,
        limit=limit,
        year_min=year_min,
        api_key=keys.pubmed_api_key,
        email=keys.openalex_email,
    )


def _core_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_core(
        query, limit=limit, year_min=year_min, api_key=keys.core_api_key
    )


@dataclass(frozen=True)
class _SourceSpec:
    """How one backend is called, logged and paced."""

    name: str  # canonical name, also the cache key
    label: str  # display name for log lines
    adapter: _Adapter
    pause_sec: float  # gap after this source, before the next one starts


# Pause between sources, not between queries — that one is
# ``inter_query_delay`` in ``search_papers_multi_query``. Values follow each
# API's documented limit: Semantic Scholar and OpenReview accept ~1 req/s,
# the rest tolerate faster calls.
_S2_SPEC = _SourceSpec(
    "semantic_scholar", "Semantic Scholar", _semantic_scholar_adapter, 1.0
)

_SOURCES: dict[str, _SourceSpec] = {
    "openalex": _SourceSpec("openalex", "OpenAlex", _openalex_adapter, 0.5),
    "semantic_scholar": _S2_SPEC,
    # ``s2`` is the shorthand used in older configs.
    "s2": _S2_SPEC,
    "arxiv": _SourceSpec("arxiv", "arXiv", _arxiv_adapter, 0.0),
    "openreview": _SourceSpec(
        "openreview", "OpenReview", _openreview_adapter, 1.0
    ),
    "crossref": _SourceSpec("crossref", "Crossref", _crossref_adapter, 0.5),
    "pubmed": _SourceSpec("pubmed", "PubMed", _pubmed_adapter, 0.5),
    "core": _SourceSpec("core", "CORE", _core_adapter, 0.5),
}


def _normalise_source(name: str) -> str:
    """``semantic-scholar`` / ``Semantic Scholar`` → ``semantic_scholar``."""
    return name.lower().replace("-", "_").replace(" ", "_")


CacheGet = Callable[[str, str, int, int], list[dict[str, object]] | None]
CachePut = Callable[[str, str, int, list[dict[str, object]], int], None]


def _cache_api() -> tuple[CacheGet, CachePut]:
    cache_mod = importlib.import_module("researchclaw.literature.cache")
    return cast(CacheGet, cache_mod.get_cached), cast(CachePut, cache_mod.put_cache)


def _papers_to_dicts(papers: list[Paper]) -> list[dict[str, object]]:
    """Convert papers to serializable dicts for caching."""
    return [asdict(p) for p in papers]


def _as_int(value: object, default: int = 0) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def _dicts_to_papers(dicts: list[dict[str, object]]) -> list[Paper]:
    """Reconstruct Paper objects from cached dicts."""
    papers: list[Paper] = []
    for d in dicts:
        try:
            authors_raw = d.get("authors", ())
            if not isinstance(authors_raw, list):
                authors_raw = []
            authors = tuple(
                Author(
                    name=str(cast(dict[str, object], a).get("name", "")),
                    affiliation=str(cast(dict[str, object], a).get("affiliation", "")),
                )
                for a in authors_raw
                if isinstance(a, dict)
            )
            paper_id = cast(str, d["paper_id"])
            title = cast(str, d["title"])
            papers.append(
                Paper(
                    paper_id=paper_id,
                    title=title,
                    authors=authors,
                    year=_as_int(d.get("year", 0), 0),
                    abstract=str(d.get("abstract", "")),
                    venue=str(d.get("venue", "")),
                    citation_count=_as_int(d.get("citation_count", 0), 0),
                    doi=str(d.get("doi", "")),
                    arxiv_id=str(d.get("arxiv_id", "")),
                    url=str(d.get("url", "")),
                    pdf_url=str(d.get("pdf_url", "")),
                    source=str(d.get("source", "")),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    return papers


def search_papers(
    query: str,
    *,
    limit: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    deduplicate: bool = True,
    keys: SourceKeys | None = None,
) -> list[Paper]:
    """Search multiple academic sources and return deduplicated results.

    Parameters
    ----------
    query:
        Free-text search query.
    limit:
        Maximum results *per source*.
    sources:
        Which backends to query.  Default: OpenAlex, Semantic Scholar, arXiv.
        Unknown names are skipped with a warning; ``s2`` is accepted as an
        alias for ``semantic_scholar``.
    year_min:
        If >0, pass to backends that support year filtering.
    deduplicate:
        Whether to remove duplicates across sources.
    keys:
        Credentials for the backends (see :class:`SourceKeys`). ``None`` means
        "no keys configured", which is enough for OpenAlex and arXiv.

    Returns
    -------
    list[Paper]
        Merged results, sorted by citation_count descending.
    """
    keys = keys if keys is not None else SourceKeys()
    all_papers: list[Paper] = []
    cache_get: CacheGet
    cache_put: CachePut
    cache_get, cache_put = _cache_api()

    source_stats: dict[str, int] = {}  # track per-source counts
    cache_hits = 0

    for src in sources:
        spec = _SOURCES.get(_normalise_source(src))
        if spec is None:
            logger.warning("Unknown literature source: %s (skipped)", src)
            continue
        try:
            papers = spec.adapter(query, limit, year_min, keys)
            all_papers.extend(papers)
            cache_put(query, spec.name, limit, _papers_to_dicts(papers), year_min)
            source_stats[spec.name] = len(papers)
            logger.info(
                "%s returned %d papers for %r", spec.label, len(papers), query
            )
            if spec.pause_sec:
                time.sleep(spec.pause_sec)
        except (
            OSError,
            RuntimeError,
            TypeError,
            ValueError,
            urllib.error.HTTPError,
            urllib.error.URLError,
        ):
            logger.warning(
                "[rate-limit] Source %s failed for %r — trying cache", src, query
            )
            cached = cache_get(query, spec.name, limit, year_min)
            if cached:
                papers = _dicts_to_papers(cached)
                all_papers.extend(papers)
                cache_hits += len(papers)
                logger.info(
                    "[cache] HIT: %d papers for %s/%r",
                    len(papers),
                    spec.name,
                    query,
                )
            else:
                logger.warning(
                    "No cache available for %s/%r — skipping", src, query
                )

    # Summary log
    total = len(all_papers)
    parts = [f"{src}: {n}" for src, n in source_stats.items()]
    if cache_hits:
        parts.append(f"cache: {cache_hits}")
    logger.info(
        "[literature] Found %d papers (%s) for %r",
        total,
        ", ".join(parts) if parts else "none",
        query,
    )

    if deduplicate:
        all_papers = _deduplicate(all_papers)

    # Sort by citation count descending, then year descending
    all_papers.sort(key=lambda p: (p.citation_count, p.year), reverse=True)

    return all_papers


def search_papers_multi_query(
    queries: list[str],
    *,
    limit_per_query: int = 20,
    sources: Sequence[str] = _DEFAULT_SOURCES,
    year_min: int = 0,
    keys: SourceKeys | None = None,
    inter_query_delay: float = 1.5,
) -> list[Paper]:
    """Run multiple queries and return deduplicated union.

    Adds a delay between queries to respect rate limits; per-source pacing
    lives in ``_SOURCES``.
    """
    all_papers: list[Paper] = []

    for i, q in enumerate(queries):
        if i > 0:
            time.sleep(inter_query_delay)
        results = search_papers(
            q,
            limit=limit_per_query,
            sources=sources,
            year_min=year_min,
            keys=keys,
            deduplicate=False,  # we dedup globally below
        )
        all_papers.extend(results)
        logger.info("Query %d/%d %r → %d papers", i + 1, len(queries), q, len(results))

    deduped = _deduplicate(all_papers)
    deduped.sort(key=lambda p: (p.citation_count, p.year), reverse=True)
    return deduped


# ------------------------------------------------------------------
# Deduplication
# ------------------------------------------------------------------


def _normalise_title(title: str) -> str:
    """Lower-case, strip punctuation, collapse whitespace."""
    t = title.lower()
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates.  Priority: DOI > arXiv ID > fuzzy title.

    When a duplicate is found, the entry with higher citation_count wins
    (i.e. Semantic Scholar data is preferred over arXiv-only data).
    """
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def _update_indices(p: Paper, idx: int) -> None:
        """Register all identifiers of *p* in the lookup dicts at *idx*."""
        if p.doi:
            seen_doi[p.doi.lower().strip()] = idx
        if p.arxiv_id:
            seen_arxiv[p.arxiv_id.strip()] = idx
        norm = _normalise_title(p.title)
        if norm:
            seen_title[norm] = idx

    def _replace_at(old: Paper, new: Paper, idx: int) -> None:
        """Replace paper at *idx* and clean up stale index entries."""
        # Remove old identifiers that the new paper does NOT share
        if old.doi:
            old_doi = old.doi.lower().strip()
            new_doi = new.doi.lower().strip() if new.doi else ""
            if old_doi != new_doi and seen_doi.get(old_doi) == idx:
                del seen_doi[old_doi]
        if old.arxiv_id:
            old_ax = old.arxiv_id.strip()
            new_ax = new.arxiv_id.strip() if new.arxiv_id else ""
            if old_ax != new_ax and seen_arxiv.get(old_ax) == idx:
                del seen_arxiv[old_ax]
        old_norm = _normalise_title(old.title)
        new_norm = _normalise_title(new.title)
        if old_norm and old_norm != new_norm and seen_title.get(old_norm) == idx:
            del seen_title[old_norm]
        result[idx] = new
        _update_indices(new, idx)

    for paper in papers:
        is_dup = False

        # Check DOI
        if paper.doi:
            doi_key = paper.doi.lower().strip()
            if doi_key in seen_doi:
                idx = seen_doi[doi_key]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True

        # Check arXiv ID
        if not is_dup and paper.arxiv_id:
            ax_key = paper.arxiv_id.strip()
            if ax_key in seen_arxiv:
                idx = seen_arxiv[ax_key]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True

        # Check fuzzy title
        if not is_dup:
            norm = _normalise_title(paper.title)
            if norm and norm in seen_title:
                idx = seen_title[norm]
                if paper.citation_count > result[idx].citation_count:
                    _replace_at(result[idx], paper, idx)
                is_dup = True

        if is_dup:
            continue

        # Not a duplicate — store indices and append
        new_idx = len(result)
        _update_indices(paper, new_idx)
        result.append(paper)

    return result


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    """Generate a combined BibTeX file from a list of papers."""
    entries = [p.to_bibtex() for p in papers]
    return "\n\n".join(entries) + "\n"
