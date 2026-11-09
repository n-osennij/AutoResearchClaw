"""Unified literature search with deduplication.

Combines results from the academic backends in ``_SOURCES`` — OpenAlex,
Semantic Scholar, arXiv, OpenReview, Crossref, PubMed, Europe PMC and CORE —
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
from dataclasses import asdict, dataclass, replace
import importlib
import logging
import re
import time
import unicodedata
import urllib.error
from typing import cast

from researchclaw.literature.arxiv_client import search_arxiv
from researchclaw.literature.core_client import search_core
from researchclaw.literature.crossref_client import search_crossref
from researchclaw.literature.europepmc_client import search_europepmc
from researchclaw.literature.models import Author, Paper, normalise_doi
from researchclaw.literature.openalex_client import search_openalex
from researchclaw.literature.openreview_client import search_openreview
from researchclaw.literature.pubmed_client import search_pubmed
from researchclaw.literature.semantic_scholar import search_semantic_scholar

logger = logging.getLogger(__name__)

# OpenAlex first (10K/day), then S2 (1K/5min), then arXiv (1/3s) — least
# pressure on the most restrictive API.  OpenReview / Crossref / PubMed /
# Europe PMC / CORE are additive sources appended by the caller when the domain
# needs them.
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


def _europepmc_adapter(
    query: str, limit: int, year_min: int, keys: SourceKeys
) -> list[Paper]:
    return search_europepmc(query, limit=limit, year_min=year_min)


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
    # Europe PMC is the biomedical superset (MEDLINE + preprints + full text);
    # listed under both its own name and the shorthand people type in configs.
    "europepmc": _SourceSpec(
        "europepmc", "Europe PMC", _europepmc_adapter, 0.5
    ),
    "europe_pmc": _SourceSpec(
        "europepmc", "Europe PMC", _europepmc_adapter, 0.5
    ),
    "core": _SourceSpec("core", "CORE", _core_adapter, 0.5),
}


def normalise_source(name: str) -> str:
    """``semantic-scholar`` / ``Semantic Scholar`` → ``semantic_scholar``."""
    return name.lower().replace("-", "_").replace(" ", "_")


# Backwards-compatible alias: the pipeline and tests imported the private name.
_normalise_source = normalise_source


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


def _as_str_tuple(value: object) -> tuple[str, ...]:
    """``["A", "B"]`` from a cache entry → ``("A", "B")``; anything else → ()."""
    if not isinstance(value, list | tuple):
        return ()
    return tuple(str(item) for item in value if isinstance(item, str) and item)


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
                    pmid=str(d.get("pmid", "")),
                    is_preprint=bool(d.get("is_preprint", False)),
                    pub_types=_as_str_tuple(d.get("pub_types")),
                    mesh_terms=_as_str_tuple(d.get("mesh_terms")),
                    retracted=bool(d.get("retracted", False)),
                    _bibtex_override=str(d.get("_bibtex_override", "")),
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

    seen_specs: set[str] = set()
    for src in sources:
        spec = _SOURCES.get(_normalise_source(src))
        if spec is None:
            logger.warning("Unknown literature source: %s (skipped)", src)
            continue
        # Aliases ("s2", "europe_pmc") resolve to one spec; querying it twice
        # would burn a request and a cache write for identical results.
        if spec.name in seen_specs:
            continue
        seen_specs.add(spec.name)
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


# Greek letters carry meaning in biomedical titles: α- and β-synuclein are
# different proteins, and Na+/K+-ATPase is not "NaKATPase". NFKD has no
# compatibility mapping for them, so fold to Latin first; then the two titles
# stay distinct in the dedup key instead of collapsing to the same string.
_GREEK_FOLD = str.maketrans(
    {
        "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e",
        "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "u", "ν": "n", "ξ": "x",
        "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u",
        "φ": "ph", "ϕ": "ph", "χ": "ch", "ψ": "ps", "ω": "o",
    }
)


def _normalise_title(title: str) -> str:
    """Lower-case, fold accents, strip punctuation, collapse whitespace."""
    t = unicodedata.normalize("NFKD", title.lower().translate(_GREEK_FOLD))
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _carry_metadata(old: Paper, new: Paper) -> Paper:
    """Merge what the discarded duplicate knew onto the surviving record.

    Citation counts come from OpenAlex/S2, retraction and MeSH knowledge only
    from the biomedical backends. Picking the winner by citations alone would
    drop the retraction flag for the exact papers where it matters most, so the
    survivor keeps: ``retracted`` (OR), ``pmid``, ``mesh_terms``/``pub_types``
    (union), and a ``pdf_url`` the winner lacks. Preprint status needs no merge:
    ``_prefer_published`` already made a journal record win when one existed.
    """
    merged = new
    changes: dict[str, object] = {}
    if old.retracted and not new.retracted:
        changes["retracted"] = True
    if old.pmid and not new.pmid:
        changes["pmid"] = old.pmid
    if old.mesh_terms and not set(old.mesh_terms) <= set(new.mesh_terms):
        changes["mesh_terms"] = tuple(dict.fromkeys((*new.mesh_terms, *old.mesh_terms)))
    if old.pub_types and not set(old.pub_types) <= set(new.pub_types):
        changes["pub_types"] = tuple(dict.fromkeys((*new.pub_types, *old.pub_types)))
    if old.pdf_url and not new.pdf_url:
        changes["pdf_url"] = old.pdf_url
    return replace(merged, **changes) if changes else merged


def _mark_retracted(results: list[Paper], idx: int) -> None:
    """Flag an already-stored duplicate as retracted, in place."""
    if not results[idx].retracted:
        results[idx] = replace(results[idx], retracted=True)


def _prefer_published(kept: Paper, candidate: Paper) -> bool:
    """Decide whether duplicate *candidate* should replace the stored *kept*.

    The published version wins over a preprint: once a bioRxiv paper is in a
    journal, citing the preprint link is simply wrong, no matter how many
    citations the preprint portal happened to accumulate. Otherwise the
    higher cited record wins (Semantic Scholar/OpenAlex data over thin
    source-only records).
    """
    if kept.is_preprint and not candidate.is_preprint:
        return True
    if candidate.is_preprint and not kept.is_preprint:
        return False
    return candidate.citation_count > kept.citation_count


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates.  Priority: DOI > arXiv ID > fuzzy title.

    Published records beat preprints (see :func:`_prefer_published`); between
    two records of the same kind the more-cited one wins. The survivor's
    identifiers are re-registered after every merge, so a third record that
    only shares the *loser's* DOI/arXiv id still folds into the right entry.
    """
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def _update_indices(p: Paper, idx: int) -> None:
        """Register all identifiers of *p* in the lookup dicts at *idx*."""
        if p.doi:
            seen_doi[normalise_doi(p.doi)] = idx
        if p.arxiv_id:
            seen_arxiv[p.arxiv_id.strip()] = idx
        norm = _normalise_title(p.title)
        if norm:
            seen_title[norm] = idx

    def _merge_duplicate(idx: int, paper: Paper) -> None:
        """Fold duplicate *paper* into ``results[idx]`` and re-index it.

        Index entries of the discarded record are deliberately *kept*: they
        still point at ``idx``, so a later record that only carries the
        loser's DOI/arXiv id folds into the same entry instead of creating a
        second copy.
        """
        kept = result[idx]
        if _prefer_published(kept, paper):
            result[idx] = _carry_metadata(kept, paper)
        else:
            _mark_retracted(result, idx)
        _update_indices(result[idx], idx)

    for paper in papers:
        is_dup = False

        # Check DOI
        if paper.doi:
            doi_key = normalise_doi(paper.doi)
            if doi_key in seen_doi:
                idx = seen_doi[doi_key]
                _merge_duplicate(idx, paper)
                is_dup = True

        # Check arXiv ID
        if not is_dup and paper.arxiv_id:
            ax_key = paper.arxiv_id.strip()
            if ax_key in seen_arxiv:
                idx = seen_arxiv[ax_key]
                _merge_duplicate(idx, paper)
                is_dup = True

        # Check fuzzy title
        if not is_dup:
            norm = _normalise_title(paper.title)
            if norm and norm in seen_title:
                idx = seen_title[norm]
                _merge_duplicate(idx, paper)
                is_dup = True

        if is_dup:
            continue

        # Not a duplicate — store indices and append
        new_idx = len(result)
        _update_indices(paper, new_idx)
        result.append(paper)

    return result


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    """Generate a combined BibTeX file from a list of papers.

    Duplicate cite keys are fatal in BibTeX — the second entry silently
    shadows the first, so a review that cites two different 2024 Zhang papers
    loses a reference. Keys are de-duplicated with ``a``/``b``/``c`` suffixes
    in input order, which keeps the first occurrence unchanged.
    """
    entries: list[str] = []
    seen: dict[str, int] = {}
    for paper in papers:
        entry = paper.to_bibtex()
        base = paper.cite_key
        count = seen.get(base, 0)
        seen[base] = count + 1
        if count:
            suffix = "abcdefghijklmnopqrstuvwxyz"[count - 1] if count < 26 else str(count)
            entry = entry.replace("{" + base + ",", "{" + base + suffix + ",", 1)
        entries.append(entry)
    return "\n\n".join(entries) + "\n"
