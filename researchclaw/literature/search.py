"""Unified literature search with deduplication.

Combines results from the academic backends in ``_SOURCES`` — OpenAlex,
Semantic Scholar, arXiv, OpenReview, Crossref, PubMed, Europe PMC and CORE —
deduplicates by DOI → arXiv ID → fuzzy title match, and returns a merged
list sorted by citation count (descending).

Source priority: OpenAlex (most generous limits) → Semantic Scholar → arXiv;
the remaining five are additive and opt-in via ``literature_search.sources``.
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
import math
import re
import time
import unicodedata
import urllib.error
from typing import cast

from researchclaw.literature.arxiv_client import search_arxiv
from researchclaw.literature.core_client import search_core
from researchclaw.literature.crossref_client import search_crossref
from researchclaw.literature.europepmc_client import search_europepmc
from researchclaw.literature.models import (
    Paper,
    _is_arxiv_venue,
    normalise_doi,
)
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
# Cap per search_papers() call. 15 min is enough for eight healthy backends and
# short enough that a stuck one cannot swallow an entire stage.
DEFAULT_SEARCH_DEADLINE_SEC = 900.0


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
    return search_europepmc(
        query, limit=limit, year_min=year_min, email=keys.openalex_email
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
    # Europe PMC is the biomedical superset (MEDLINE + preprints + full text);
    # listed under both its own name and the shorthand people type in configs.
    "europepmc": _SourceSpec(
        "europepmc", "Europe PMC", _europepmc_adapter, 0.5
    ),
    "core": _SourceSpec("core", "CORE", _core_adapter, 0.5),
}
# Shorthand people type in configs.
_SOURCES["europe_pmc"] = _SOURCES["europepmc"]


def normalise_source(name: str) -> str:
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
    """Best-effort int for cache rows; never raises.

    ``json.loads`` produces ``Infinity``/``NaN`` for a corrupt or hand-edited
    cache file, and ``int(float("inf"))`` raises ``OverflowError`` — which used
    to escape the per-row ``try`` and abort the whole restore.
    """
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value) if math.isfinite(value) else default
    if isinstance(value, str):
        try:
            return int(float(value)) if value.strip() else default
        except (ValueError, OverflowError):
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
            papers.append(Paper.from_dict(d))
        except (KeyError, TypeError, ValueError):
            # One corrupt cache entry must not lose the whole result set.
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
    deadline_sec: float = DEFAULT_SEARCH_DEADLINE_SEC,
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
    deadline_sec:
        Wall-clock budget for the whole call. Each backend can retry for
        minutes (arXiv is unbounded), and a stage runs ~13 queries, so without
        a budget a slow day at one source turns into hours and then looks like
        a hang. ``0`` disables the budget.

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

    started = time.monotonic()
    seen_specs: set[str] = set()
    for src in sources:
        if deadline_sec > 0 and time.monotonic() - started > deadline_sec:
            logger.warning(
                "Literature search deadline (%.0fs) reached after %d/%d sources; "
                "returning %d paper(s) found so far",
                deadline_sec,
                len(seen_specs),
                len(sources),
                len(all_papers),
            )
            break
        spec = _SOURCES.get(normalise_source(src))
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
            # Only cache non-empty answers: an adapter returns [] both for "no
            # hits" and for "the source is down", and the TTL is days long, so
            # caching the failure would make one 503 hide real papers until the
            # entry expires.
            if papers:
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
        ) as exc:
            # Not every failure is throttling: a wrong ``limit`` type or a dead
            # endpoint also lands here, and calling that a rate limit made the
            # logs useless. The fallback is the same (serve cache, degrade) but
            # the message names the actual class, with the traceback at DEBUG.
            logger.warning(
                "[literature] Source %s failed (%s: %s) for %r — trying cache",
                src,
                type(exc).__name__,
                exc,
                query,
            )
            logger.debug("source %s failure detail", src, exc_info=True)
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


# Greek letters carry meaning in biomedical titles: α-synuclein and
# β-synuclein are different proteins. NFKD has no compatibility mapping for
# them (they are not accented Latin letters), so transliterate to their
# conventional Latin spellings; "α-synuclein" then matches the ASCII
# "a-synuclein" that a journal typesets, which is the pair that actually
# collides in practice.
_GREEK_FOLD = str.maketrans(
    {
        "α": "a", "β": "b", "γ": "g", "δ": "d", "ε": "e", "ζ": "z", "η": "e",
        "θ": "th", "ι": "i", "κ": "k", "λ": "l", "μ": "u", "ν": "n", "ξ": "x",
        "ο": "o", "π": "p", "ρ": "r", "σ": "s", "ς": "s", "τ": "t", "υ": "u",
        "φ": "ph", "ϕ": "ph", "χ": "ch", "ψ": "ps", "ω": "o",
    }
)


def normalise_title(title: str) -> str:
    """Lower-case, fold accents, strip punctuation, collapse whitespace.

    NFKD runs *first*: it decomposes accented Latin (``é`` → ``e`` +
    combining acute) and accented Greek (``ά`` → ``α`` + combining acute), so
    the Greek fold sees a bare letter and the combining mark is dropped
    afterwards. Folding before NFKD left ``ά-`` as ``α`` + accent, which the
    ``[^a-z0-9]`` strip then deleted — losing the letter entirely.
    """
    t = unicodedata.normalize("NFKD", title.lower()).translate(_GREEK_FOLD)
    t = "".join(ch for ch in t if not unicodedata.combining(ch))
    t = re.sub(r"[^a-z0-9\s]", "", t)
    return re.sub(r"\s+", " ", t).strip()


def _title_is_latin(title: str) -> bool:
    """True when the title is Latin or Greek script throughout."""
    return all(ord(ch) < 0x0400 for ch in title if not ch.isspace())


def _canonical_title(title: str) -> str:
    """Dedup key for a title, preserving non-Latin script.

    ASCII-only normalisation reduces "研究一：results" and "研究二：results" to
    the same key, so CJK titles are keyed on a script-preserving form instead
    of being dropped: without a key, two records of the same Chinese paper
    (Crossref + CNKI) stay separate rows, references and bibtex entries.
    """
    if _title_is_latin(title):
        return normalise_title(title)
    folded = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(folded.split()).strip(" .,:;!?…")


def _carry_metadata(old: Paper, new: Paper) -> Paper:
    """Merge what the discarded duplicate knew onto the surviving record.

    Citation counts come from OpenAlex/S2, retraction and MeSH knowledge only
    from the biomedical backends. Picking the winner by citations alone would
    drop the retraction flag for the exact papers where it matters most, so the
    survivor keeps: ``retracted``/``concern`` (OR), ``pmid``,
    ``mesh_terms``/``pub_types``
    (union), and a ``pdf_url`` the winner lacks. Preprint status needs no merge:
    ``_prefer_published`` already made a journal record win when one existed.
    """
    merged = new
    changes: dict[str, object] = {}
    if old.retracted and not new.retracted:
        changes["retracted"] = True
    if old.concern and not new.concern:
        changes["concern"] = True
    if old.pmid and not new.pmid:
        changes["pmid"] = old.pmid
    if old.mesh_terms and not set(old.mesh_terms) <= set(new.mesh_terms):
        changes["mesh_terms"] = tuple(dict.fromkeys((*new.mesh_terms, *old.mesh_terms)))
    if old.pub_types and not set(old.pub_types) <= set(new.pub_types):
        # A journal record must not inherit the loser's preprint label: the
        # Type column would then say "Preprint" for peer-reviewed work.
        merged_types = tuple(
            dict.fromkeys((*new.pub_types, *old.pub_types))
        )
        changes["pub_types"] = tuple(
            t for t in merged_types if new.is_preprint or t.casefold() != "preprint"
        )
    if old.pdf_url and not new.pdf_url:
        changes["pdf_url"] = old.pdf_url
    # A thin winner (a journal stub from Crossref) must not erase what the
    # loser knew: the DOI feeds the bibtex, the venue the Type column, the
    # arXiv id a preprint link, and the citation count the ranking.
    for field in ("doi", "venue", "arxiv_id", "url"):
        if not getattr(new, field) and getattr(old, field):
            changes[field] = getattr(old, field)
    if old.citation_count > new.citation_count:
        changes["citation_count"] = old.citation_count
    merged = replace(merged, **changes) if changes else merged
    # Merging a cited preprint with an uncited journal record produced an
    # uncited entry that then ranked below unrelated papers.
    if merged.citation_count != max(old.citation_count, new.citation_count):
        merged = replace(merged, citation_count=max(old.citation_count, new.citation_count))
    return merged


def _prefer_published(kept: Paper, candidate: Paper) -> bool:
    """Decide whether duplicate *candidate* should replace the stored *kept*.

    The published version wins over a preprint: once a bioRxiv paper is in a
    journal, citing the preprint link is simply wrong, no matter how many
    citations the preprint portal happened to accumulate. Otherwise the
    higher cited record wins (Semantic Scholar/OpenAlex data over thin
    source-only records).
    """
    kept_pre = kept.is_preprint or _is_arxiv_venue(kept.venue)
    cand_pre = candidate.is_preprint or _is_arxiv_venue(candidate.venue)
    if kept_pre and not cand_pre:
        return True
    if cand_pre and not kept_pre:
        return False
    return candidate.citation_count > kept.citation_count


def _arxiv_key(arxiv_id: str) -> str:
    """Identity key for an arXiv id: ``arXiv:2401.00001v3`` → ``2401.00001``."""
    text = (arxiv_id or "").strip()
    if text.lower().startswith("arxiv:"):
        text = text.split(":", 1)[1].strip()
    return re.sub(r"v\d+$", "", text)


def _deduplicate(papers: list[Paper]) -> list[Paper]:
    """Remove duplicates.  Priority: DOI > arXiv ID > PMID > fuzzy title.

    Published records beat preprints (see :func:`_prefer_published`); between
    two records of the same kind the more-cited one wins, and the survivor
    keeps the loser's metadata (DOI, venue, arXiv id, citations). The
    survivor's identifiers are re-registered after every merge, so a third
    record that only shares the *loser's* DOI/arXiv id/pmid still folds into
    the right entry.
    """
    seen_doi: dict[str, int] = {}
    seen_arxiv: dict[str, int] = {}
    seen_pmid: dict[str, int] = {}
    seen_title: dict[str, int] = {}
    result: list[Paper] = []

    def _update_indices(p: Paper, idx: int) -> None:
        """Register all identifiers of *p* in the lookup dicts at *idx*."""
        if p.doi:
            seen_doi[normalise_doi(p.doi)] = idx
        if p.arxiv_id:
            seen_arxiv[_arxiv_key(p.arxiv_id)] = idx
        if p.pmid:
            # PubMed and Europe PMC serve the same MEDLINE records. The DOI is
            # missing from plenty of older articles, so pmid is the only shared
            # identity those two backends give us.
            seen_pmid[p.pmid.strip()] = idx
        norm = _canonical_title(p.title)
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
            # The candidate loses, but its flags still count: the discarded
            # copy may be the only record marked retracted or carrying an
            # expression of concern. Its DOI/arXiv id keeps pointing at the
            # survivor through ``_update_indices`` below.
            result[idx] = _carry_metadata(paper, kept)
        # Both records' identifiers now point at the survivor: a third record
        # that only carries the discarded one's arXiv id must not become a
        # second copy of the same paper.
        _update_indices(paper, idx)
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
            ax_key = _arxiv_key(paper.arxiv_id)
            if ax_key in seen_arxiv:
                idx = seen_arxiv[ax_key]
                _merge_duplicate(idx, paper)
                is_dup = True

        # Check PubMed id
        if not is_dup and paper.pmid:
            idx = seen_pmid.get(paper.pmid.strip())
            if idx is not None:
                _merge_duplicate(idx, paper)
                is_dup = True

        # Check fuzzy title
        if not is_dup:
            norm = _canonical_title(paper.title)
            idx = seen_title.get(norm) if norm else None
            if idx is not None and _canonical_title(result[idx].title) != norm:
                # The entry was replaced by a different paper (two records
                # shared a DOI but not a title); a title-only match against a
                # different paper is not evidence that they are the same.
                del seen_title[norm]
                idx = None
            if idx is not None:
                _merge_duplicate(idx, paper)
                is_dup = True

        if is_dup:
            continue

        # Not a duplicate — store indices and append
        new_idx = len(result)
        _update_indices(paper, new_idx)
        result.append(paper)

    return result


_KEY_RE = re.compile(r"^(\s*@[A-Za-z]+\s*\{)([^,{}]*)(,)")


def _suffix(n: int) -> str:
    """0→'a' … 25→'z', 26→'aa' … (letters only, never digits)."""
    letters = ""
    n += 1
    while n:
        n, rem = divmod(n - 1, 26)
        letters = chr(97 + rem) + letters
    return letters


def _entry_key(entry: str) -> str:
    """Return the cite key inside one rendered entry, or ""."""
    match = _KEY_RE.match(entry)
    return match.group(2).strip() if match else ""


def uniquify_bibtex_entry(
    entry: str,
    seen: dict[str, int],
    *,
    reserved: set[str] | None = None,
) -> tuple[str, str]:
    """Give one rendered entry a key nobody used yet; return (entry, key).

    ``seen`` maps an already-used key to how many times it occurred. The key is
    read back from the entry text, so a cached ``_bibtex_override`` that carries
    its own key is handled too. Callers that also advertise ``cite_key`` to the
    LLM (stage 4 writes both ``candidates.jsonl`` and ``references.bib``) must
    store the returned key, otherwise the advertised key and the key written in
    the bibliography diverge and the citation is dropped as missing.
    """
    match = _KEY_RE.match(entry)
    key = match.group(2).strip() if match else ""
    if not key:
        return entry, ""  # unparseable: keep it, cannot rename safely
    # BibTeX matches keys case-insensitively (biber rejects a case mismatch
    # outright), so "Smith2024" and "smith2024" are the same key to the
    # compiler even though they are different strings to Python.
    fold = key.casefold()
    count = seen.get(fold, 0)
    seen[fold] = count + 1
    if not count:
        # First use of this key keeps it, even though the same key is in
        # ``reserved`` — that entry is the one that owns it.
        return entry, key
    # Skip candidates that collide with a key already in use: a key made of the
    # suffix letter itself ("aaaa…") truncates to the same string, so a blind
    # rename would emit the duplicate again. ``reserved`` holds every key that
    # appears later in the same batch: renaming onto one of them would hand
    # paper #2 the key paper #3 legitimately carries.
    attempt = count or 1
    while attempt < count + 1000:
        suffix = _suffix(attempt - 1)
        new_key = key[: max(1, 40 - len(suffix))] + suffix
        new_fold = new_key.casefold()
        if (
            new_fold != fold
            and new_fold not in seen
            and (reserved is None or new_fold not in reserved)
        ):
            break
        attempt += 1
    else:  # pragma: no cover - 1000 consecutive collisions
        return entry, key
    seen[new_fold] = 1
    renamed = (
        entry[: match.start()]
        + match.group(1)
        + new_key
        + match.group(3)
        + entry[match.end() :]
    )
    return renamed, new_key


def dedupe_bibtex_entries(entries: Sequence[str]) -> list[str]:
    """Rename duplicate cite keys in already-rendered BibTeX entries.

    Operates on the emitted text, not on ``Paper.cite_key``: a cached
    ``_bibtex_override`` carries its own key, so bookkeeping keyed on
    ``cite_key`` would miss the clash and ship two entries BibTeX resolves to
    one.
    """
    seen: dict[str, int] = {}
    # Two passes: every key present in the batch is claimed in advance, so a
    # generated suffix can never land on one that a later entry already has.
    reserved = {fold for fold in (_entry_key(e).casefold() for e in entries) if fold}
    return [
        uniquify_bibtex_entry(entry, seen, reserved=reserved)[0] for entry in entries
    ]


def papers_to_bibtex(papers: Sequence[Paper]) -> str:
    """Generate a combined BibTeX file from a list of papers.

    Duplicate cite keys are fatal in BibTeX — the second entry silently
    shadows the first, so a review that cites two different 2024 Zhang papers
    loses a reference. Keys are de-duplicated with ``a``/``b``/``c`` suffixes
    in input order, which keeps the first occurrence unchanged.
    """
    return (chr(10) * 2).join(dedupe_bibtex_entries([p.to_bibtex() for p in papers])) + chr(10)
