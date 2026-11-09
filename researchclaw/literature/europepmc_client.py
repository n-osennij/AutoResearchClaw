"""Europe PMC client — MEDLINE + preprints + full-text links, one endpoint.

Europe PMC is the cheapest way to cover the biomedical literature that the
general backends miss: it indexes MEDLINE (same records as PubMed) but adds the
bioRxiv / medRxiv / Research Square preprints, publication types and the
retraction flags in one response — MeSH descriptors are only in PubMed's XML,
so we do not claim them here. It also carries open-access full-text links.
One query can span peer-reviewed articles and preprints, which is exactly the
split that matters for molecular biology (bioRxiv is primary literature there,
months before the journal version exists).

No API key, no documented quota for the ``rest/search`` endpoint.

Public API
----------
- ``search_europepmc(query, limit, year_min, preprints_only)`` → ``list[Paper]``
- ``EPMC_PREPRINT_FILTER`` — ``"SRC:PPR"``, the field to AND into a query when
  only preprints are wanted.
"""

from __future__ import annotations

import html
import logging
import re
import urllib.parse
from typing import Any

from researchclaw.literature._http import fetch_json
from researchclaw.literature.models import Author, Paper

logger = logging.getLogger(__name__)

_BASE = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
_TIMEOUT = 60
_USER_AGENT = "ResearchClaw/1.0"
# Europe PMC caps pageSize at 1000; asking for more returns an error.
_MAX_PAGE = 1000
# ``FIRST_PDATE`` is the earliest publication date (preprint date for PPR),
# which is the field that matches "published in year >= N" in every other
# backend. ``PUB_YEAR`` would silently drop preprints, whose pubYear is the
# year the record was created.
_YEAR_FIELD = "FIRST_PDATE"

EPMC_PREPRINT_FILTER = "SRC:PPR"
# Same vocabulary as PubMed: a partially retracted paper is still retracted.
_RETRACTION_TYPES = frozenset({"Retraction in", "Partial Retraction in"})

_TAG_RE = re.compile(
    r"</?(?:i|b|em|strong|sub|sup|u|p|br|sec|title|h[1-6])(?:\s[^<>]*)?/?>",
    re.IGNORECASE,
)


def _clean(value: Any) -> str:
    """Strip the inline markup Europe PMC leaves in titles and abstracts."""
    if not value:
        return ""
    text = _TAG_RE.sub(" ", str(value))
    return re.sub(r"\s+", " ", html.unescape(text)).strip()


def _build_query(query: str, year_min: int) -> str:
    parts = [f"({query})"]
    if year_min > 0:
        parts.append(f"{_YEAR_FIELD}:[{year_min}-01-01 TO 3000-12-31]")
    return " AND ".join(parts)


def _authors(payload: dict[str, Any]) -> tuple[Author, ...]:
    listed = ((payload.get("authorList") or {}).get("author")) or []
    authors = []
    for entry in listed:
        if not isinstance(entry, dict):
            continue
        name = entry.get("fullName") or " ".join(
            part
            for part in (
                entry.get("firstName"),
                entry.get("lastName"),
            )
            if part
        )
        name = _clean(name)
        if name:
            authors.append(
                Author(name=name, affiliation=_clean(entry.get("affiliation")))
            )
    if not authors:
        # Preprints often only carry the flat ``authorString``.
        flat = _clean(payload.get("authorString")).rstrip(".")
        if flat:
            authors = [
                Author(name=name.strip())
                for name in flat.split(",")
                if name.strip() and name.strip().lower() != "et al"
            ]
    return tuple(authors)


def _pub_types(payload: dict[str, Any]) -> tuple[str, ...]:
    raw = ((payload.get("pubTypeList") or {}).get("pubType")) or []
    return tuple(_clean(t) for t in raw if _clean(t))


def _retracted(payload: dict[str, Any]) -> bool:
    """True when the record itself is retracted.

    Two markers appear on a retracted article: the ``Retracted Publication``
    publication type, and a ``Retraction in`` reference to the notice. The
    notice itself carries ``Retraction of Publication`` / ``Retraction of`` —
    the opposite direction — and is a legitimate citation (you cite it *to*
    warn readers), so it must not be flagged here.
    """
    if "Retracted Publication" in _pub_types(payload):
        return True
    corrections = (
        (payload.get("commentCorrectionList") or {}).get("commentCorrection")
    ) or []
    return any(
        isinstance(entry, dict) and entry.get("type") in _RETRACTION_TYPES
        for entry in corrections
    )


def _concern(payload: dict[str, Any]) -> bool:
    """Expression of Concern: citable, but must be surfaced, never dropped."""
    return "Expression of Concern" in _pub_types(payload)


def _venue(payload: dict[str, Any]) -> str:
    journal = ((payload.get("journalInfo") or {}).get("journal") or {}).get("title")
    if journal:
        return _clean(journal)
    # Preprints: bioRxiv / medRxiv sit in bookOrReportDetails.publisher.
    book = payload.get("bookOrReportDetails") or {}
    return _clean(book.get("publisher"))


def _pdf_url(payload: dict[str, Any]) -> str:
    urls = ((payload.get("fullTextUrlList") or {}).get("fullTextUrl")) or []
    for entry in urls:
        if not isinstance(entry, dict):
            continue
        if entry.get("documentStyle") == "pdf" and entry.get("url"):
            return _clean(entry["url"])
    return ""


def _to_paper(payload: dict[str, Any]) -> Paper | None:
    title = _clean(payload.get("title"))
    if not title:
        return None
    identifier = _clean(payload.get("id"))
    source_code = _clean(payload.get("source")).upper() or "MED"
    pmid = _clean(payload.get("pmid"))
    doi = _clean(payload.get("doi"))
    year_text = _clean(payload.get("pubYear"))
    year = int(year_text[:4]) if year_text[:4].isdigit() else 0
    pub_types = _pub_types(payload)
    return Paper(
        paper_id=f"pmid:{pmid}" if pmid else (f"doi:{doi}" if doi else f"epmc:{identifier}"),
        title=title,
        authors=_authors(payload),
        year=year,
        abstract=_clean(payload.get("abstractText")),
        venue=_venue(payload),
        citation_count=_as_int(payload.get("citedByCount")),
        doi=doi,
        url=f"https://europepmc.org/article/{source_code}/{identifier}",
        pdf_url=_pdf_url(payload),
        source="europepmc",
        pmid=pmid,
        is_preprint=source_code == "PPR" or "Preprint" in pub_types,
        pub_types=pub_types,
        retracted=_retracted(payload),
        concern=_concern(payload),
    )


def _as_int(value: Any, default: int = 0) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return int(value)
    if isinstance(value, str):
        try:
            return int(value)
        except ValueError:
            return default
    return default


def search_europepmc(
    query: str,
    *,
    limit: int = 50,
    year_min: int = 0,
    preprints_only: bool = False,
) -> list[Paper]:
    """Search Europe PMC and return parsed records.

    Parameters
    ----------
    query:
        Free-text query. Boolean operators and Europe PMC fields
        (``TITLE:``, ``SRC:PPR``, ``PUB_TYPE:``) are passed through.
    limit:
        Maximum records (capped at 1000 by the API).
    year_min:
        If > 0, only records first published in this year or later.
    preprints_only:
        AND ``SRC:PPR`` into the query — bioRxiv / medRxiv / Research Square
        records only.

    Returns
    -------
    list[Paper]
        Parsed records. Empty list on failure, like every other backend.
    """
    limit = max(1, min(limit, _MAX_PAGE))
    if preprints_only:
        query = f"({query}) AND {EPMC_PREPRINT_FILTER}"
    params = {
        "query": _build_query(query, year_min),
        "format": "json",
        "pageSize": str(limit),
        "resultType": "core",
    }
    url = f"{_BASE}?{urllib.parse.urlencode(params)}"
    try:
        payload = fetch_json(
            url,
            source="Europe PMC",
            headers={"User-Agent": _USER_AGENT},
            timeout=_TIMEOUT,
        )
    except Exception as exc:  # noqa: BLE001 - one backend failing is not fatal
        logger.warning("Europe PMC search failed for %r: %s", query, exc)
        return []

    try:
        results = ((payload.get("resultList") or {}).get("result")) or []
        papers = [
            paper
            for paper in (_to_paper(entry) for entry in results if isinstance(entry, dict))
            if paper is not None
        ]
    except (AttributeError, TypeError, ValueError) as exc:
        # A schema change or a truncated response must degrade to "no hits",
        # never take down the literature stage.
        logger.warning("Europe PMC: unparseable response for %r: %s", query, exc)
        return []
    if year_min > 0:
        # The date filter is server-side, but records with a missing/odd
        # FIRST_PDATE slip through; trust but verify.
        papers = [p for p in papers if not p.year or p.year >= year_min]
    logger.debug("Europe PMC: %d/%d records kept for %r", len(papers), len(results), query)
    return papers
