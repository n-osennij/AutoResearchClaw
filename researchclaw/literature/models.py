"""Data models for literature search results.

Paper and Author are frozen dataclasses — immutable after creation.
``Paper.to_bibtex()`` generates a valid BibTeX entry from metadata,
and ``Paper.cite_key`` returns a normalised citation key.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

_DOI_PREFIX_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE
)


def normalise_doi(value: str) -> str:
    """Canonical DOI for comparison: no ``https://doi.org/`` / ``doi:`` prefix.

    ``10.1/ABC.`` and ``https://doi.org/10.1/abc`` are the same paper; without
    this, the dedup pass keeps both and the verifier 404s on the prefixed form.
    """
    doi = _DOI_PREFIX_RE.sub("", (value or "").strip())
    return doi.strip().strip("<>").rstrip(".,;").strip().lower()


@dataclass(frozen=True)
class Author:
    """A paper author."""

    name: str
    affiliation: str = ""

    def last_name(self) -> str:
        """Return ASCII-folded last name for citation keys."""
        parts = self.name.strip().split()
        raw = parts[-1] if parts else "unknown"
        # Fold accented characters to ASCII
        nfkd = unicodedata.normalize("NFKD", raw)
        ascii_name = nfkd.encode("ascii", "ignore").decode("ascii")
        return re.sub(r"[^a-zA-Z]", "", ascii_name).lower() or "unknown"


@dataclass(frozen=True)
class Paper:
    """A single paper from Semantic Scholar, arXiv, or similar sources.

    Fields are designed to hold the union of metadata available from both
    Semantic Scholar and arXiv APIs.
    """

    paper_id: str
    title: str
    authors: tuple[Author, ...] = ()
    year: int = 0
    abstract: str = ""
    venue: str = ""
    citation_count: int = 0
    doi: str = ""
    arxiv_id: str = ""
    url: str = ""
    pdf_url: str = ""  # open-access document URL (CORE and friends)
    # "semantic_scholar" | "arxiv" | "openalex" | "openreview" | "crossref"
    # | "pubmed" | "europepmc" | "core"
    source: str = ""
    # Biomedical provenance, filled by PubMed / Europe PMC. ``is_preprint``
    # matters for ranking and for the review text: a 2025 bioRxiv paper has no
    # citations yet, so anything that sorts by citations will bury it.
    pmid: str = ""
    is_preprint: bool = False
    pub_types: tuple[str, ...] = ()
    mesh_terms: tuple[str, ...] = ()
    retracted: bool = False
    # "Expression of Concern": citable but flagged; unlike ``retracted`` this
    # must never cause the paper to be dropped.
    concern: bool = False
    _bibtex_override: str = field(default="", repr=False)

    # ------------------------------------------------------------------
    # Citation key
    # ------------------------------------------------------------------

    @property
    def cite_key(self) -> str:
        """Normalised citation key: ``lastname<year><keyword>``.

        Example: ``smith2024transformer``
        """
        last = self.authors[0].last_name() if self.authors else "anon"
        yr = str(self.year) if self.year else "0000"
        # First meaningful noun-ish word from title (>3 chars, alpha only)
        kw = ""
        for word in self.title.split():
            cleaned = re.sub(r"[^a-zA-Z]", "", word).lower()
            if len(cleaned) > 3 and cleaned not in _STOPWORDS:
                kw = cleaned
                break
        # 40 chars is plenty to disambiguate and keeps the .bib key readable;
        # an unbounded key comes from a long unbroken token and breaks column
        # widths in every bibliography style.
        return f"{last}{yr}{kw}"[:40]

    # ------------------------------------------------------------------
    # BibTeX generation
    # ------------------------------------------------------------------

    def to_bibtex(self) -> str:
        """Generate a BibTeX entry string.

        If ``_bibtex_override`` was populated (e.g. from CrossRef), return
        that directly.  Otherwise construct from metadata.
        """
        if self._bibtex_override:
            override = self._bibtex_override.strip()
            # Cached/edited entries come from disk: a stray ``}`` closes the
            # field early and corrupts the rest of references.bib. Accept only
            # a balanced entry that starts like one; otherwise re-render.
            if _looks_like_bibtex(override):
                return override

        key = self.cite_key
        authors_str = " and ".join(_bibtex_escape(a.name) for a in self.authors) or "Unknown"

        # T1.4: Detect arXiv category codes used as venue (e.g. "cs.CY", "math.OC")
        # These are NOT journal names and must be treated as arXiv preprints.
        _venue = self.venue or ""
        _is_arxiv_category = bool(
            re.match(
                r"^(?:cs|math|stat|eess|physics|q-bio|q-fin|astro-ph|cond-mat|"
                r"gr-qc|hep-ex|hep-lat|hep-ph|hep-th|nlin|nucl-ex|nucl-th|"
                r"quant-ph)\.[A-Z]{2}$",
                _venue,
            )
        )

        # Decide entry type
        if _venue and not _is_arxiv_category and any(
            kw in _venue.lower()
            for kw in (
                "conference",
                "proc",
                "workshop",
                "neurips",
                "icml",
                "iclr",
                "aaai",
                "cvpr",
                "acl",
                "emnlp",
                "naacl",
                "eccv",
                "iccv",
                "sigir",
                "kdd",
                "www",
                "ijcai",
            )
        ):
            entry_type = "inproceedings"
            venue_field = f"  booktitle = {{{_bibtex_escape(_venue)}}},"
        elif self.is_preprint:
            # bioRxiv / medRxiv / Research Square: no journal, no volume —
            # the canonical entry is @misc naming the preprint server, with
            # a plain url field instead of a \url{} macro (no LaTeX package
            # assumption for whoever imports references.bib).
            entry_type = "misc"
            venue_field = (
                f"  howpublished = {{{_bibtex_escape(_venue)}}},"
                if _venue
                else ""
            )
        elif self.arxiv_id and (not _venue or _is_arxiv_category):
            # arXiv paper: use standard format with eprint ID
            entry_type = "article"
            venue_field = f"  journal = {{arXiv preprint arXiv:{self.arxiv_id}}},"
        else:
            entry_type = "article"
            venue_field = (
                f"  journal = {{{_bibtex_escape(_venue or 'Unknown')}}}," if _venue else ""
            )

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{_bibtex_escape(self.title)}}},")
        lines.append(f"  author = {{{authors_str}}},")
        lines.append(f"  year = {{{self.year or 'Unknown'}}},")
        if venue_field:
            lines.append(venue_field)
        if self.doi:
            lines.append(f"  doi = {{{_bibtex_escape_url(normalise_doi(self.doi))}}},")
        if self.arxiv_id:
            lines.append(f"  eprint = {{{_bibtex_escape_url(self.arxiv_id)}}},")
            lines.append("  archiveprefix = {arXiv},")
        if self.url:
            lines.append(f"  url = {{{_bibtex_escape_url(self.url)}}},")
        lines.append("}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialisation helpers (for JSONL output)
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, object]:
        """Serialise to a plain dict for JSON/JSONL output."""
        return {
            "paper_id": self.paper_id,
            "title": self.title,
            "authors": [
                {"name": a.name, "affiliation": a.affiliation} for a in self.authors
            ],
            "year": self.year,
            "abstract": self.abstract,
            "venue": self.venue,
            "citation_count": self.citation_count,
            "doi": self.doi,
            "arxiv_id": self.arxiv_id,
            "url": self.url,
            "pdf_url": self.pdf_url,
            "source": self.source,
            "cite_key": self.cite_key,
            "pmid": self.pmid,
            "is_preprint": self.is_preprint,
            "pub_types": list(self.pub_types),
            "mesh_terms": list(self.mesh_terms),
            "retracted": self.retracted,
            "concern": self.concern,
        }


# BibTeX treats these as markup; an unescaped "&" in a title is a broken
# .bib file that Zotero/Overleaf silently truncate.
_BIBTEX_SPECIALS = {
    "\\": r"\textbackslash{}",
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def _bibtex_escape(text: str) -> str:
    """Escape the characters BibTeX reads as markup."""
    return "".join(_BIBTEX_SPECIALS.get(ch, ch) for ch in text)


# In a url/doi/eprint field only these are actually dangerous: ``%`` starts a
# LaTeX comment and an unbalanced brace ends the field early. Backslashes are
# escaped too, since a Windows-style path in a url would otherwise be read as
# a command.
_URL_SPECIALS = {"%": r"\%", "{": r"\{", "}": r"\}", "\\": r"\textbackslash{}"}


def _bibtex_escape_url(text: str) -> str:
    """Escape a URL/DOI for a BibTeX field."""
    return "".join(_URL_SPECIALS.get(ch, ch) for ch in text)


def _looks_like_bibtex(entry: str) -> bool:
    """Cheap structural check for a cached BibTeX override."""
    if not entry.startswith("@") or "{" not in entry:
        return False
    if entry.count("{") != entry.count("}"):
        return False
    return entry.rstrip().endswith("}")


# Common English stopwords to skip when picking a keyword for cite_key
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "from",
        "that",
        "this",
        "into",
        "over",
        "upon",
        "about",
        "through",
        "using",
        "based",
        "towards",
        "toward",
        "between",
        "under",
        "more",
        "than",
        "when",
        "what",
        "which",
        "where",
        "does",
        "have",
        "been",
        "some",
        "each",
        "also",
        "much",
        "very",
        "learning",  # too generic for ML papers
    }
)
