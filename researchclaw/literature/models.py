"""Data models for literature search results.

Paper and Author are frozen dataclasses — immutable after creation.
``Paper.to_bibtex()`` generates a valid BibTeX entry from metadata,
and ``Paper.cite_key`` returns a normalised citation key.
"""

from __future__ import annotations

import hashlib
import html
import re
import urllib.parse
import unicodedata
from dataclasses import dataclass, field

_DOI_PREFIX_RE = re.compile(
    r"^(?:https?://(?:dx\.)?doi\.org/|doi:)\s*", re.IGNORECASE
)

_HTML_TAG_RE = re.compile(r"</?[a-zA-Z][^>]*>")
_WHITESPACE_RE = re.compile(r"\s+")

#: Canonical literature-backend names. Mirrors the ``_SOURCES`` table in
#: ``researchclaw/literature/search.py``; kept in this leaf module (no HTTP
#: clients, no config import) so ``researchclaw.config.validate_config`` can
#: reject a typo instead of letting Stage 4 skip the backend with a warning.
LITERATURE_SOURCES: tuple[str, ...] = (
    "openalex",
    "semantic_scholar",
    "arxiv",
    "openreview",
    "crossref",
    "pubmed",
    "europepmc",
    "core",
)


def _plain(text: str) -> str:
    """Strip provider markup from a metadata string.

    Crossref, Semantic Scholar and OpenAlex all hand back HTML fragments
    (``<i>``, ``<sub>``, ``&amp;``) in titles and venue names. Escaping those
    as BibTeX would print the tags; stripping them first is the only
    character-level answer that does not need a full LaTeX tokeniser. Control
    whitespace (tabs, newlines from API payloads) collapses to a single space
    because a newline inside a ``title = {...}`` field breaks the entry.
    """
    value = _HTML_TAG_RE.sub("", text or "")
    value = html.unescape(value)
    return _WHITESPACE_RE.sub(" ", value).strip()


def _ascii_fold(text: str) -> str:
    """Fold accents to ASCII, dropping characters with no ASCII form."""
    decomposed = unicodedata.normalize("NFKD", text)
    return decomposed.encode("ascii", "ignore").decode("ascii")


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
        """Return an ASCII key component for citation keys.

        Handles the shapes that actually appear in provider metadata:
        ``Smith, John`` (comma convention), ``John Smith Jr.`` (suffix),
        ``The Global Burden of Disease Study`` (collective name — PubMed
        ``CollectiveName``, where the *last* word is never the surname), and
        non-Latin names, which fold to nothing and would otherwise collapse
        every Cyrillic/CJK paper onto ``unknown``.
        """
        raw = _author_name(self.name)
        if not raw:
            return "unknown"
        folded = _ascii_fold(raw)
        if not folded.strip():
            # Non-Latin script: no ASCII form exists, so keep a stable short
            # digest of the raw name. Unique enough for cite keys, and much
            # better than mapping every such paper onto ``unknown``.
            digest = hashlib.blake2s(raw.encode("utf-8"), digest_size=2).hexdigest()
            return f"u{digest}"
        if "," in folded:
            # ``Smith, John`` / ``Smith, Jr., J.``: the surname is before the
            # first comma, which is exactly what the comma convention means.
            return _key_part(folded.split(",", 1)[0])
        words = [w for w in folded.split() if w.strip(".").lower() not in _NAME_SUFFIXES]
        if not words:
            words = folded.split()
        if _COLLECTIVE_RE.search(folded):
            # Collective author: the first meaningful word identifies it far
            # better than the trailing ``Study``/``Group``.
            meaningful = [
                w for w in words if _ascii_fold(w).lower() not in _STOPWORDS
            ]
            return _key_part((meaningful or words)[0])
        return _key_part(words[-1])


_NAME_SUFFIXES = {
    "jr",
    "jr.",
    "sr",
    "sr.",
    "ii",
    "iii",
    "iv",
    "phd",
    "md",
    "msc",
    "bsc",
}

# A collective name has no surname; these words are the tell.
_COLLECTIVE_RE = re.compile(
    r"\b(group|consortium|committee|collaboration|team|study|organisation|"
    r"organization|institute|university|department|society|network|initiative|"
    r"consortium|foundation|agency|association)\b",
    re.IGNORECASE,
)


def _author_name(name: str) -> str:
    """Normalise an author name, dropping ``et al.`` and stray punctuation."""
    value = _WHITESPACE_RE.sub(" ", (name or "").strip())
    value = re.sub(r"\bet\s+al\.?", "", value, flags=re.IGNORECASE).strip(" ,;")
    return value


def _key_part(text: str) -> str:
    """Lowercase ASCII slug suitable for a cite key."""
    slug = re.sub(r"[^a-z0-9]", "", _ascii_fold(text).lower())
    return slug[:20] or "unknown"


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
    # Not part of identity: a cache-restored record must compare equal to the
    # same record fetched fresh, override or not.
    _bibtex_override: str = field(default="", repr=False, compare=False)

    # ------------------------------------------------------------------
    # Citation key
    # ------------------------------------------------------------------

    @property
    def cite_key(self) -> str:
        """Normalised citation key: ``lastname<year><keyword>``.

        Example: ``smith2024transformer``. Not guaranteed unique — two
        papers by the same author in the same year collapse to one key, so
        anything that emits a bibliography must pass the entries through
        ``papers_to_bibtex``/``dedupe_bibtex_entries`` (which is keyed on
        the rendered text, overrides included).
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
        names = [
            _bibtex_escape(a.name) for a in self.authors if (a.name or "").strip()
        ]
        authors_str = " and ".join(names) or "Unknown"

        venue = _plain(self.venue)
        arxiv_venue = _is_arxiv_venue(venue)

        notes: list[str] = []
        if venue and not arxiv_venue and _CONFERENCE_RE.search(venue.lower()):
            entry_type = "inproceedings"
            venue_field = f"  booktitle = {{{_bibtex_escape(venue)}}},"
        elif self.is_preprint:
            # bioRxiv / medRxiv / Research Square: no journal, no volume. The
            # canonical entry names the server in ``howpublished``; when the
            # provider left ``venue`` empty, recover the server from the url
            # rather than emit an entry with no provenance at all.
            entry_type = "misc"
            server = venue or _preprint_server(self.url)
            venue_field = (
                f"  howpublished = {{{_bibtex_escape(server)}}}," if server else ""
            )
            if self.arxiv_id and server != "arXiv":
                # Classic styles ignore ``eprint`` on @misc; the note is what
                # survives there.
                notes.append(f"arXiv preprint arXiv:{self.arxiv_id}")
        elif self.arxiv_id and (not venue or arxiv_venue):
            # arXiv paper: use standard format with eprint ID
            entry_type = "article"
            venue_field = f"  journal = {{arXiv preprint arXiv:{self.arxiv_id}}},"
        elif venue:
            entry_type = "article"
            venue_field = f"  journal = {{{_bibtex_escape(venue)}}},"
        else:
            entry_type = "article"
            venue_field = ""

        if self.retracted:
            notes.insert(0, "RETRACTED")
        elif self.concern:
            notes.insert(0, "Expression of concern")

        lines = [f"@{entry_type}{{{key},"]
        lines.append(f"  title = {{{_bibtex_escape(self.title)}}},")
        lines.append(f"  author = {{{authors_str}}},")
        # No ``year = {Unknown}``: a non-numeric year breaks numeric sorting in
        # plain/unsrt and prints "Unknown" under natbib. Omitting the field is
        # what every reference manager does for an undated record.
        if self.year:
            lines.append(f"  year = {{{self.year}}},")
        if venue_field:
            lines.append(venue_field)
        if notes:
            lines.append(f"  note = {{{' ; '.join(notes)}}},")
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

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Paper":
        """Rebuild a :class:`Paper` from :meth:`to_dict` output.

        Single source of truth for the cache round-trip: the hand-rolled
        deserialiser this replaces had to be edited for every new field, and
        forgetting one (``concern``) silently dropped it for months.
        """

        def _text(value: object) -> str:
            return "" if value is None else str(value)

        def _strs(value: object) -> tuple[str, ...]:
            if isinstance(value, (list, tuple)):
                return tuple(str(v) for v in value if v)
            return ()

        def _is_preprint(data: dict[str, object]) -> bool:
            """``is_preprint`` with a fallback for rows cached before the field.

            An arXiv record, or one whose venue is an arXiv category code, is a
            preprint by construction; defaulting those to ``False`` let a cached
            arXiv row outrank its published version in dedup.
            """
            if data.get("is_preprint") is not None:
                return bool(data["is_preprint"])
            return (
                str(data.get("source") or "").casefold() == "arxiv"
                or _is_arxiv_venue(_text(data.get("venue")))
            )

        def _int(value: object) -> int:
            if isinstance(value, bool) or value is None:
                return 0
            try:
                return int(value)  # type: ignore[arg-type]
            except (TypeError, ValueError, OverflowError):
                # ``inf``/``nan`` arrive as JSON numbers and cannot be ints.
                return 0

        raw_authors = data.get("authors") or ()
        authors: tuple[Author, ...] = ()
        if isinstance(raw_authors, (list, tuple)):
            authors = tuple(
                Author(
                    name=_text(entry.get("name")),
                    affiliation=_text(entry.get("affiliation")),
                )
                for entry in raw_authors
                if isinstance(entry, dict)
            )

        return cls(
            paper_id=_text(data.get("paper_id")),
            title=_text(data.get("title")),
            authors=authors,
            year=_int(data.get("year")),
            abstract=_text(data.get("abstract")),
            venue=_text(data.get("venue")),
            citation_count=_int(data.get("citation_count")),
            doi=_text(data.get("doi")),
            arxiv_id=_text(data.get("arxiv_id")),
            url=_text(data.get("url")),
            pdf_url=_text(data.get("pdf_url")),
            source=_text(data.get("source")),
            pmid=_text(data.get("pmid")),
            is_preprint=_is_preprint(data),
            pub_types=_strs(data.get("pub_types")),
            mesh_terms=_strs(data.get("mesh_terms")),
            retracted=bool(data.get("retracted", False)),
            concern=bool(data.get("concern", False)),
            _bibtex_override=_text(data.get("_bibtex_override")),
        )


_CONFERENCE_RE = re.compile(
    r"\b(conference|proceedings|workshop|neurips|nips|icml|iclr|aaai|cvpr|"
    r"iccv|eccv|acl|emnlp|naacl|sigir|kdd|www|ijcai)\b"
)

# arXiv category codes that providers put into ``venue`` instead of a
# journal. Dotted archives (``cs.CY``, ``cond-mat.mtrl-sci``) and the bare
# endonyms (``quant-ph``) are the two shapes the APIs emit.
_ARXIV_PLAIN_VENUES = {
    "cs", "math", "stat", "eess", "physics", "q-bio", "q-fin", "nlin", "econ",
    "astro-ph", "cond-mat", "gr-qc", "math-ph", "quant-ph", "hep-ex",
    "hep-lat", "hep-ph", "hep-th", "nucl-ex", "nucl-th",
}
_ARXIV_ARCHIVE_RE = re.compile(
    r"^(?:cs|math|stat|eess|physics|q-bio|q-fin|astro-ph|cond-mat|gr-qc|"
    r"math-ph|quant-ph|hep-ex|hep-lat|hep-ph|hep-th|nucl-ex|nucl-th|nlin|econ)"
    r"\.[A-Za-z][A-Za-z-]*$"
)

_PREPRINT_HOSTS = {
    "biorxiv.org": "bioRxiv",
    "medrxiv.org": "medRxiv",
    "arxiv.org": "arXiv",
    "ssrn.com": "SSRN",
    "researchsquare.com": "Research Square",
    "preprints.org": "Preprints.org",
    "techrxiv.org": "TechRxiv",
    "chemrxiv.org": "ChemRxiv",
    "authorea.com": "Authorea",
    "osf.io": "OSF Preprints",
}


def _is_arxiv_venue(venue: str) -> bool:
    """True for the arXiv category codes that leak into ``venue``."""
    text = (venue or "").strip()
    if not text:
        return False
    return text.lower() in _ARXIV_PLAIN_VENUES or bool(
        _ARXIV_ARCHIVE_RE.match(text)
    )


def _preprint_server(url: str) -> str:
    """Recover the preprint server name from a record url."""
    host = urllib.parse.urlsplit(url or "").netloc.lower()
    for domain, label in _PREPRINT_HOSTS.items():
        if host == domain or host.endswith("." + domain):
            return label
    return ""


# Characters that break a field when unescaped. Deliberately excludes ``\\``,
# ``{``/``}`` and ``$``: providers ship real LaTeX (``R{\'i}os``, ``$x_1$``,
# ``{World Health Organization}``) and escaping that prints the markup in the
# PDF instead of the symbol it encodes.
_BIBTEX_SPECIALS = {
    "&": r"\&",
    "%": r"\%",
    "#": r"\#",
    "_": r"\_",
    "$": r"\$",
    "~": "\\textasciitilde{}",
    "^": "\\textasciicircum{}",
}

# A backslash or a balanced pair of dollar signs means the value already *is*
# LaTeX. Escaping it again yields a visible backslash in the PDF (``100\%``)
# and destroys math mode.
_LATEX_MARKUP_RE = re.compile(r"\\|\$[^$]*\$")


def _bibtex_escape(text: str) -> str:
    """Escape a metadata value for a BibTeX field."""
    value = _plain(text)
    if _LATEX_MARKUP_RE.search(value):
        return value
    escaped = "".join(_BIBTEX_SPECIALS.get(ch, ch) for ch in value)
    if escaped.count("{") != escaped.count("}"):
        # Unbalanced protection braces close the field early; drop them.
        escaped = escaped.replace("{", "").replace("}", "")
    return escaped


# In a url/doi/eprint field a raw ``%`` truncates the entry at the comment and
# a raw ``#`` raises "Illegal parameter number"; braces and backslashes end or
# corrupt the field. Escaping is cosmetic for ``\url{}`` consumers, but a
# truncated entry loses the reference outright, so escape.
_URL_SPECIALS = {
    "%": r"\%",
    "#": r"\#",
    "&": r"\&",
    "{": r"\{",
    "}": r"\}",
    "\\": r"\textbackslash{}",
}


def _bibtex_escape_url(text: str) -> str:
    """Escape a URL/DOI for a BibTeX field."""
    return "".join(_URL_SPECIALS.get(ch, ch) for ch in _plain(text))


def _looks_like_bibtex(entry: str) -> bool:
    """Structural check for a cached BibTeX override.

    Balanced braces alone are not what makes an override safe: an extra entry
    appended after the first one (``@article{x, title={a}}

@article{evil,
    ...}``) balances perfectly and would inject a fabricated reference into
    ``references.bib``. So the scan demands exactly one top-level entry that
    opens at ``@``, closes on the last character, and has nothing but
    whitespace after it.
    """
    if not entry.startswith("@"):
        return False
    depth = 0
    opened = False
    for index, char in enumerate(entry):
        if char == "@" and depth == 0:
            if opened:
                return False  # a second entry at top level → injection
            opened = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth < 0:
                return False
            if depth == 0:
                # Entry closed: only trailing whitespace may follow.
                return not entry[index + 1 :].strip()
    return False  # never closed


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
