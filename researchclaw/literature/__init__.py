"""Real literature search and citation management for ResearchClaw.

Provides API clients for OpenAlex, Semantic Scholar, arXiv, OpenReview,
Crossref, PubMed and CORE, plus unified search with deduplication and BibTeX
generation.  All network I/O uses stdlib ``urllib`` — **zero** extra pip
dependencies.
"""

from researchclaw.literature.models import Author, Paper
from researchclaw.literature.search import (
    SourceKeys,
    known_sources,
    search_papers,
    unknown_sources,
)
from researchclaw.literature.verify import (
    CitationResult,
    VerificationReport,
    VerifyStatus,
    verify_citations,
)

__all__ = [
    "Author",
    "CitationResult",
    "Paper",
    "SourceKeys",
    "VerificationReport",
    "VerifyStatus",
    "search_papers",
    "known_sources",
    "unknown_sources",
    "verify_citations",
]
