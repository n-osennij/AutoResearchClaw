"""Local query cache for literature search results.

Cache entries are keyed by ``(query, source, limit, year_min)`` so results
fetched under one publication-year policy are never served under another —
``min_year`` changes what each source returns, and a cache hit that ignores it
would silently resurrect the old policy. Entries expire per source (24 h arXiv,
3 days OpenAlex/S2, 7 days the rest).

The directory also carries ``.policy.json``, the policy that filled it, so the
pipeline and ``researchclaw doctor`` can warn when a run's policy differs from
the cached one, or when entries predate policy tracking entirely.
Cache directory: ``.researchclaw_cache/literature/``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DEFAULT_CACHE_DIR = Path(".researchclaw_cache") / "literature"
_POLICY_FILE = ".policy.json"
_TTL_SEC = 86400 * 7  # 7 days — fallback for sources without a TTL above

# Per-source TTLs: arXiv updates daily at midnight, so 24h cache is optimal.
# Citation verification results are permanent (verified papers don't change).
_SOURCE_TTL: dict[str, float] = {
    "arxiv": 86400,         # 24 hours — arXiv metadata updates once/day
    "semantic_scholar": 86400 * 3,  # 3 days
    "openalex": 86400 * 3,  # 3 days
    "citation_verify": 86400 * 365,  # ~permanent
}


def _cache_dir(base: Path | None = None) -> Path:
    d = base or _DEFAULT_CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    return d


def _canonical_source(source: str) -> str:
    """Same alias mapping the searcher uses, so ``s2`` and
    ``semantic_scholar`` share one cache namespace."""
    name = source.strip().lower().replace("-", "_").replace(" ", "_")
    return "semantic_scholar" if name == "s2" else name


def cache_key(query: str, source: str, limit: int, year_min: int = 0) -> str:
    """Deterministic cache key from the query and its year policy."""
    raw = (
        f"{query.strip().lower()}|{_canonical_source(source)}|{limit}|y{int(year_min)}"
    )
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def get_cached(
    query: str,
    source: str,
    limit: int,
    year_min: int = 0,
    *,
    cache_base: Path | None = None,
    ttl: float | None = None,
) -> list[dict[str, Any]] | None:
    """Return cached results or None if miss/expired.

    If *ttl* is not provided, uses source-specific TTL from
    ``_SOURCE_TTL``, falling back to the global ``_TTL_SEC``.
    """
    d = _cache_dir(cache_base)
    key = cache_key(query, source, limit, year_min)
    path = d / f"{key}.json"

    if not path.exists():
        return None

    effective_ttl = ttl if ttl is not None else _SOURCE_TTL.get(source, _TTL_SEC)

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        ts = data.get("timestamp", 0)
        age_sec = time.time() - ts
        if age_sec > effective_ttl:
            logger.debug("Cache expired for key %s (age=%.0fs > ttl=%.0fs)",
                         key, age_sec, effective_ttl)
            return None
        papers = data.get("papers", [])
        if not isinstance(papers, list):
            return None
        age_str = _format_age(age_sec)
        logger.info(
            "[cache] HIT query=%r source=%s year_min=%d age=%s (%d papers)",
            query[:50], source, int(year_min), age_str, len(papers),
        )
        return papers
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return None


def _format_age(seconds: float) -> str:
    """Human-readable age string."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.0f}m"
    if seconds < 86400:
        return f"{seconds / 3600:.1f}h"
    return f"{seconds / 86400:.1f}d"


def put_cache(
    query: str,
    source: str,
    limit: int,
    papers: list[dict[str, Any]],
    year_min: int = 0,
    *,
    cache_base: Path | None = None,
) -> None:
    """Write search results to cache."""
    d = _cache_dir(cache_base)
    key = cache_key(query, source, limit, year_min)
    path = d / f"{key}.json"

    payload = {
        "query": query,
        "source": source,
        "limit": limit,
        "year_min": int(year_min),
        "timestamp": time.time(),
        "papers": papers,
    }
    # Write through a temp file: a concurrent read must never see a partially
    # written entry, and a cache failure must not be reported as a source
    # rate-limit (the caller logs exceptions from this block that way).
    tmp = path.with_name(path.name + ".tmp")
    try:
        tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        os.replace(tmp, path)
        logger.debug("Cached %d papers for key %s", len(papers), key)
    except OSError:
        logger.warning("Could not write literature cache entry %s", path, exc_info=True)
    finally:
        try:
            tmp.unlink()
        except OSError:
            pass


# ---------------------------------------------------------------------------
# Year-policy tracking
# ---------------------------------------------------------------------------


def inspect_search_policy(
    *,
    min_year: int,
    hybrid: bool,
    cache_base: Path | None = None,
) -> dict[str, Any]:
    """Describe the cache relative to the policy that would fill it now.

    Read-only: does not create the directory, so ``doctor`` stays a no-op on a
    clean checkout.

    Returns keys: ``cache_dir``, ``exists``, ``entries``, ``policy`` (the
    recorded policy or None), ``matches`` (recorded policy equals the current
    one) and ``legacy_entries`` (entries exist but no policy was ever recorded,
    i.e. they predate policy tracking).
    """
    d = cache_base or _DEFAULT_CACHE_DIR
    entries = 0
    policy: dict[str, Any] | None = None
    if d.is_dir():
        entries = sum(
            1
            for path in d.glob("*.json")
            if path.name != _POLICY_FILE and path.is_file()
        )
        policy_path = d / _POLICY_FILE
        if policy_path.is_file():
            try:
                raw = json.loads(policy_path.read_text(encoding="utf-8"))
                if isinstance(raw, dict):
                    policy = raw
            except (OSError, json.JSONDecodeError):
                policy = None

    # Only the year floor enters the cache key; ``hybrid`` is recorded but a
    # hybrid toggle reuses every primary-pass entry, so comparing it here
    # would tell users to throw away a warm cache for no reason.
    try:
        matches = bool(policy and int(policy.get("min_year", -1)) == int(min_year))
    except (TypeError, ValueError):
        matches = False
    return {
        "cache_dir": str(d),
        "exists": d.is_dir(),
        "entries": entries,
        "policy": policy,
        "matches": matches,
        "legacy_entries": entries > 0 and policy is None,
    }


def note_search_policy(
    *,
    min_year: int,
    hybrid: bool,
    cache_base: Path | None = None,
) -> dict[str, Any]:
    """Record the current policy and report what the cache held before.

    Returns the :func:`inspect_search_policy` snapshot taken **before** the
    policy file was written, plus ``previous`` (the policy that filled the
    cache) so the caller can warn about a policy switch.
    """
    snapshot = inspect_search_policy(
        min_year=min_year, hybrid=hybrid, cache_base=cache_base
    )
    d = _cache_dir(cache_base)
    policy_path = d / _POLICY_FILE
    policy_path.write_text(
        json.dumps(
            {
                "min_year": int(min_year),
                "hybrid": bool(hybrid),
                "updated_at": time.time(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    snapshot["previous"] = snapshot.get("policy")
    return snapshot


def clear_cache(*, cache_base: Path | None = None) -> int:
    """Remove all cache files. Return count of files deleted."""
    d = _cache_dir(cache_base)
    count = 0
    for f in d.glob("*.json"):
        if f.name == _POLICY_FILE:
            # Keep the policy record: the next fill can still tell which
            # package filled the now-empty directory.
            continue
        try:
            f.unlink()
        except OSError:
            continue
        count += 1
    return count


def cache_stats(*, cache_base: Path | None = None) -> dict[str, Any]:
    """Return cache statistics (policy file excluded from the entry count)."""
    d = _cache_dir(cache_base)
    files = [f for f in d.glob("*.json") if f.name != _POLICY_FILE]
    total_bytes = sum(f.stat().st_size for f in files)
    return {
        "entries": len(files),
        "total_bytes": total_bytes,
        "cache_dir": str(d),
    }
