"""Stages 3-6: Search strategy, literature collection, screening, and knowledge extraction."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import yaml

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.llm.client import LLMClient
from researchclaw.pipeline._helpers import (
    StageResult,
    _build_fallback_queries,
    _chat_with_prompt,
    _extract_topic_keywords,
    _extract_yaml_block,
    _get_evolution_overlay,
    _parse_jsonl_rows,
    _read_prior_artifact,
    _safe_filename,
    _safe_json_loads,
    _utcnow_iso,
    _write_jsonl,
)
from researchclaw.pipeline.stages import Stage, StageStatus
from researchclaw.prompts import PromptManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _expand_search_queries(queries: list[str], topic: str) -> list[str]:
    """Expand search queries for broader literature coverage.

    Generates additional queries by extracting key phrases from the topic
    and creating focused sub-queries. This ensures we find papers even when
    the original queries are too narrow or specific for arXiv.
    """
    expanded = list(queries)  # keep originals
    seen = {q.lower().strip() for q in queries}

    # Extract key phrases from topic by splitting on common delimiters
    # e.g. "Comparing A, B, and C on X with Y" → ["A", "B", "C", "X", "Y"]
    topic_words = topic.split()

    # Generate shorter, broader queries from the topic
    if len(topic_words) > 5:
        # First 5 words as a broader query
        broad = " ".join(topic_words[:5])
        if broad.lower().strip() not in seen:
            expanded.append(broad)
            seen.add(broad.lower().strip())

        # Last 5 words as another perspective
        tail = " ".join(topic_words[-5:])
        if tail.lower().strip() not in seen:
            expanded.append(tail)
            seen.add(tail.lower().strip())

    # Add "survey" and "benchmark" variants of the topic
    for suffix in ("survey", "benchmark", "comparison"):
        # Take first 4 content words + suffix
        short_topic = " ".join(topic_words[:4])
        variant = f"{short_topic} {suffix}"
        if variant.lower().strip() not in seen:
            expanded.append(variant)
            seen.add(variant.lower().strip())

    return expanded


def _paper_identity_keys(paper: Any) -> set[str]:
    """Every identifier a paper can be recognised by across search passes.

    The primary and foundational passes come from different backends, so one
    paper often arrives with a DOI in one pass and only an arXiv id (or a
    slightly different title) in the other. Comparing one winner-only identity
    per paper lets those copies through; comparing all three does not.
    """
    keys: set[str] = set()
    doi = str(getattr(paper, "doi", "") or "").strip().lower()
    if doi:
        keys.add(f"doi:{doi}")
    arxiv_id = str(getattr(paper, "arxiv_id", "") or "").strip().lower()
    if arxiv_id:
        keys.add(f"arxiv:{arxiv_id}")
        # 1706.03762v5 and 1706.03762 are the same paper
        arxiv_base = re.sub(r"v\d+$", "", arxiv_id)
        keys.add(f"arxiv:{arxiv_base}")
    title = str(getattr(paper, "title", "") or "")
    normalised = re.sub(r"[^a-z0-9\s]", "", title.lower())
    normalised = re.sub(r"\s+", " ", normalised).strip()
    if normalised:
        keys.add(f"title:{normalised}")
    return keys


def _coerce_int(value: Any, default: int) -> int:
    """Best-effort int conversion for hand-edited artifacts."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------
# Stage executors
# ---------------------------------------------------------------------------


def _collect_queries(raw: Any, out: list[str]) -> None:
    """Append query strings from ``raw`` into ``out``.

    Search plans come back in several shapes depending on the model: a list
    of plain strings, or a list of dicts wrapping the query under an
    arbitrary key (``{"query": "..."}``, ``{"boolean": "..."}``). Accepting
    both keeps a plan from silently yielding zero queries.
    """
    if not isinstance(raw, list):
        return
    for q in raw:
        if isinstance(q, str) and q.strip():
            out.append(q.strip())
        elif isinstance(q, dict):
            for v in q.values():
                if isinstance(v, str) and v.strip():
                    out.append(v.strip())


def _execute_search_strategy(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    problem_tree = _read_prior_artifact(run_dir, "problem_tree.md") or ""
    topic = config.research.topic
    plan: dict[str, Any] | None = None
    sources: list[dict[str, Any]] | None = None
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "search_strategy")
        sp = _pm.for_stage("search_strategy", evolution_overlay=_overlay, topic=topic, problem_tree=problem_tree)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict):
            yaml_text = str(payload.get("search_plan_yaml", "")).strip()
            if yaml_text:
                try:
                    parsed = yaml.safe_load(_extract_yaml_block(yaml_text))
                except yaml.YAMLError:
                    parsed = None
                if isinstance(parsed, dict):
                    plan = parsed
            src = payload.get("sources", [])
            if isinstance(src, list):
                sources = [item for item in src if isinstance(item, dict)]
    model_plan_parsed = plan is not None
    if plan is None:
        # Build smart fallback queries by extracting key terms from topic
        # instead of using the raw (often very long) topic string.
        _fallback_queries = _build_fallback_queries(topic)
        plan = {
            "topic": topic,
            "generated": _utcnow_iso(),
            "search_strategies": [
                {
                    "name": "keyword_core",
                    "queries": _fallback_queries[:5],
                    "sources": ["arxiv", "semantic_scholar", "openreview"],
                    "max_results_per_query": 60,
                },
                {
                    "name": "backward_forward_citation",
                    "queries": _fallback_queries[5:10] or _fallback_queries[:3],
                    "sources": ["semantic_scholar", "google_scholar"],
                    "depth": 1,
                },
            ],
            "filters": {
                "min_year": 2020,
                "language": ["en"],
                "peer_review_preferred": True,
            },
            "deduplication": {"method": "title_doi_hash", "fuzzy_threshold": 0.9},
        }
    if not sources:
        sources = [
            {
                "id": "arxiv",
                "name": "arXiv",
                "type": "api",
                "url": "https://export.arxiv.org/api/query",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
            {
                "id": "semantic_scholar",
                "name": "Semantic Scholar",
                "type": "api",
                "url": "https://api.semanticscholar.org/graph/v1/paper/search",
                "status": "available",
                "query": topic,
                "verified_at": _utcnow_iso(),
            },
        ]
    if config.openclaw_bridge.use_web_fetch:
        for src in sources:
            try:
                response = adapters.web_fetch.fetch(str(src.get("url", "")))
                src["status"] = (
                    "verified"
                    if response.status_code in (200, 301, 302, 405)
                    else "unreachable"
                )
                src["http_status"] = response.status_code
            except Exception:  # noqa: BLE001
                src["status"] = "unknown"
    (stage_dir / "search_plan.yaml").write_text(
        yaml.dump(plan, default_flow_style=False, allow_unicode=True),
        encoding="utf-8",
    )
    (stage_dir / "sources.json").write_text(
        json.dumps(
            {"sources": sources, "count": len(sources), "generated": _utcnow_iso()},
            indent=2,
        ),
        encoding="utf-8",
    )

    # F1.5: Extract queries from plan for Stage 4 real literature search.
    # The publication-year floor comes from config, not from the model: the
    # search-strategy prompt is not asked for a date range, so an LLM-chosen
    # value was effectively random. ``research.min_year: 0`` disables the
    # filter entirely (needed for pre-2020 foundations).
    queries_list: list[str] = []
    year_min = config.research.min_year
    if isinstance(plan, dict):
        strategies = (
            plan.get("search_strategies")
            or plan.get("search_phases")
            or plan.get("phases")
            or []
        )
        if isinstance(strategies, list):
            for strat in strategies:
                if isinstance(strat, dict):
                    qs = strat.get("queries", [])
                    if isinstance(qs, list):
                        _collect_queries(qs, queries_list)
        # Alternate schema: queries under
        # query_strategies.<sub_question>.{boolean_seeds, queries}.
        if not queries_list:
            qstrats = plan.get("query_strategies", {})
            if isinstance(qstrats, dict):
                for sub in qstrats.values():
                    if not isinstance(sub, dict):
                        continue
                    for key in ("boolean_seeds", "queries"):
                        _collect_queries(sub.get(key, []), queries_list)
        # Alternate schema: a flat top-level `queries` list.
        if not queries_list:
            _collect_queries(plan.get("queries"), queries_list)

    # --- Sanitize queries: shorten overly long queries ---
    # LLMs often produce the full topic title as a query, which is too long for
    # arXiv and Semantic Scholar (they work best with 3-8 keyword queries).
    _stop = {
        "a", "an", "the", "of", "for", "in", "on", "and", "or", "with",
        "to", "by", "from", "its", "is", "are", "was", "be", "as", "at",
        "via", "using", "based", "study", "analysis", "empirical",
        "towards", "toward", "into", "exploring", "comparison", "tasks",
        "effectiveness", "investigation", "comprehensive", "novel",
        "challenge", "challenges", "gaps", "gap", "critical", "survey", "review",
    }

    def _extract_search_terms(text: str) -> list[str]:
        """Extract meaningful search terms from text, removing stop words."""
        return [
            w for w in re.split(r"[^a-zA-Z0-9]+", text)
            if w.lower() not in _stop and len(w) > 1
        ]

    _MAX_QUERY_LEN = 60  # characters — beyond this, shorten to keywords
    _SEARCH_SUFFIXES = ["benchmark", "survey", "seminal", "state of the art"]

    def _shorten_query(q: str, max_kw: int = 6) -> str:
        """Shorten a query to *max_kw* keywords, preserving any trailing suffix."""
        q_stripped = q.strip()
        # Check if query ends with a known search suffix
        suffix = ""
        q_core = q_stripped
        for sfx in _SEARCH_SUFFIXES:
            if q_stripped.lower().endswith(sfx):
                suffix = sfx
                q_core = q_stripped[: -len(sfx)].strip()
                break
        # Extract keywords from the core part
        kws = _extract_search_terms(q_core)
        shortened = " ".join(kws[:max_kw])
        if suffix:
            shortened = f"{shortened} {suffix}"
        return shortened

    if queries_list:
        sanitized: list[str] = []
        for q in queries_list:
            if len(q) > _MAX_QUERY_LEN:
                shortened = _shorten_query(q)
                if shortened.strip():
                    sanitized.append(shortened)
            else:
                sanitized.append(q)
        queries_list = sanitized

    def _build_default_search_queries(topic_text: str) -> list[str]:
        """Generate concept-style search queries from the topic instead of copying the title."""
        _words = _extract_search_terms(topic_text)
        if not _words:
            return [topic_text[:60]]
        kw_primary = " ".join(_words[:6])
        kw_short = " ".join(_words[:4])
        kw_alt = " ".join(_words[1:5]) if len(_words) > 4 else kw_short
        return [
            kw_primary,
            f"{kw_short} benchmark",
            f"{kw_short} survey",
            kw_alt,
            f"{kw_short} recent advances",
        ]

    fell_back_to_defaults = False
    if not queries_list:
        queries_list = _build_default_search_queries(topic)
        fell_back_to_defaults = True

    # Ensure minimum query diversity — if dedup leaves too few, add variants
    _all_kw = _extract_search_terms(topic)
    _seen_q: set[str] = set()
    unique_queries: list[str] = []
    for q in queries_list:
        q_lower = q.strip().lower()
        if q_lower and q_lower not in _seen_q:
            _seen_q.add(q_lower)
            unique_queries.append(q.strip())
    # If we have fewer than 5 unique queries, generate supplemental keyword variants
    if len(unique_queries) < 5 and len(_all_kw) >= 3:
        supplements = [
            " ".join(_all_kw[:4]) + " survey",
            " ".join(_all_kw[:4]) + " benchmark",
            " ".join(_all_kw[1:5]),  # shifted window for diversity
            " ".join(_all_kw[:3]) + " comparison",
            " ".join(_all_kw[:3]) + " deep learning",
            " ".join(_all_kw[2:6]),  # another shifted window
        ]
        for s in supplements:
            s_lower = s.strip().lower()
            if s_lower not in _seen_q:
                _seen_q.add(s_lower)
                unique_queries.append(s.strip())
            if len(unique_queries) >= 8:
                break
    queries_list = unique_queries
    silent_fallback = fell_back_to_defaults and model_plan_parsed
    if silent_fallback:
        logger.warning(
            "Stage 3: model plan parsed but no queries harvested; "
            "queries.json fell back to topic-derived defaults"
        )
    queries_meta = {
        "queries": queries_list,
        "year_min": year_min,
        "model_queries_extracted": model_plan_parsed and not fell_back_to_defaults,
        "fallback_reason": (
            "model_plan_used_unknown_schema" if silent_fallback else None
        ),
    }
    (stage_dir / "queries.json").write_text(
        json.dumps(queries_meta, indent=2),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.SEARCH_STRATEGY,
        status=StageStatus.DONE,
        artifacts=("search_plan.yaml", "sources.json", "queries.json"),
        evidence_refs=(
            "stage-03/search_plan.yaml",
            "stage-03/sources.json",
            "stage-03/queries.json",
        ),
    )


# Substring-matched against ``research.domains``. The list has to cover the
# common biomedical departments, not just the ones whose spelling contains
# "bio"/"med": with the genealogy of a pharmacology or immunology group, a miss
# here silently switches MEDLINE off for exactly the users who need it.
# Short hints must match as whole words: as substrings, "med" fires on
# "multimedia", "social media" and "remedial", and "bio" on "biometrics"
# and "biomimetic". The longer hints below are unambiguous inside a word.
_BIOMED_SHORT_HINTS = ("bio", "med", "nurs")
_BIOMED_SHORT_RE = re.compile(r"(?<![a-z])(?:bio|med|nurs)(?![a-z])")
_BIOMED_DOMAIN_HINTS = (
    "biolog",
    "biomed",
    "biochem",
    "medic",
    "pharma",
    "clin",
    "neuro",
    "epidem",
    "health",
    "genom",
    "oncolog",
    "immun",
    "pharm",
    "drug",
    "physiolog",
    "cardio",
    "patholog",
    "microbio",
    "infect",
    "vaccin",
    "metabol",
    "nutrition",
    "dental",
    "surg",
)


# Backends that only make sense for life-sciences topics. PubMed's query
# translation is word-level ("KV inference" expands to "kv" AND (cache...)
# AND (infer...)), so on an ML topic it returns hundreds of irrelevant clinical
# hits; Europe PMC indexes the same MEDLINE records plus preprints.
_BIOMED_SOURCES = frozenset({"pubmed", "europepmc", "europe_pmc"})


def _sources_for_domains(
    sources: tuple[str, ...], domains: tuple[str, ...], *, gate_pubmed: bool = True
) -> tuple[str, ...]:
    """Drop the biomedical backends when the run targets another domain.

    Both PubMed and Europe PMC are word-level-matching biomedical indexes: on
    an ML topic they return hundreds of irrelevant clinical hits and pollute
    screening. The gate keys off ``research.domains`` because a topic string
    alone does not say whether "sequence" means genomes or tokens.
    """
    if not gate_pubmed:
        return sources
    # Imported here, not at module scope: ``researchclaw.literature.search``
    # pulls in every source client, and this module is imported by the stage
    # registry long before any search runs.
    from researchclaw.literature.search import normalise_source

    labels = " ".join(str(d).lower() for d in domains)
    if not labels.strip():
        # The gate is on but there is no domain to judge by. Dropping the
        # biomedical backends on a guess is the worse failure — a biomedical
        # run whose config omits ``research.domains`` loses PubMed and Europe
        # PMC entirely — so keep them and say so.
        logger.warning(
            "[literature] pubmed_domain_gate is enabled but research.domains "
            "is empty — keeping PubMed/Europe PMC (set domains to enable the gate)"
        )
        return sources
    if _BIOMED_SHORT_RE.search(labels) or any(
        hint in labels for hint in _BIOMED_DOMAIN_HINTS
    ):
        return sources
    filtered = tuple(
        src for src in sources if normalise_source(src) not in _BIOMED_SOURCES
    )
    if len(filtered) != len(sources):
        dropped = sorted(
            src for src in sources if normalise_source(src) in _BIOMED_SOURCES
        )
        logger.info(
            "[literature] %s skipped — domain(s) %s are not biomedical "
            "(set literature_search.pubmed_domain_gate: false to force it)",
            ", ".join(dropped),
            ", ".join(str(d) for d in domains) or "none",
        )
    return filtered


def _execute_literature_collect(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    """Stage 4: Collect literature — prefer real APIs, fallback to LLM."""
    topic = config.research.topic

    # Read queries.json from Stage 3 (F1.5 output)
    queries_text = _read_prior_artifact(run_dir, "queries.json")
    queries_data = _safe_json_loads(queries_text or "{}", {})
    queries: list[str] = queries_data.get("queries", [topic])
    year_min = _coerce_int(
        queries_data.get("year_min", config.research.min_year),
        config.research.min_year,
    )

    # --- Try real API search first ---
    candidates: list[dict[str, Any]] = []
    from researchclaw.literature.search import uniquify_bibtex_entry

    bibtex_entries: list[str] = []
    # Keys already emitted into references.bib, so a later entry (including the
    # Scholar and fallback paths) cannot shadow an earlier one.
    bibtex_keys: dict[str, int] = {}
    real_search_succeeded = False
    foundational_papers: list[Any] = []
    hybrid_enabled = False
    cache_state: dict[str, Any] = {}

    try:
        from researchclaw.literature.search import (
            SourceKeys,
            normalise_source,
            search_papers_multi_query,
            papers_to_bibtex,
        )

        # Expand queries for broader coverage
        expanded_queries = _expand_search_queries(queries, config.research.topic)
        literature_config = config.literature_search
        s2_api_key = (
            literature_config.s2_api_key
            or config.llm.s2_api_key
            or os.environ.get(literature_config.s2_api_key_env, "")
        )
        openalex_api_key = (
            literature_config.openalex_api_key
            or os.environ.get(literature_config.openalex_api_key_env, "")
        )
        pubmed_api_key = (
            literature_config.pubmed_api_key
            or os.environ.get(literature_config.pubmed_api_key_env, "")
        )
        core_api_key = (
            literature_config.core_api_key
            or os.environ.get(literature_config.core_api_key_env, "")
        )
        sources = _sources_for_domains(
            literature_config.sources,
            tuple(str(d) for d in config.research.domains),
            gate_pubmed=literature_config.pubmed_domain_gate,
        )
        hybrid_enabled = (
            bool(getattr(config.research, "hybrid_search", False)) and year_min > 0
        )
        logger.info(
            "[literature] Searching %d queries (expanded from %d) "
            "across %s",
            len(expanded_queries),
            len(queries),
            " -> ".join(sources),
        )
        if year_min > 0 and not hybrid_enabled:
            logger.warning(
                "[literature] YEAR FILTER ACTIVE: research.min_year=%d — "
                "papers published before %d are excluded by every source "
                "(pre-2020 foundations such as Transformer 2017 / Adam 2014 "
                "will not be found). Set research.min_year: 0 to include them, "
                "or enable research.hybrid_search.",
                year_min,
                year_min,
            )
        elif hybrid_enabled:
            logger.info(
                "[literature] Year floor %d applies to the primary pass; the "
                "hybrid pass adds pre-%d foundations (cap %d papers).",
                year_min,
                year_min,
                int(getattr(config.research, "foundational_max_papers", 40)),
            )
        search_keys = SourceKeys(
            s2_api_key=s2_api_key,
            openalex_email=literature_config.openalex_email,
            openalex_api_key=openalex_api_key,
            pubmed_api_key=pubmed_api_key,
            core_api_key=core_api_key,
        )
        papers = search_papers_multi_query(
            expanded_queries,
            limit_per_query=literature_config.max_results_per_query,
            sources=sources,
            year_min=year_min,
            keys=search_keys,
            inter_query_delay=literature_config.inter_query_delay_sec,
        )

        if hybrid_enabled:
            # Second pass without the year floor: it recovers the foundational
            # papers the filtered ranking pushed out. Only papers older than the
            # floor are kept, capped and sorted by citations, so the recent
            # cohort is not crowded out. A cap or per-query limit of 0 disables
            # the pass outright instead of paying for a search that adds nothing.
            found_cap = _coerce_int(
                getattr(config.research, "foundational_max_papers", 40), 40
            )
            found_limit = _coerce_int(
                getattr(config.research, "foundational_results_per_query", 10), 10
            )
            if found_cap <= 0 or found_limit <= 0:
                logger.info(
                    "[literature] Hybrid foundational pass skipped "
                    "(cap=%d, results/query=%d)",
                    found_cap,
                    found_limit,
                )
            else:
                try:
                    logger.info(
                        "[literature] Hybrid foundational pass: %d queries "
                        "without the year filter (limit %d/source, cap %d papers)",
                        len(expanded_queries or queries),
                        found_limit,
                        found_cap,
                    )
                    older = search_papers_multi_query(
                        expanded_queries or queries,
                        limit_per_query=found_limit,
                        sources=sources,
                        year_min=0,
                        keys=search_keys,
                        inter_query_delay=literature_config.inter_query_delay_sec,
                    )
                    seen: set[str] = set()
                    for paper in papers:
                        seen |= _paper_identity_keys(paper)
                    for paper in older:
                        if len(foundational_papers) >= found_cap:
                            break
                        if not paper.year or paper.year >= year_min:
                            continue
                        paper_keys = _paper_identity_keys(paper)
                        if paper_keys & seen:
                            continue
                        seen |= paper_keys
                        foundational_papers.append(paper)
                    # Merge in citation order: the char cap in Stage 5 keeps
                    # the head of the list, and appending would put every
                    # foundation last (and first to be truncated).
                    papers = sorted(
                        papers + foundational_papers,
                        key=lambda p: (
                            getattr(p, "citation_count", 0) or 0,
                            getattr(p, "year", 0) or 0,
                        ),
                        reverse=True,
                    )
                    logger.info(
                        "[literature] Hybrid pass added %d foundational papers "
                        "(published before %d)",
                        len(foundational_papers),
                        year_min,
                    )
                except Exception:  # noqa: BLE001 - keep the primary results
                    logger.warning(
                        "Hybrid foundational pass failed — continuing with the "
                        "primary search results",
                        exc_info=True,
                    )
        # Record the policy only after a successful fill, so an aborted run
        # does not label the cache with a policy it never filled. There is no
        # need to compare ``hybrid``: it does not enter the cache key.
        if papers:
            try:
                from researchclaw.literature.cache import note_search_policy

                cache_state = note_search_policy(
                    min_year=year_min, hybrid=hybrid_enabled
                )
                previous = cache_state.get("previous")
                if previous and _coerce_int(
                    previous.get("min_year", -1), -1
                ) != year_min:
                    logger.warning(
                        "[literature] Literature cache was filled under a "
                        "different year policy (min_year=%s); those entries are "
                        "not reused. Delete %s to reclaim space.",
                        previous.get("min_year"),
                        cache_state.get("cache_dir"),
                    )
                if cache_state.get("legacy_entries"):
                    logger.warning(
                        "[literature] %d cache entries predate year-policy "
                        "tracking and are no longer reused. Delete %s to "
                        "reclaim space.",
                        cache_state.get("entries", 0),
                        cache_state.get("cache_dir"),
                    )
            except Exception:  # noqa: BLE001 - cache policy must never block a run
                logger.debug("Cache policy check skipped", exc_info=True)
        # --- Retraction filter (PubMed / Europe PMC records) ---
        # A retracted paper is not evidence: citing it without the notice is
        # how a literature review loses credibility. Cheaper to drop here than
        # to hope the screener notices the ``retracted`` flag.
        retracted = [p for p in papers if getattr(p, "retracted", False)]
        if retracted:
            papers = [p for p in papers if not getattr(p, "retracted", False)]
            logger.warning(
                "[literature] Dropped %d retracted paper(s): %s",
                len(retracted),
                "; ".join(p.title[:60] for p in retracted[:3]),
            )

        if papers:
            real_search_succeeded = True
            # Count by source
            src_counts: dict[str, int] = {}
            for p in papers:
                src_counts[p.source] = src_counts.get(p.source, 0) + 1
                d = p.to_dict()
                if year_min > 0 and p.year and p.year < year_min:
                    d["year_band"] = "foundational"
                elif year_min > 0:
                    d["year_band"] = "recent"
                else:
                    d["year_band"] = "all"
                d["collected_at"] = _utcnow_iso()
                # ``cite_key`` is derived from author/year/title, so two 2024
                # Zhang papers can collide. Rename here and advertise the same
                # key downstream: the writer stage cites ``[cite_key]`` and the
                # bibliography must contain it verbatim.
                entry, unique_key = uniquify_bibtex_entry(p.to_bibtex(), bibtex_keys)
                if unique_key:
                    d["cite_key"] = unique_key
                candidates.append(d)
                bibtex_entries.append(entry)
            src_str = ", ".join(f"{s}: {n}" for s, n in src_counts.items())
            logger.info(
                "[literature] Found %d papers (%s)", len(papers), src_str
            )
    except Exception:  # noqa: BLE001
        logger.warning(
            "[rate-limit] Literature search failed — falling back to LLM",
            exc_info=True,
        )

    # --- Inject foundational/seminal papers ---
    try:
        from researchclaw.data import load_seminal_papers
        seminal = load_seminal_papers(topic)
        if seminal:
            _existing_titles = {c.get("title", "").lower() for c in candidates}
            _injected = 0
            for sp in seminal:
                if sp.get("title", "").lower() not in _existing_titles:
                    candidates.append({
                        "id": f"seminal-{sp.get('cite_key', '')}",
                        "title": sp.get("title", ""),
                        "source": "seminal_library",
                        "url": "",
                        "year": sp.get("year", 2020),
                        "abstract": f"Foundational paper on {', '.join(sp.get('keywords', [])[:3])}.",
                        "authors": [{"name": sp.get("authors", "")}],
                        "cite_key": sp.get("cite_key", ""),
                        "venue": sp.get("venue", ""),
                        "year_band": "foundational",
                        "collected_at": _utcnow_iso(),
                    })
                    _injected += 1
            if _injected:
                logger.info("Stage 4: Injected %d seminal papers from seed library", _injected)
    except Exception:  # noqa: BLE001
        logger.debug("Seminal paper injection skipped", exc_info=True)

    # --- Fallback: LLM-generated candidates ---
    if not candidates and llm is not None:
        plan_text = _read_prior_artifact(run_dir, "search_plan.yaml") or ""
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_collect")
        sp = _pm.for_stage("literature_collect", evolution_overlay=_overlay, topic=topic, plan_text=plan_text)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("candidates"), list):
            candidates = [row for row in payload["candidates"] if isinstance(row, dict)]

    # --- Web search augmentation (Tavily/DDG + Google Scholar + Crawl4AI) ---
    web_context_parts: list[str] = []
    if config.web_search.enabled:
        try:
            from researchclaw.web.agent import WebSearchAgent

            tavily_key = config.web_search.tavily_api_key or os.environ.get(
                config.web_search.tavily_api_key_env, ""
            )
            web_agent = WebSearchAgent(
                tavily_api_key=tavily_key,
                enable_scholar=config.web_search.enable_scholar,
                enable_crawling=config.web_search.enable_crawling,
                enable_pdf=config.web_search.enable_pdf_extraction,
                max_web_results=config.web_search.max_web_results,
                max_scholar_results=config.web_search.max_scholar_results,
                max_crawl_urls=config.web_search.max_crawl_urls,
            )
            # Open-access links captured in candidates.jsonl (OpenAlex / S2 /
            # arXiv / CORE) feed the PDF extractor, so the full text reaches
            # stage-04/web_context.md instead of only titles and abstracts.
            pdf_targets = [
                str(c["pdf_url"]) for c in candidates if c.get("pdf_url")
            ]
            web_result = web_agent.search_and_extract(
                topic,
                search_queries=queries,
                pdf_urls=pdf_targets[:5] or None,
            )

            # Convert Google Scholar papers into candidates
            for sp in web_result.scholar_papers:
                _existing_titles = {
                    str(c.get("title", "")).lower().strip() for c in candidates
                }
                if sp.title.lower().strip() not in _existing_titles:
                    lit_paper = sp.to_literature_paper()
                    d = lit_paper.to_dict()
                    d["collected_at"] = _utcnow_iso()
                    entry, unique_key = uniquify_bibtex_entry(
                        lit_paper.to_bibtex(), bibtex_keys
                    )
                    if unique_key:
                        d["cite_key"] = unique_key
                    candidates.append(d)
                    bibtex_entries.append(entry)

            # Save web search context for downstream stages
            web_context = web_result.to_context_string(max_length=20_000)
            if web_context.strip():
                (stage_dir / "web_context.md").write_text(
                    web_context, encoding="utf-8"
                )
                web_context_parts.append(web_context)

            # Save full web search metadata
            (stage_dir / "web_search_result.json").write_text(
                json.dumps(web_result.to_dict(), indent=2, default=str),
                encoding="utf-8",
            )

            logger.info(
                "[web-search] Added %d scholar papers, %d web results, %d crawled pages",
                len(web_result.scholar_papers),
                len(web_result.web_results),
                len(web_result.crawled_pages),
            )
        except Exception:  # noqa: BLE001
            logger.warning(
                "[web-search] Web search augmentation failed — continuing with academic APIs only",
                exc_info=True,
            )

    # --- Ultimate fallback: placeholder data ---
    # BUG-L2: Do NOT overwrite real_search_succeeded here — it was already
    # set correctly in the search block above. Overwriting would mislabel
    # LLM-hallucinated or seminal papers as "real search" results.
    if not candidates:
        logger.warning("Stage 4: All literature searches failed — using placeholder papers")
        candidates = [
            {
                "id": f"candidate-{idx + 1}",
                "title": f"[Placeholder] Study {idx + 1} on {topic}",
                "source": "arxiv" if idx % 2 == 0 else "semantic_scholar",
                "url": f"https://example.org/{_safe_filename(topic.lower())}/{idx + 1}",
                "year": 2024,
                "abstract": f"This candidate investigates {topic} and reports preliminary findings.",
                "collected_at": _utcnow_iso(),
                "is_placeholder": True,
            }
            for idx in range(max(20, config.research.daily_paper_count or 20))
        ]

    # Write candidates
    out = stage_dir / "candidates.jsonl"
    _write_jsonl(out, candidates)

    # BUG-50 fix: Generate BibTeX from candidates when real search failed
    # (LLM/placeholder fallback paths don't populate bibtex_entries)
    if not bibtex_entries and candidates:
        for c in candidates:
            if c.get("is_placeholder"):
                continue
            _ck = c.get("cite_key", "")
            if not _ck:
                # Derive cite_key from first author surname + year
                _authors = c.get("authors", [])
                _surname = "unknown"
                if isinstance(_authors, list) and _authors:
                    _a0 = _authors[0] if isinstance(_authors[0], str) else (_authors[0].get("name", "") if isinstance(_authors[0], dict) else "")
                    _surname = _a0.split()[-1].lower() if _a0.strip() else "unknown"
                _yr = c.get("year", 2024)
                _title_word = "".join(
                    w[0] for w in str(c.get("title", "study")).split()[:3]
                ).lower()
                _ck = f"{_surname}{_yr}{_title_word}"
            # Same key must be advertised downstream and written here.
            _ck, _ = uniquify_bibtex_entry(f"@article{{{_ck}, x=1}}", bibtex_keys)
            _ck = _ck.split("{", 1)[1].split(",", 1)[0]
            c["cite_key"] = _ck
            _title = c.get("title", "Untitled")
            _year = c.get("year", 2024)
            _author_str = ""
            _raw_authors = c.get("authors", [])
            if isinstance(_raw_authors, list):
                _names = []
                for _a in _raw_authors:
                    if isinstance(_a, str):
                        _names.append(_a)
                    elif isinstance(_a, dict):
                        _names.append(_a.get("name", ""))
                _author_str = " and ".join(n for n in _names if n)
            bibtex_entries.append(
                f"@article{{{_ck},\n"
                f"  title={{{_title}}},\n"
                f"  author={{{_author_str or 'Unknown'}}},\n"
                f"  year={{{_year}}},\n"
                f"  url={{{c.get('url', '')}}},\n"
                f"}}"
            )
        logger.info(
            "Stage 4: Generated %d BibTeX entries from candidates (fallback)",
            len(bibtex_entries),
        )

    # Write references.bib (F2.4)
    artifacts = ["candidates.jsonl"]
    if web_context_parts:
        artifacts.append("web_context.md")
    if (stage_dir / "web_search_result.json").exists():
        artifacts.append("web_search_result.json")
    if bibtex_entries:
        # Duplicate keys are fatal in BibTeX (the second entry shadows the
        # first), and the stage assembles entries from several paths, so the
        # dedup runs on the rendered text right before the write.
        from researchclaw.literature.search import dedupe_bibtex_entries

        bib_content = "\n\n".join(dedupe_bibtex_entries(bibtex_entries)) + "\n"
        (stage_dir / "references.bib").write_text(bib_content, encoding="utf-8")
        artifacts.append("references.bib")
        logger.info(
            "Stage 4: Wrote %d BibTeX entries to references.bib", len(bibtex_entries)
        )

    # Write search metadata
    (stage_dir / "search_meta.json").write_text(
        json.dumps(
            {
                "real_search": real_search_succeeded,
                "queries_used": queries,
                "year_min": year_min,
                "year_filter": {
                    "min_year": year_min,
                    "active": year_min > 0,
                    "hybrid": hybrid_enabled,
                    "hint": (
                        f"the primary pass excludes papers before {year_min}; "
                        "the hybrid pass adds older foundations "
                        f"(cap {int(getattr(config.research, 'foundational_max_papers', 40))})"
                        if hybrid_enabled
                        else (
                            f"papers published before {year_min} are excluded by "
                            "every source; set research.min_year: 0 to include them"
                            if year_min > 0
                            else "no publication-year cutoff"
                        )
                    ),
                },
                "hybrid_search": {
                    "enabled": hybrid_enabled,
                    "foundational_added": len(foundational_papers),
                    "foundational_max_papers": int(
                        getattr(config.research, "foundational_max_papers", 40)
                    ),
                },
                "cache": {
                    "dir": cache_state.get("cache_dir", ""),
                    "previous_policy": cache_state.get("previous"),
                    "legacy_entries": cache_state.get("legacy_entries", False),
                },
                "total_candidates": len(candidates),
                "bibtex_entries": len(bibtex_entries),
                "ts": _utcnow_iso(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    artifacts.append("search_meta.json")

    return StageResult(
        stage=Stage.LITERATURE_COLLECT,
        status=StageStatus.DONE,
        artifacts=tuple(artifacts),
        evidence_refs=tuple(f"stage-04/{a}" for a in artifacts),
    )


# Screening budget defaults live in LiteratureSearchConfig
# (``max_abstract_chars`` / ``max_screen_chars`` / ``min_shortlist``); they are
# read below so a large-context model can be fed hundreds of candidates.


# Provenance that must survive the LLM screen. The screening prompt asks the
# model to echo rows, but models drop fields they consider decorative, and a
# dropped ``is_preprint``/``concern``/``pub_types`` disables every evidence
# marker in ``review.md`` and in the synthesis prompt.
_EVIDENCE_FIELDS = (
    "paper_id",
    "doi",
    "arxiv_id",
    "url",
    "pdf_url",
    "year",
    "venue",
    "source",
    "pmid",
    "is_preprint",
    "pub_types",
    "retracted",
    "concern",
)


def _row_keys(row: dict[str, Any]) -> list[str]:
    """Identifiers a shortlist row can be matched back on."""
    keys = []
    for field in ("paper_id", "doi", "cite_key", "arxiv_id"):
        value = str(row.get(field, "")).strip().lower()
        if value:
            keys.append(f"{field}:{value}")
    title = str(row.get("title", "")).strip().lower()
    if title:
        keys.append(f"title:{title}")
    return keys


def _provenance_block(row: dict[str, Any]) -> dict[str, Any]:
    """Evidence metadata a knowledge card must carry forward."""
    block: dict[str, Any] = {}
    for field in _EVIDENCE_FIELDS:
        value = row.get(field)
        if value in (None, "", [], (), False) and field != "retracted":
            continue
        if field == "pub_types" and isinstance(value, (list, tuple)):
            block[field] = list(value)
        else:
            block[field] = value
    return block


def _reapply_provenance(
    shortlist: list[dict[str, Any]], source_rows: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    """Restore evidence metadata from the pre-screen rows onto the shortlist.

    The candidate rows are the authoritative record: the model may restate a
    title but never a ``pmid`` or a preprint flag. Rows the pre-screen dropped
    as retracted are dropped here too, since the model is not told about them.
    """
    index: dict[str, dict[str, Any]] = {}
    for row in source_rows:
        for key in _row_keys(row):
            index.setdefault(key, row)
    merged: list[dict[str, Any]] = []
    dropped = 0
    for row in shortlist:
        match = None
        for key in _row_keys(row):
            match = index.get(key)
            if match is not None:
                break
        if match is not None:
            if match.get("retracted"):
                dropped += 1
                continue
            for field in _EVIDENCE_FIELDS:
                value = match.get(field)
                if value not in (None, "", [], (), False) or field == "retracted":
                    row[field] = value
        elif row.get("retracted"):
            dropped += 1
            continue
        merged.append(row)
    return merged, dropped


def _execute_literature_screen(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    candidates_text = _read_prior_artifact(run_dir, "candidates.jsonl") or ""
    screening = config.literature_search

    # --- P1-1: keyword relevance pre-filter ---
    # Before LLM screening, drop papers whose title+abstract share no keywords
    # with the research topic.  This catches cross-domain noise cheaply.
    topic_keywords = _extract_topic_keywords(
        config.research.topic, config.research.domains
    )
    filtered_rows: list[dict[str, Any]] = []
    candidates_seen = 0
    dropped_count = 0
    retracted_dropped = 0
    for raw_line in candidates_text.strip().splitlines():
        row = _safe_json_loads(raw_line, {})
        if not isinstance(row, dict):
            continue
        candidates_seen += 1
        # Stage 4 drops retracted records, but a resumed or hand-edited run can
        # still carry them; never let one reach the shortlist.
        if row.get("retracted"):
            retracted_dropped += 1
            continue
        title = str(row.get("title", "")).lower()
        abstract = str(row.get("abstract", "")).lower()
        text_blob = f"{title} {abstract}"
        overlap = sum(1 for kw in topic_keywords if kw in text_blob)
        # T2.2: Relaxed from ≥2 to ≥1 keyword hit — previous threshold was
        # too aggressive (94% rejection rate).  Single-keyword matches are
        # still screened by the LLM in the next step.
        if overlap >= 1:
            row["keyword_overlap"] = overlap
            filtered_rows.append(row)
        else:
            dropped_count += 1
    if retracted_dropped:
        logger.warning(
            "[literature] Screening skipped %d retracted paper(s)",
            retracted_dropped,
        )

    prefiltered_rows = list(filtered_rows)
    # If pre-filter dropped everything, fall back to original (safety valve)
    if not filtered_rows:
        # Safety valve: the keyword pre-filter dropped everything. Re-read the
        # raw rows, but keep the retraction filter — otherwise a retracted
        # paper that the pre-filter rejected for off-topic content comes
        # straight back and ends up in the shortlist.
        filtered_rows = [
            row
            for row in _parse_jsonl_rows(candidates_text)
            if not row.get("retracted")
        ]
    # Sort by keyword overlap (descending) before the char cap below drops
    # anything — candidates arrive in raw search order (whichever query/
    # source happened to return them), not relevance order. Without this,
    # the max_screen_chars truncation can silently drop every relevant
    # paper while keeping only whatever off-topic results came first, and
    # the LLM correctly (but uselessly) rejects the truncated set it saw.
    filtered_rows.sort(key=lambda r: r.get("keyword_overlap", 0), reverse=True)
    # Truncate abstracts and strip authors to reduce token usage
    for row in filtered_rows:
        abstract = row.get("abstract", "")
        if (
            isinstance(abstract, str)
            and len(abstract) > screening.max_abstract_chars
        ):
            row["abstract"] = abstract[: screening.max_abstract_chars] + "..."
        # Strip authors list — not needed for screening and inflates tokens.
        # MeSH descriptors are the same story: keep the term bare (``pub_types``
        # is retained, it is the evidence-level signal), because a 20-tag MeSH
        # list per paper adds ~200 chars and no decision changes.
        row.pop("authors", None)
        row.pop("mesh_terms", None)

    # Rebuild candidates_text from filtered rows
    candidates_text = "\n".join(
        json.dumps(r, ensure_ascii=False) for r in filtered_rows
    )
    # Cap total candidates text size to avoid blowing token budget
    candidates_truncated = False
    if len(candidates_text) > screening.max_screen_chars:
        candidates_truncated = True
        # Truncate at newline boundary to avoid cutting mid-JSON-line
        candidates_text = candidates_text[: screening.max_screen_chars].rsplit("\n", 1)[0]
        logger.info(
            "Candidates text truncated to %d chars for screening",
            len(candidates_text),
        )
    logger.info(
        "Domain pre-filter: kept %d, dropped %d (keywords: %s)",
        len(filtered_rows),
        dropped_count,
        topic_keywords[:8],
    )

    shortlist: list[dict[str, Any]] = []
    model_rejected_all = False
    parse_failed = False
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "literature_screen")
        sp = _pm.for_stage(
            "literature_screen",
            evolution_overlay=_overlay,
            topic=config.research.topic,
            domains=", ".join(config.research.domains)
            if config.research.domains
            else "general",
            quality_threshold=config.research.quality_threshold,
            candidates_text=candidates_text,
        )
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        raw = payload.get("shortlist") if isinstance(payload, dict) else None
        if isinstance(raw, list):
            if len(raw) == 0:
                # Distinguish a valid strict-screen empty result from a
                # malformed list whose entries are not dicts.
                model_rejected_all = True
            else:
                shortlist = [row for row in raw if isinstance(row, dict)]
                if not shortlist:
                    parse_failed = True
        else:
            parse_failed = True
    # Ensure a usable shortlist size for adequate related work (configurable)
    _MIN_SHORTLIST = screening.min_shortlist
    if model_rejected_all:
        # Strict screen returned an empty shortlist — do not backfill with
        # rejected papers. Pause the pipeline so the user can decide whether
        # to refine the search or accept the rejection.
        (stage_dir / "screen_meta.json").write_text(
            json.dumps(
                {
                    "outcome": "model_rejected_all",
                    "candidates_screened": len(filtered_rows),
                    "shortlist_size": 0,
                    "note": (
                        "Strict screen returned empty shortlist. Pipeline paused; "
                        "consider rerunning SEARCH_STRATEGY with refined queries "
                        "before resuming."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.warning(
            "Stage 5: model rejected all %d candidates — pausing pipeline",
            len(filtered_rows),
        )
        return StageResult(
            stage=Stage.LITERATURE_SCREEN,
            status=StageStatus.PAUSED,
            artifacts=("screen_meta.json",),
            error="Model returned empty shortlist after strict screening",
            evidence_refs=("stage-05/screen_meta.json",),
            decision="rejected_all",
        )
    template_fallback = not shortlist
    if not shortlist:
        rows = (
            filtered_rows[:_MIN_SHORTLIST]
            if filtered_rows
            else [
                row
                for row in _parse_jsonl_rows(candidates_text)
                if not row.get("retracted")
            ][:_MIN_SHORTLIST]
        )
        for idx, item in enumerate(rows):
            item["relevance_score"] = round(0.75 - idx * 0.02, 3)
            item["quality_score"] = round(0.72 - idx * 0.015, 3)
            item["keep_reason"] = (
                "Template fallback (parse failure)"
                if parse_failed
                else "Template screened entry"
            )
            shortlist.append(item)
    elif len(shortlist) < _MIN_SHORTLIST:
        # T2.2: LLM returned too few — supplement from filtered candidates
        existing_titles = {
            str(s.get("title", "")).lower().strip() for s in shortlist
        }
        for row in filtered_rows:
            if len(shortlist) >= _MIN_SHORTLIST:
                break
            title_lower = str(row.get("title", "")).lower().strip()
            if title_lower and title_lower not in existing_titles:
                row.setdefault("relevance_score", 0.5)
                row.setdefault("quality_score", 0.5)
                row.setdefault("keep_reason", "Supplemented to meet minimum shortlist")
                shortlist.append(row)
                existing_titles.add(title_lower)
        logger.info(
            "Stage 5: Supplemented shortlist to %d papers (minimum: %d)",
            len(shortlist), _MIN_SHORTLIST,
        )
    # The model echoes rows, so provenance comes back from the pre-screen rows
    # rather than from the model's memory of them.
    shortlist, retracted_in_shortlist = _reapply_provenance(shortlist, filtered_rows)
    if retracted_in_shortlist:
        logger.warning(
            "Stage 5: dropped %d retracted paper(s) that the screen kept",
            retracted_in_shortlist,
        )

    out = stage_dir / "shortlist.jsonl"
    _write_jsonl(out, shortlist)
    # Always written: the pre-screen and the char cap are invisible otherwise,
    # and with a 1M-context model the cap can silently discard most of the
    # corpus while the run still looks healthy.
    (stage_dir / "screen_meta.json").write_text(
        json.dumps(
            {
                "outcome": "parsed" if shortlist and not parse_failed else "fallback",
                "candidates_seen": candidates_seen,
                "candidates_after_prefilter": len(prefiltered_rows),
                "candidates_prefilter_dropped": dropped_count,
                "candidates_retracted_dropped": retracted_dropped,
                "candidates_truncated": candidates_truncated,
                "chars_sent": len(candidates_text),
                "shortlist_size": len(shortlist),
                "shortlist_retracted_dropped": retracted_in_shortlist,
                "parse_failed": parse_failed,
                "template_fallback": template_fallback,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return StageResult(
        stage=Stage.LITERATURE_SCREEN,
        status=StageStatus.DONE,
        artifacts=("shortlist.jsonl", "screen_meta.json"),
        evidence_refs=("stage-05/shortlist.jsonl", "stage-05/screen_meta.json"),
    )


def _execute_knowledge_extract(
    stage_dir: Path,
    run_dir: Path,
    config: RCConfig,
    adapters: AdapterBundle,
    *,
    llm: LLMClient | None = None,
    prompts: PromptManager | None = None,
) -> StageResult:
    shortlist = _read_prior_artifact(run_dir, "shortlist.jsonl") or ""

    # IMP-21: Defensive gate — refuse to run on an empty/missing shortlist.
    # Stage 5 (LITERATURE_SCREEN) returns PAUSED with decision="rejected_all"
    # when its strict screen rejects all candidates, in which case no
    # shortlist.jsonl is written. Without this gate, Stage 6 spends an LLM
    # turn extracting knowledge cards from empty input, produces low-quality
    # fallback cards, and lets downstream stages cascade on garbage.
    #
    # We return PAUSED (mirroring Stage 5's pattern) rather than
    # BLOCKED_APPROVAL because the runner halts on PAUSED unconditionally
    # (runner.py: ``if result.status == StageStatus.PAUSED: break``), whereas
    # BLOCKED_APPROVAL only halts when ``stop_on_gate=True`` — and
    # ``--auto-approve`` explicitly sets ``stop_on_gate=False``, which is
    # the exact scenario this gate must prevent the cascade for.
    if not _parse_jsonl_rows(shortlist):
        logger.warning(
            "Stage 6: shortlist.jsonl is empty or missing — Stage 5 likely "
            "rejected all candidates. Refusing to extract from empty input."
        )
        (stage_dir / "knowledge_meta.json").write_text(
            json.dumps(
                {
                    "outcome": "upstream_empty_shortlist",
                    "shortlist_rows": 0,
                    "note": (
                        "Stage 6 (knowledge_extract) requires a non-empty "
                        "shortlist.jsonl from Stage 5 (literature_screen). "
                        "Resolve Stage 5 (refine search queries, manually "
                        "approve a shortlist, or restart from "
                        "search_strategy) before resuming."
                    ),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        return StageResult(
            stage=Stage.KNOWLEDGE_EXTRACT,
            status=StageStatus.PAUSED,
            artifacts=("knowledge_meta.json",),
            error=(
                "Cannot extract knowledge cards: shortlist.jsonl is empty "
                "or missing. Resolve Stage 5 (literature_screen) first."
            ),
            evidence_refs=("stage-06/knowledge_meta.json",),
            decision="upstream_blocked",
        )

    # Inject web context from Stage 4 if available
    web_context = _read_prior_artifact(run_dir, "web_context.md") or ""
    if web_context:
        shortlist = shortlist + "\n\n--- Web Search Context ---\n" + web_context[:10_000]

    cards_dir = stage_dir / "cards"
    cards_dir.mkdir(parents=True, exist_ok=True)
    cards: list[dict[str, Any]] = []
    if llm is not None:
        _pm = prompts or PromptManager()
        _overlay = _get_evolution_overlay(run_dir, "knowledge_extract")
        sp = _pm.for_stage("knowledge_extract", evolution_overlay=_overlay, shortlist=shortlist)
        resp = _chat_with_prompt(
            llm,
            sp.system,
            sp.user,
            json_mode=sp.json_mode,
            max_tokens=sp.max_tokens,
        )
        payload = _safe_json_loads(resp.content, {})
        if isinstance(payload, dict) and isinstance(payload.get("cards"), list):
            cards = [item for item in payload["cards"] if isinstance(item, dict)]
    if not cards:
        rows = _parse_jsonl_rows(shortlist)
        for idx, paper in enumerate(rows[:6]):
            title = str(paper.get("title", f"Paper {idx + 1}"))
            cards.append(
                {
                    "card_id": f"card-{idx + 1}",
                    "title": title,
                    "problem": f"How to improve {config.research.topic}",
                    "method": "Template method summary",
                    "data": "Template dataset",
                    "metrics": "Template metric",
                    "findings": "Template key finding",
                    "limitations": "Template limitation",
                    "citation": str(paper.get("url", "")),
                    "cite_key": str(paper.get("cite_key", "")),
                    "provenance": _provenance_block(paper),
                }
            )
    # Provenance is attached from the shortlist row, never from the model: a
    # card is the only evidence stage 7+ sees, and without these fields a
    # preprint is indistinguishable from a randomised trial.
    rows_by_key: dict[str, dict[str, Any]] = {}
    for paper in _parse_jsonl_rows(shortlist):
        for key in _row_keys(paper):
            rows_by_key.setdefault(key, paper)

    for idx, card in enumerate(cards):
        card_id = _safe_filename(str(card.get("card_id", f"card-{idx + 1}")))
        if not card.get("provenance"):
            for key in _row_keys(card):
                row = rows_by_key.get(key)
                if row is not None:
                    card["provenance"] = _provenance_block(row)
                    break
        parts = [f"# {card.get('title', card_id)}", ""]
        for key in (
            "cite_key",
            "problem",
            "method",
            "data",
            "metrics",
            "findings",
            "limitations",
            "citation",
        ):
            parts.append(f"## {key.title()}")
            value = card.get(key, "")
            # Models emit JSON null for "nothing here"; ``str(None)`` put the
            # literal word "None" into every such card section.
            parts.append("" if value is None else str(value))
            parts.append("")
        provenance = card.get("provenance")
        if isinstance(provenance, dict) and provenance:
            parts.append("## Provenance")
            for field, value in provenance.items():
                if value not in (None, "", [], (), False) or field == "retracted":
                    parts.append(f"- {field}: {value}")
            parts.append("")
        (cards_dir / f"{card_id}.md").write_text("\n".join(parts), encoding="utf-8")
    return StageResult(
        stage=Stage.KNOWLEDGE_EXTRACT,
        status=StageStatus.DONE,
        artifacts=("cards/",),
        evidence_refs=("stage-06/cards/",),
    )
