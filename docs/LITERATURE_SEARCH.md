# Literature search (Stages 3–6)

How a topic becomes an evidence base: seven academic backends, a publication-year
policy with a second pass for older foundations, a configurable screening budget,
and a year-policy-aware cache.

- **Configure it:** `research.*` and `literature_search.*` in
  `config.researchclaw.example.yaml` (the file is heavily commented).
- **Defaults:** `ResearchConfig` / `LiteratureSearchConfig` in
  `researchclaw/config.py`.
- **Implementation:** `researchclaw/literature/` (clients + cache) and
  `researchclaw/pipeline/stage_impls/_literature.py` (Stages 3–6).

**Literature sources and their keys.** Stage 4 queries any subset of seven
backends; list them in `literature_search.sources` (order matters, unknown
names are skipped). Keys go inline (`*_api_key`) or into the env var named by
`*_env` — inline wins.

| Source | What it adds | Key |
|---|---|---|
| `openalex` | metadata + citation counts; cheap and broad | `OPENALEX_API_KEY` optional (metered); without it a shared per-IP pool |
| `semantic_scholar` | citation counts, strongest ML coverage | `S2_API_KEY` optional, recommended — unauthenticated S2 trips its 429 circuit breaker |
| `arxiv` | preprints | none |
| `openreview` | ICLR/NeurIPS submissions with PDFs | none |
| `crossref` | publisher metadata (ACM/IEEE/Springer/Elsevier) + DOI | none; `openalex_email` is reused as the polite contact |
| `pubmed` | MEDLINE | `NCBI_API_KEY` optional — raises 3 → 10 req/s |
| `core` | open-access full text | `CORE_API_KEY` **required** (free at <https://core.ac.uk/services/api>) |

### How Stages 4–6 build the evidence base

1. **Queries.** Stage 3 writes up to 8 queries to `stage-03/queries.json`; Stage 4 adds up to 5 broader variants.
2. **Primary pass.** For each query × source: up to `literature_search.max_results_per_query` (40) papers, published in `research.min_year` or later.
3. **Hybrid pass** (on by default). The same base queries run again with no year floor, 10 results per source (`research.foundational_results_per_query`). Only papers older than the floor are kept, deduplicated against the primary pass, sorted by citations and capped at `research.foundational_max_papers` (40).
4. **Merge.** Everything is deduplicated by DOI → arXiv ID → title and written to `stage-04/candidates.jsonl` (+ `references.bib`), each row tagged `year_band`: `recent`, `foundational`, or `all` when there is no floor.
5. **Screen** (gate). Stage 5 keeps rows that share ≥1 topic keyword, sorts by overlap, truncates abstracts and caps the prompt at `max_screen_chars`; the LLM returns a shortlist (target `min_shortlist`, backfilled if shorter). Foundations are kept only when they define the problem, a metric or a baseline.
6. **Extract.** Stage 6 turns the shortlist into knowledge cards; synthesis and hypotheses see only those cards.

### Year policy

| Knob | Default | Meaning |
|---|---|---|
| `research.min_year` | 2020 | Floor for the primary pass. `0` = no floor (the hybrid pass then turns itself off) |
| `research.hybrid_search` | true | Second, unfiltered pass recovering older foundations. Set `false` to disable |
| `research.foundational_results_per_query` | 10 | Results per source per query in that pass |
| `research.foundational_max_papers` | 40 | Cap on foundations added per run (highest-cited first) |

Why it exists: the year floor changes what each backend returns, so a filtered
ranking pushes classics out entirely (Transformer 2017, Adam 2014, ResNet 2015).
One extra pass (~25–50% more Stage-4 wall time) brings them back without
displacing recent work.

Verify a run: `grep -c '"year_band": "foundational"' artifacts/<run>/stage-04/candidates.jsonl`
and see `year_filter` / `hybrid_search` in `stage-04/search_meta.json`.
`researchclaw doctor` warns while a floor is active.

### Screening budget

Stage 4 may collect 600–900 unique papers; the screener only sees what fits.
`max_screen_chars` caps the blob, `max_abstract_chars` truncates each abstract
first, `min_shortlist` is the target shortlist. One candidate line measures
~1 340 chars with an 800-char abstract, ~1 740 with 1 200, so capacity ≈
chars / line size (tokens ≈ chars/4):

| Context window | `max_screen_chars` | `max_abstract_chars` | Papers seen | Tokens |
|---|---|---|---|---|
| 128k | 100 000 | 800 | ~74 | ~25k |
| 200k | 200 000 | 1 000 | ~130 | ~50k |
| 1M (example config) | 600 000 | 1 200 | ~345 | ~150k |

The example config ships the 1M preset (600 000 / 1 200 / 25) and
`researchclaw init` copies it. Omitting the keys falls back to the code defaults
(100 000 / 800 / 15), which fit any 128k model. Longer abstracts buy evidence
per paper but cost slots (600k holds ~448 papers at 800 chars vs ~345 at
1 200); past ~400–500 papers in one call, batch instead of enlarging the prompt.

### Literature cache

Search results are cached in `.researchclaw_cache/literature/` with per-source
TTLs (24 h arXiv, 3 days OpenAlex/S2, 7 days the rest). The cache key includes
`min_year`, so results fetched under one year policy are never served under
another, and the directory records the policy that filled it (`.policy.json`).
Stage 4 warns when the policy changed or when entries predate this tracking, and
`researchclaw doctor` reports it as `literature_cache`; both are warnings, so
`Result:` stays `PASS`. Clear it with `rm -rf .researchclaw_cache/literature`
(it is gitignored and holds only search results).

**PubMed and open-access PDFs.** `pubmed_domain_gate: true` drops PubMed unless
`research.domains` looks biomedical — its word-level query translation floods an
ML topic with irrelevant clinical hits. Open-access `pdf_url` links captured by
any source are passed to the PDF extractor when
`web_search.enable_pdf_extraction` is on (pip extra `pdf`); the extracted text
lands in `stage-04/web_context.md`.

## Where to change what

| Want | Knob | Code default | Example config |
|---|---|---|---|
| Search more/fewer results per query | `literature_search.max_results_per_query` | 40 | 40 |
| Add or reorder backends | `literature_search.sources` | openalex, semantic_scholar, arxiv | same |
| Include old papers | `research.min_year` | 2020 (`0` = no floor) | 2020 |
| Recover foundations without losing recent work | `research.hybrid_search` + `foundational_*` | on, 10/source, cap 40 | same |
| Show the screener more papers | `literature_search.max_screen_chars` / `max_abstract_chars` | 100 000 / 800 | 600 000 / 1 200 (1M preset) |
| Bigger shortlist | `literature_search.min_shortlist` | 15 | 25 |
| Clear the search cache | — | `rm -rf .researchclaw_cache/literature` | — |

Checks after a run: `stage-04/search_meta.json` (`year_filter`,
`hybrid_search`, `cache`), `stage-04/candidates.jsonl` (`year_band`), and
`researchclaw doctor` (`min_year`, `literature_cache` warnings).
