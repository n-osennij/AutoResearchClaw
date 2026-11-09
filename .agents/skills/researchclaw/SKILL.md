---
name: researchclaw
description: >
  Drive the ResearchClaw 23-stage autonomous research pipeline on this machine:
  configure and run it (topic -> paper), supervise gate stages, inspect or resume
  runs, read the paper, charts and verification report. Use when asked to run or
  resume research, write a paper on a topic, check a run's progress, or fix a
  failed stage.
metadata:
  category: orchestration
  trigger-keywords: "researchclaw,run pipeline,research topic,paper,gate,resume,artifacts"
---

# ResearchClaw — driving the pipeline

ResearchClaw turns a research topic into a paper through 23 stages: literature →
hypotheses → experiment code and runs → analysis → paper → review → LaTeX export.
It is a normal CLI + filesystem project: run it with bash, read `artifacts/`.
Nothing here needs OpenClaw.

`python -m researchclaw ...` works from the repo root; after `pip install -e .`
the shorter `researchclaw ...` entry point is available too (used below).

## 1. Preflight

```bash
ls config.arc.yaml config.yaml 2>/dev/null || researchclaw init   # writes config.arc.yaml
researchclaw validate -c config.arc.yaml      # "Config validation passed"
researchclaw doctor   -c config.arc.yaml      # "Result: PASS" = ready
```

`doctor` checks the LLM endpoint (or the ACP agent **and** `acpx`), the sandbox
interpreter and matplotlib. Fix whatever it flags before running.

## 2. Configure

Everything is one YAML file; `config.arc.yaml` / `config.yaml` are gitignored.
The knobs that actually matter:

| Knob | Notes |
|---|---|
| `research.topic` / `research.domains` | topic text; domains drive search and domain detection |
| `research.min_year` | publication-year floor, default **2020**. `0` = no cutoff (needed for pre-2020 foundations) |
| `llm.provider` + `base_url` + `api_key_env` + `primary_model` | OpenAI-compatible. Presets: openai, openrouter, deepseek, atlascloud, anthropic, kimi-anthropic, minimax*, novita, ollama |
| `llm.extra_headers` | Headers on every request and doctor probe. `auto` → per-client session id |
| `llm.reviewer_model` / `fallback_models` | optional second opinion and fallback chain |
| `literature_search.sources` | subset of openalex, semantic_scholar, arxiv, openreview, crossref, pubmed, core |
| `literature_search.max_results_per_query` | **40** results per source per query (default) |
| `literature_search.*_api_key` / `*_env` | `CORE_API_KEY` required for `core`; S2/NCBI optional |
| `experiment.mode` | `sandbox` (real subprocess), `simulated` (fake numbers), `docker`, `ssh_remote`, `colab_drive`, `agentic`, `collider_agent`, `biology_agent`, `stat_agent` |
| `experiment.sandbox.python_path` | interpreter for generated code — on Windows `...\Scripts\python.exe` |
| `experiment.time_budget_sec` | real wall clock per experiment in sandbox mode; too small trims the plan (Stage 9 warns) |
| `security.hitl_required_stages` | gates, default `[5, 9, 20]` |

### OpenCode Go backend (verified)

pi's `opencode-go` credentials are an API key, so the pipeline can call the same
endpoint directly:

```bash
export OPENCODE_GO_API_KEY="$(pi auth print-api-key --provider opencode-go)"
```

```yaml
llm:
  provider: "openai-compatible"
  base_url: "https://opencode.ai/inference/go/v1"   # /inference/go, not /inference/openai
  api_key_env: "OPENCODE_GO_API_KEY"
  extra_headers:
    x-opencode-session: "auto"                     # gateway rejects requests without it
  primary_model: "deepseek-v4.1-flash"
  fallback_models: ["glm-5.3-flash"]
```

Without `x-opencode-session` every call is `400 MissingSessionID` and doctor
reports a misleading `api_key_valid: 403`.

## 3. Run

```bash
# unattended, from scratch
researchclaw run -c config.arc.yaml -t "topic" --auto-approve

# cheap slice first (literature + hypotheses only)
researchclaw run -c config.arc.yaml --to-stage HYPOTHESIS_GEN --auto-approve

# resume the newest run for the same topic
researchclaw run -c config.arc.yaml --resume --auto-approve
```

Long runs: start with `nohup`/background and poll
`artifacts/<run>/heartbeat.json` (`last_stage`) — the log carries warnings only.
Expect 1.5–3 h for stages 1–23 with API models.

## 4. Gates and supervision

Gates at 5, 9, 20. Unattended runs need `--auto-approve`; otherwise the stage
returns `BLOCKED_APPROVAL` and the pipeline stops. To supervise:

```bash
researchclaw status artifacts/<run>              # stages + current gate
researchclaw approve artifacts/<run> -m "ok"
researchclaw reject  artifacts/<run> -r "why"
researchclaw guide --stage 9 -m "add an ablation" artifacts/<run>
```

`attach` is a TUI — prefer the subcommands from an agent.

## 5. Read the results

```
artifacts/<run>/
├── checkpoint.json / heartbeat.json   # resume state, last completed stage
├── stage-NN/                          # per-stage artifacts + stage_health.json
├── stage-03/queries.json              # search queries + year_min
├── stage-04/candidates.jsonl          # collected literature (pdf_url filled)
├── stage-04/search_meta.json          # per-source counts, cache hits
├── stage-05/screen_meta.json          # candidates screened, shortlist size
├── stage-05/shortlist.jsonl           # papers kept for extraction
├── stage-06/cards/                    # knowledge cards (one per shortlisted paper)
├── stage-12/runs/…                    # generated code + per-run results.json
├── stage-12/experiment_history.json   # metric trajectory
├── stage-13/experiment_final/         # refined code + final results
├── stage-14/experiment_summary.json   # aggregated metrics
├── stage-14/analysis.md, charts/      # analysis + figures
├── stage-17/paper_draft.md            # paper markdown
├── stage-18/reviews.md                # peer review
├── stage-22/paper.tex, charts/        # LaTeX + figures referenced by the paper
├── stage-23/verification_report.json  # citation integrity + relevance
└── deliverables/                      # compile-ready bundle for Overleaf
```

`researchclaw report --run-dir artifacts/<run>` renders a human summary.
If a results table in the paper is empty, check `stage-12/stage_health.json`
and `stage-14/experiment_summary.json` first: the anti-fabrication layer strips
numbers it cannot trace back to a run, so a failed experiment leaves no numbers
to cite.

## 6. Limits worth knowing

- **Per query per source:** `literature_search.max_results_per_query` (40).
  Stage 3 yields up to 8 queries, Stage 4 adds up to 5 broader variants, so a
  3-source run collects roughly 960–1560 candidates before dedup.
- **Screening:** candidates are pre-filtered by keyword overlap, sorted by
  overlap, abstracts cut to 800 chars, and the whole blob for the LLM is capped
  at 100 000 chars — the model does not see every candidate on huge runs.
- **Shortlist:** the target is 15 papers (backfilled from the prefiltered set if
  the model returns fewer; an empty model verdict pauses the stage).
- **Knowledge cards:** Stage 6 asks for one card per shortlisted paper; the
  template fallback (used when the model returns nothing) takes 6.
- **Year:** `research.min_year`, default 2020 (see above). Only 31 curated
  classics (`researchclaw/data/seminal_papers.yaml`) are injected regardless.
  The literature cache key is `(query, source, limit)` — **`min_year` is not
  part of it**, so after lowering the cutoff old cached hits can still come
  back until the entry expires (24 h arXiv, 3 days OpenAlex/S2, 7 days the
  rest). Delete `.researchclaw_cache/literature/` to force fresh searches.
- **Web layer:** 10 web results, 10 scholar results, 5 crawled pages, up to 5
  PDFs per run.

## 7. MCP server

```bash
researchclaw mcp --start -c config.arc.yaml        # stdio, newline-delimited JSON-RPC
```

Six working tools: `run_pipeline`, `get_pipeline_status`,
`get_experiment_results`, `search_literature`, `review_paper`, `get_paper`.
`run_pipeline` launches the real pipeline in a background thread and returns a
`run_id`; the status tools also read runs started from the CLI.

Register it in pi once (cwd must be the repo so `python -m researchclaw`
resolves), then verify:

```bash
pi mcp add researchclaw --cwd "$PWD" -- python -m researchclaw mcp --start -c config.arc.yaml
pi mcp list        # researchclaw: connected, 6 tools
pi mcp remove researchclaw
```

stdout is the protocol stream — the server logs to stderr, so never wrap it in
something that prints to stdout. Only the stdio transport exists.

## 8. pi as the LLM backend (optional)

```yaml
llm:
  provider: "acp"
  acp: {agent: "pi", cwd: ".", max_turns: 1, timeout_sec: 1800}
```

`acpx` must be on PATH (`npm install -g acpx`); `doctor` checks it. Caveat: the
reviewer and debate panel reuse the same agent, and all 23 stages share one long
session. For real work keep the pipeline on API models and let pi supervise.

## Troubleshooting

| Symptom | Action |
|---|---|
| `Missing required field` | `researchclaw validate -c ...` lists every missing key |
| `400 MissingSessionID` / `api_key_valid: 403` | set `llm.extra_headers: {x-opencode-session: auto}` |
| stage 4 too few papers | add sources/keys, lower `min_year`; inspect `stage-04/search_meta.json` |
| `Stage 9: Plan has N conditions (limit …)` | raise `experiment.time_budget_sec` |
| experiment produced nothing | `stage-12/stage_health.json`, then rerun `--from-stage EXPERIMENT_RUN` |
| gate blocked the run | `--auto-approve` or `approve` the run directory |
| citations stripped | `stage-23/verification_report.json` says which and why |
