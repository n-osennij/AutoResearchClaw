---
name: researchclaw
description: >
  Drive the ResearchClaw 23-stage autonomous research pipeline on this machine:
  configure and run it (topic -> paper), supervise gate stages, inspect or resume
  runs, read the paper/charts/verification report. Use when asked to run or
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

`python -m researchclaw ...` always works from the repo root; the `researchclaw`
entry point exists only after `pip install -e .`.

## 1. Preflight

```bash
ls config.arc.yaml config.yaml 2>/dev/null || python -m researchclaw init   # writes config.arc.yaml
python -m researchclaw validate -c config.arc.yaml      # "Config validation passed"
python -m researchclaw doctor   -c config.arc.yaml      # "Result: PASS" = ready
```

`doctor` checks the LLM endpoint (or the ACP agent + acpx), the sandbox
interpreter and matplotlib. Fix whatever it flags before running.

## 2. Configure

Everything is one YAML file; `config.arc.yaml` / `config.yaml` are gitignored.
The knobs that actually matter:

| Knob | Notes |
|---|---|
| `research.topic` / `research.domains` | topic text; domains drive search and domain detection |
| `llm.provider` + `base_url` + `api_key_env` + `primary_model` | OpenAI-compatible. Presets: openai, openrouter, deepseek, atlascloud, anthropic, kimi-anthropic, minimax*, novita, ollama |
| `llm.extra_headers` | Headers on every request and doctor probe. `auto` → per-client session id |
| `llm.reviewer_model` / `fallback_models` | optional second opinion and fallback chain |
| `literature_search.sources` | subset of openalex, semantic_scholar, arxiv, openreview, crossref, pubmed, core |
| `literature_search.*_api_key` / `*_env` | `CORE_API_KEY` required for `core`; S2/NCBI optional |
| `experiment.mode` | `sandbox` (real subprocess), `simulated` (fake numbers), `docker`, `ssh_remote`, `collider_agent`, `biology_agent`, `stat_agent` |
| `experiment.sandbox.python_path` | interpreter for generated code — on Windows `...\Scripts\python.exe` |
| `experiment.time_budget_sec` | real wall clock per experiment in sandbox mode |
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
python -m researchclaw run -c config.arc.yaml -t "topic" --auto-approve

# cheap slice first (literature + hypotheses only)
python -m researchclaw run -c config.arc.yaml --to-stage HYPOTHESIS_GEN --auto-approve

# resume the newest run for the same topic
python -m researchclaw run -c config.arc.yaml --resume --auto-approve
```

Long runs: start with `nohup`/background and poll
`artifacts/<run>/heartbeat.json` (`last_stage`) — the log carries warnings only.
Budget 1.5–3 h for stages 1–23.

## 4. Gates and supervision

Gates at 5, 9, 20. Unattended runs need `--auto-approve`; otherwise the stage
returns `BLOCKED_APPROVAL` and the pipeline stops. To supervise:

```bash
python -m researchclaw status artifacts/<run>              # stages + current gate
python -m researchclaw approve artifacts/<run> -m "ok"
python -m researchclaw reject  artifacts/<run> -r "why"
python -m researchclaw guide --stage 9 -m "add an ablation"
```

`attach` is a TUI — prefer the subcommands from an agent.

## 5. Read the results

```
artifacts/<run>/
├── checkpoint.json / heartbeat.json        # resume state, last completed stage
├── stage-01..23/                           # per-stage artifacts + stage_health.json
├── stage-04/candidates.jsonl               # collected literature (pdf_url filled)
├── stage-12/runs/…, experiment_final/      # generated code + metrics
├── stage-14/experiment_summary.json        # aggregated metrics
├── stage-17/paper_draft.md                 # paper markdown
├── stage-22/paper.tex, charts/             # LaTeX + figures
└── deliverables/                           # compile-ready bundle (Overleaf)
```

`python -m researchclaw report artifacts/<run>` renders a human summary.
Citation verification lives in `verification_report.json`; fabricated numbers are
stripped from the paper by the VerifiedRegistry, so an empty table usually means
the experiment stage failed — check `stage-12/*/stage_health.json`.

## 6. MCP server (optional)

```bash
python -m researchclaw mcp --start -c config.arc.yaml
```

Speaks JSON-RPC 2.0 over stdio with newline-delimited framing (MCP spec) and
exposes `run_pipeline`, `get_pipeline_status`, `get_experiment_results`,
`search_literature`, `review_paper`, `get_paper`. Register it exactly once:

```bash
pi mcp add researchclaw -- python -m researchclaw mcp --start -c config.arc.yaml
```

Do not pass `-p`/`--print` style output flags: stdout is the protocol stream (the
server logs to stderr).

## 7. pi as the LLM backend (optional)

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
| `Missing required field` | `python -m researchclaw validate -c ...` lists every missing key |
| `400 MissingSessionID` / `api_key_valid: 403` | set `llm.extra_headers: {x-opencode-session: auto}` |
| stage 4 too few papers | add sources / keys; check `stage-04/search_meta.json` |
| `Stage 9: Plan has N conditions (limit 8 …)` | raise `experiment.time_budget_sec` |
| experiment produced nothing | `stage-12/*/stage_health.json`, then rerun `--from-stage EXPERIMENT_RUN` |
| gate blocked the run | `--auto-approve` or `approve` the run directory |
