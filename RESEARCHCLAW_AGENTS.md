# AutoResearchClaw — Agent Configuration

## Overview

ResearchClaw is an autonomous research pipeline that takes a research topic and produces a complete academic paper through 23 automated stages. This file defines how AI agents should bootstrap and interact with the system.

## Agent Role: Research Orchestrator

You are an AI research assistant operating ResearchClaw. Your job is to:

1. **Understand the user's research interest** — clarify the topic, scope, and constraints
2. **Configure the pipeline** — set up `config.yaml` with appropriate LLM settings and experiment mode
3. **Execute the pipeline** — run the 23-stage pipeline via CLI or Python API
4. **Monitor and intervene** — handle gate stages (5, 9, 20), review intermediate outputs
5. **Deliver results** — present the final paper, charts, and experiment data to the user

## Quick Setup

```bash
# Install
pip install -e .

# Configure (copy and edit)
cp config.researchclaw.example.yaml config.yaml
# Set llm.base_url, llm.api_key, experiment.mode

# Run
researchclaw run --topic "Your topic" --auto-approve
```

## Pipeline Stages (23 stages, 8 phases)

| Phase | Stages | Description |
|-------|--------|-------------|
| A: Research Scoping | 1-2 | Define topic, decompose into sub-problems |
| B: Literature Discovery | 3-6 | Search strategy, collect papers, screen [GATE@5], extract knowledge |
| C: Knowledge Synthesis | 7-8 | Cluster topics, generate hypotheses |
| D: Experiment Design | 9-11 | Design experiments [GATE@9], generate code, plan resources |
| E: Experiment Execution | 12-13 | Run experiments, iterative refinement |
| F: Analysis & Decision | 14-15 | Analyze results, decide proceed/pivot/iterate |
| G: Paper Writing | 16-19 | Outline, draft, peer review, revision |
| H: Finalization | 20-23 | Quality gate [GATE@20], archive, export with charts, citation verification |

## Gate Stages

Three stages require approval (use `--auto-approve` for fully autonomous mode):
- **Stage 5** (Literature Screen): Validates collected literature quality
- **Stage 9** (Experiment Design): Validates experiment protocol before code generation
- **Stage 20** (Quality Gate): Validates overall paper quality before export

## Experiment Modes

- `simulated`: LLM generates synthetic results (fast, no code execution)
- `sandbox`: Execute generated code locally (requires Python environment)
- `ssh_remote`: Execute on remote GPU server (requires SSH configuration)

## Key Files

| File | Purpose |
|------|---------|
| `config.yaml` | Pipeline configuration (LLM, experiment mode, etc.) |
| `config.researchclaw.example.yaml` | Configuration template |
| `researchclaw/cli.py` | CLI entry point |
| `researchclaw/pipeline/executor.py` | Stage execution logic |
| `researchclaw/pipeline/runner.py` | Pipeline orchestration |
| `researchclaw/experiment/validator.py` | Code validation (AST, security, imports) |
| `researchclaw/experiment/visualize.py` | Chart generation |

## Decision Guide

| Situation | Action |
|-----------|--------|
| User provides a clear topic | Run full pipeline with `--auto-approve` |
| User wants to review stages | Run without `--auto-approve`, pause at gates |
| Pipeline fails at a stage | Check error, fix config or retry from that stage with `--from-stage` |
| User wants only a literature review | `researchclaw review -c config.yaml -t "topic"` (Stages 1–7, writes `review.md`) |
| User wants iteration | Use `execute_iterative_pipeline()` with `max_iterations` |
| Experiment code fails | Validator auto-retries up to 3 times; if still failing, switch to `simulated` mode |

## Config Knobs Agents Commonly Need

Run `researchclaw info -c config.yaml` to see the resolved values before guessing.

| Knob | What it does |
|---|---|
| `llm.extra_headers` | Fixed HTTP headers on every LLM request **and** on `researchclaw doctor` probes. `auto` → fresh `researchclaw-<hex>` session id per client. Required by gateways that route on headers: OpenCode Go needs `x-opencode-session: "auto"`, otherwise every call is `400 MissingSessionID` and doctor misreports `api_key_valid: 403` |
| `literature_search.sources` | Any subset of `openalex`, `semantic_scholar`, `arxiv`, `openreview`, `crossref`, `pubmed`, `core`; called in the listed order, unknown names skipped, `s2` is an alias for `semantic_scholar` |
| `literature_search.*_api_key` / `*_env` | Source credentials. Inline wins over env. `CORE_API_KEY` is **required** for `core`; `S2_API_KEY` avoids unauthenticated 429s; `NCBI_API_KEY` raises PubMed 3 → 10 req/s; OpenAlex works keyless from a shared per-IP pool |
| `literature_search.pubmed_domain_gate` | `true` → PubMed is dropped unless `research.domains` looks biomedical (word-level query translation floods ML topics with clinical hits) |
| `web_search.enable_pdf_extraction` | Passes open-access `pdf_url` links from Stage 4 to the PDF extractor; text lands in `stage-04/web_context.md` (needs `pip install "researchclaw[pdf]"`) |
| `experiment.sandbox.allowed_imports` | **Documentation only** — not enforced anywhere; isolation is the subprocess plus the timeout |

## Integration Platforms

ResearchClaw works with:
- **Claude Code**: Load via `.agents/skills/researchclaw/SKILL.md`
- **pi**: Auto-loads `AGENTS.md` at the repo root; can also serve as the LLM backend (`llm.provider: acp`, `llm.acp.agent: "pi"`, needs `acpx` on PATH)
- **OpenClaw**: Read this `AGENTS.md` + `README.md` for bootstrapping
- **OpenCode**: Compatible skill format in `.agents/skills/`
- **Standalone**: Direct CLI or Python API usage
