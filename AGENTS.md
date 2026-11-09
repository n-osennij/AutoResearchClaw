# AutoResearchClaw — working notes for coding agents

ResearchClaw (package `researchclaw`, v0.5.0, MIT) is a 23-stage autonomous
research pipeline: a topic goes in, a conference paper comes out — real
literature from seven backends, sandboxed experiments, multi-agent review,
LaTeX export. Nothing here is OpenClaw-specific; any agent that can run a
shell command can drive it.

Deeper references, read on demand:

| File | What it holds |
|---|---|
| `RESEARCHCLAW_AGENTS.md` | the upstream agent guide: role, phases, decision table |
| `RESEARCHCLAW_CLAUDE.md` | compact architecture brief |
| `README.md` → ⚙️ Configuration Reference | every config key, provider presets, ACP table |
| `docs/integration-guide.md` | human walkthrough: setup, stages, artifacts, troubleshooting |

## First commands

```bash
pip install -e .                                  # Python 3.11+
researchclaw init                                 # writes config.arc.yaml from the example
researchclaw validate -c config.arc.yaml
researchclaw doctor   -c config.arc.yaml          # PASS = ready to run
researchclaw run      -c config.arc.yaml --topic "..." --auto-approve
```

On Windows: set `experiment.sandbox.python_path` to the venv's
`Scripts\python.exe` (the example ships the POSIX path) and export
`PYTHONUNBUFFERED=1` when redirecting output. For long unattended runs:

```bash
PYTHONUNBUFFERED=1 nohup researchclaw run -c config.arc.yaml --auto-approve \
  > artifacts/run.log 2>&1 &
```

Progress is visible only through `artifacts/<run>/heartbeat.json` (`last_stage`)
and the per-stage `stage-NN/` directories — the log carries warnings, not a
stage trace. `--resume` continues the newest run for the same topic hash.

## Knobs a new agent actually needs

Everything is one YAML file; `config.yaml` overrides the example.

| Knob | Notes |
|---|---|
| `llm.provider` / `base_url` / `api_key_env` / `primary_model` | Any OpenAI-compatible endpoint. Presets: `openai`, `openrouter`, `deepseek`, `atlascloud`, `anthropic`, `kimi-anthropic`, `minimax*`, `novita`, `ollama`, `openai-compatible` |
| `llm.extra_headers` | Fixed headers on every request **and** on doctor probes. `auto` → a fresh `researchclaw-<hex>` id per LLM client. Required by gateways that route on headers (OpenCode Go: `x-opencode-session`, otherwise `400 MissingSessionID`) |
| `llm.provider: acp` + `llm.acp.agent` | Any coding CLI becomes the LLM backend through `acpx` (`claude`, `codex`, `gh`, `opencode`, `pi`). No API key needed — the agent uses its own. Reviewer and debate panel reuse the same agent |
| `literature_search.sources` | Any subset of `openalex`, `semantic_scholar`, `arxiv`, `openreview`, `crossref`, `pubmed`, `core`; order matters, unknown names are skipped |
| `literature_search.*_api_key` / `*_env` | Inline wins over env. `CORE_API_KEY` is required for `core`; `S2_API_KEY` and `NCBI_API_KEY` are optional improvements; OpenAlex works keyless from a shared pool |
| `literature_search.pubmed_domain_gate` | `true` → drop PubMed unless `research.domains` looks biomedical |
| `experiment.mode` | `simulated` (fake numbers), `sandbox` (real subprocess), `docker`, `ssh_remote`, `colab_drive`, plus domain agents `collider_agent` / `biology_agent` / `stat_agent` |
| `experiment.sandbox.python_path` | Interpreter that runs generated experiment code |
| `security.hitl_required_stages` | Gate stages (default 5, 9, 20) |
| `prompts.extra_prompts` | Per-stage extra instructions — the supported way to steer a stage without forking |

## Gates and supervision

Stages 5, 9, 20 are gates. Unattended runs need `--auto-approve`; without it a
gate returns `BLOCKED_APPROVAL` and the pipeline stops. With HITL enabled, the
scriptable path is `researchclaw status <run-dir>`, then `approve -m "..."`,
`reject -r "..."`, `guide --stage N -m "..."`. The `attach` TUI needs a real
terminal — prefer the subcommands from an agent.

## Driving this project with pi

- pi auto-loads this file (`AGENTS.md`, also `CLAUDE.md`); the tables above
  point at the deeper docs.
- pi as the **LLM backend** is verified working: `llm.provider: acp`,
  `llm.acp.agent: "pi"`, `acpx` on PATH (`acpx pi ...`), `max_turns: 1`.
  `researchclaw doctor` checks both the agent binary and `acpx`.
- The recommended split is still: let the pipeline generate with its own API
  models, and use pi for gate decisions, artifact review and repair. A 23-stage
  run on one ACP session is slow and accumulates a lot of context.

## Codegen and experiment CLIs

- `experiment.cli_agent.provider` accepts `llm`, `claude_code`, `codex`, `pi`.
  `pi` runs `pi -p <prompt> --no-session --mode text --tools read,bash,edit,write`
  in the experiment workdir; pi has no `--max-budget-usd`, so that setting is
  ignored for it.
- The domain experiment agents (`collider_agent`, `biology_agent`,
  `stat_agent`) shell out to `<claude_binary> -p <instruction>`. `--max-turns`
  is appended **only** for Claude binaries, so `pi`, `codex` or `opencode` can
  be used via `claude_binary` without hitting an unknown flag.
- `experiment.sandbox.allowed_imports` is **documentation only** — nothing
  enforces it; isolation is the subprocess plus the timeout.

## MCP server (real)

`researchclaw mcp --start -c config.yaml` serves JSON-RPC 2.0 over stdio with
newline-delimited framing (MCP spec) and exposes six working tools. Register it
in pi once:

```bash
pi mcp add researchclaw -- python -m researchclaw mcp --start -c config.arc.yaml
```

`run_pipeline` starts a background thread running the real pipeline,
`get_pipeline_status` reads `checkpoint.json` / `heartbeat.json` / per-stage
health, `search_literature` calls the literature layer, `review_paper` asks the
configured LLM. Only stdio transport is implemented; stdout is the protocol
stream, so diagnostics go to stderr.
- `.agents/skills/` is shared on purpose: the pipeline's skill library
  (builtin + `~/.researchclaw/skills` + project dirs, listed by
  `researchclaw skills list`) *and* pi's project skill directory, so the
  `researchclaw` skill there loads for both. `~/.agents/skills/` is the
  user-level pi directory; `.claude/skills/` is a legacy fallback the pipeline
  still scans.
