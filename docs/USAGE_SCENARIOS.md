# Usage scenarios

AutoResearchClaw is a CLI project, not a service: it reads a config, writes a
run directory, and prints where things went. How you drive it depends on the
job. There are three honest scenarios; they share the same config and
artifacts, so you can move between them mid-project.

| You want | Scenario | Cost | Reproducible |
|---|---|---|---|
| A literature review (or full paper) once, polished with an agent | **B — agent-driven** | higher (two LLM layers) | by committed config + run dir |
| Many reviews, batch runs, CI, "re-run exactly this" | **A — plain CLI** | lowest | yes, byte-stable inputs |
| Change AutoResearchClaw itself | **C — develop in the repo** | n/a | tests |

The rest of this page is the exact commands for each.

## Prerequisites (once)

```bash
python --version                  # 3.11+ (3.13 is what this repo is tested on)
pip install -e D:/AutoResearchClaw
```

Why the install: `python -m researchclaw` works **only from the repo root**.
After `pip install -e` the shorter `researchclaw` entry point is on `PATH`, so
the commands below work from any project folder.

You also need an LLM. Either a provider with an API key in the config, or — if
you already use pi — let pi be the backend:

```yaml
llm:
  provider: acp
  acp:
    agent: pi
```

## Scenario A — one command per review

One folder per review, config next to it. Everything (config discovery,
`artifacts/`, the literature cache) is resolved **relative to the current
directory**, so each folder is self-contained.

```bash
mkdir D:/research/attention-sparsity && cd D:/research/attention-sparsity
researchclaw init                          # writes config.arc.yaml
# edit: research.topic, research.min_year, literature_search.*, llm.*
researchclaw validate -c config.arc.yaml
researchclaw doctor   -c config.arc.yaml   # "Result: PASS" = LLM reachable, tools ok

researchclaw review -c config.arc.yaml -t "Sparse attention for long-context LLMs" \
  -o artifacts/review-1
```

This runs stages 1–7 only (topic → literature → synthesis) and renders one
document:

```
artifacts/review-1/review.md              # the review itself
artifacts/review-1/references.bib          # BibTeX for the kept papers
artifacts/review-1/stage-03/queries.json   # expanded queries + year policy
artifacts/review-1/stage-04/candidates.jsonl
artifacts/review-1/stage-04/search_meta.json   # real_search, cache, year_filter
artifacts/review-1/stage-05/shortlist.jsonl
artifacts/review-1/stage-06/cards/
artifacts/review-1/stage-07/synthesis.md
```

**Exit codes:** `0` — complete; `2` — the pipeline paused (a gate, or Stage 5
rejected every paper) but the document was still written; `1` — nothing to
render. On `2`, read the warnings at the top of the markdown before trusting
it. `doctor` failing only on `llm_connectivity` / `api_key_valid` means the
config's LLM block is not usable yet — fix that first, nothing else will run.

**Cheap iteration without re-running everything:**

```bash
# narrower/older literature only: redo stages 3-7 in the same run dir
researchclaw run -c config.arc.yaml -o artifacts/review-1 \
  --from-stage SEARCH_STRATEGY --to-stage SYNTHESIS --auto-approve
# then re-render. No pipeline, no LLM calls — just the document.
researchclaw review -c config.arc.yaml --run-dir artifacts/review-1
```

Repeat runs are nearly free while the cache lives: it is
`.researchclaw_cache/literature` **in the current directory**, keyed by
`query|source|limit|year_min`.

**For reproducibility:** commit `config.arc.yaml` next to the run, pin
`literature_search.sources` and `max_results_per_query`, and treat
`.researchclaw_cache` as disposable.

## Scenario B — agent-driven iteration

Best for a review you will actually polish: the agent reads the artifacts,
finds the gaps, and fixes them. Run **pi in your research folder**, not inside
the AutoResearchClaw repository.

```bash
cd D:/research/attention-sparsity
pi
```

Then talk in outcomes, not commands:

```
> Прогони литературный обзор по теме "Sparse attention for long-context
  LLMs" через researchclaw review. Потом покажи, какие статьи отсеклись
  на скрининге и почему. Если классики мало — снизь min_year до 2015 и
  перезапусти только сбор литературы, не трогая синтез. Проверь, что
  real_search в search_meta.json = true.
```

The loop the agent should follow (works with bash alone — no MCP needed):

1. `researchclaw review -c config.arc.yaml -t "…" -o artifacts/review-1`
2. read `review.md`, `stage-05/shortlist.jsonl`,
   `stage-04/search_meta.json` (is `real_search` true? which sources returned
   nothing?)
3. adjust `config.arc.yaml` (queries come from Stage 3, so prefer editing
   `research.topic` framing or `min_year`), then re-enter at the right stage:

   ```bash
   researchclaw run -c config.arc.yaml -o artifacts/review-1 \
     --from-stage LITERATURE_COLLECT --to-stage SYNTHESIS --auto-approve
   researchclaw review -c config.arc.yaml --run-dir artifacts/review-1
   ```

4. verify each `cite_key` in `references.bib` against a real DOI/arXiv record
   before the review leaves the machine.

Agent guardrails worth stating in the prompt: never accept a run whose
`search_meta.json` has `real_search: false` (those candidates are LLM-written),
and never edit `stage-*/` artifacts by hand — edit the config and re-run the
stage.

### MCP, when you want it

MCP gives the agent structured tools (status, results, ad-hoc search) instead
of dumping JSON into its context. Register the server with the **project
folder** as cwd:

```bash
cd D:/research/attention-sparsity
pi mcp add researchclaw --cwd "$PWD" -- researchclaw mcp --start -c "$PWD/config.arc.yaml"
pi mcp list     # researchclaw: connected, 6 tools
```

Limits, so you pick the right path:

- There is no review renderer in MCP. A literature-only run is
  `run_pipeline(topic=…, to_stage="SYNTHESIS", auto_approve=true)`, and the
  review text comes from `stage-07/synthesis.md` (or run the CLI `review`
  afterwards).
- `run_pipeline` defaults to `auto_approve: false` and there is no
  approve-gate tool, so pass `auto_approve: true` or the run stops at Stage 5.
- Runs started over MCP are named `mcp-<timestamp>-<slug>-<hash>` and are
  waited for on shutdown; closing the client does not kill them.

### Hybrid (the one that usually pays off)

Let Scenario A produce the first draft (cheap, deterministic), then switch to
Scenario B to attack the weak spots — missing classics, one-source coverage,
weak synthesis — and re-render with `--run-dir`. You get the reproducibility of
A for the base run and the judgement of B for the polish.

## Scenario C — developing AutoResearchClaw itself

Only this scenario expects pi to be opened in `D:/AutoResearchClaw`: that is
where `AGENTS.md` and `.agents/skills/researchclaw` are auto-loaded, and where
`doctor`/`pytest` live.

```bash
cd D:/AutoResearchClaw
researchclaw doctor -c config.researchclaw.example.yaml
python -m pytest -q                      # 3,000+ tests
```

Your own research topics do not belong here: their `artifacts/` would mix with
the engine's own runs and the repo's config.

## Choosing in one line

- **Repeatable/cheap** → A. **One review worth polishing** → B. **Changing the
  tool** → C.
- Start with A, escalate to B when the draft is weak. The run directory and the
  config are the handoff between the two.
