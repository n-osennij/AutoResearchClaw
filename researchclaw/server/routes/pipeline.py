"""Pipeline control API routes."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

logger = logging.getLogger(__name__)

import re as _re
_RUN_ID_RE = _re.compile(r"^rc-\d{8}-\d{6}-[a-f0-9]+$")


def _validated_run_dir(run_id: str) -> Path:
    """Validate run_id format and return the run directory path."""
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail=f"Invalid run_id format: {run_id}")
    run_dir = Path("artifacts") / run_id
    # Ensure resolved path is under artifacts/
    if not run_dir.resolve().is_relative_to(Path("artifacts").resolve()):
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id}")
    return run_dir


# Stage 4 (LITERATURE_COLLECT) writes its artifacts here.
_LITERATURE_STAGE_DIR = "stage-04"
_LITERATURE_CANDIDATE_HEAD = 20
#: The fields a client needs to triage a candidate: identity, provenance and
#: the two integrity flags. Abstracts are dropped — they are the bulk of the
#: file and no UI renders them from here.
_LITERATURE_CANDIDATE_FIELDS = (
    "id",
    "title",
    "year",
    "venue",
    "doi",
    "url",
    "source",
    "is_preprint",
    "retracted",
    "concern",
    "cite_key",
)


def _read_literature(run_dir: Path) -> dict[str, Any]:
    """Summarise Stage 4's output for the run detail endpoint.

    Never raises: a half-written run directory is the normal case while the
    pipeline is running, so a missing or malformed artifact degrades to
    ``None``/empty lists rather than turning the whole request into a 500.
    """
    stage_dir = run_dir / _LITERATURE_STAGE_DIR
    search_meta: dict[str, Any] | None = None
    meta_path = stage_dir / "search_meta.json"
    if meta_path.is_file():
        try:
            loaded = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            loaded = None
        if isinstance(loaded, dict):
            search_meta = loaded

    candidates: list[dict[str, Any]] = []
    by_source: dict[str, int] = {}
    total = 0
    candidates_path = stage_dir / "candidates.jsonl"
    if candidates_path.is_file():
        try:
            with candidates_path.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if not isinstance(row, dict):
                        continue
                    total += 1
                    source = str(row.get("source") or "")
                    if source:
                        by_source[source] = by_source.get(source, 0) + 1
                    if len(candidates) < _LITERATURE_CANDIDATE_HEAD:
                        candidates.append(
                            {
                                key: row.get(key)
                                for key in _LITERATURE_CANDIDATE_FIELDS
                            }
                        )
        except OSError:
            pass

    sources: list[str] | None = None
    if search_meta is not None:
        meta_sources = search_meta.get("sources")
        if isinstance(meta_sources, list | tuple) and meta_sources:
            sources = [str(item) for item in meta_sources]
    if sources is None:
        # Older runs carry no ``sources`` key in search_meta.json; the sources
        # that actually returned something are the ones worth showing.
        observed = sorted(by_source)
        sources = observed or None

    return {
        "sources": sources,
        "counts": {"total": total, "by_source": by_source},
        "search_meta": search_meta,
        "candidates": candidates,
    }


router = APIRouter(prefix="/api", tags=["pipeline"])


class PipelineStartRequest(BaseModel):
    """Request body for starting a pipeline run."""

    topic: str | None = None
    auto_approve: bool = True


class PipelineStartResponse(BaseModel):
    """Response after starting a pipeline."""

    run_id: str
    status: str
    output_dir: str


# In-memory tracking of the active run (single-tenant MVP)
_active_run: dict[str, Any] | None = None
_run_task: asyncio.Task[Any] | None = None
_run_lock = asyncio.Lock()


def _get_app_state() -> dict[str, Any]:
    """Get shared application state (set by app.py)."""
    from researchclaw.server.app import _app_state
    return _app_state


@router.post("/pipeline/start", response_model=PipelineStartResponse)
async def start_pipeline(req: PipelineStartRequest) -> PipelineStartResponse:
    """Start a new pipeline run."""
    global _active_run, _run_task

    async with _run_lock:
        if _active_run and _active_run.get("status") == "running":
            raise HTTPException(status_code=409, detail="A pipeline is already running")

        state = _get_app_state()
        config = state["config"]

        if req.topic:
            import dataclasses
            new_research = dataclasses.replace(config.research, topic=req.topic)
            config = dataclasses.replace(config, research=new_research)

        import hashlib
        from datetime import datetime, timezone

        ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        topic_hash = hashlib.sha256(config.research.topic.encode()).hexdigest()[:6]
        run_id = f"rc-{ts}-{topic_hash}"
        run_dir = _validated_run_dir(run_id)
        run_dir.mkdir(parents=True, exist_ok=True)

        _active_run = {
            "run_id": run_id,
            "status": "running",
            "output_dir": str(run_dir),
            "topic": config.research.topic,
        }

    async def _run_in_background() -> None:
        global _active_run
        try:
            from researchclaw.adapters import AdapterBundle
            from researchclaw.pipeline.runner import execute_pipeline

            kb_root = Path(config.knowledge_base.root) if config.knowledge_base.root else None
            if kb_root:
                kb_root.mkdir(parents=True, exist_ok=True)

            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: execute_pipeline(
                    run_dir=run_dir,
                    run_id=run_id,
                    config=config,
                    adapters=AdapterBundle(),
                    auto_approve_gates=req.auto_approve,
                    skip_noncritical=True,
                    kb_root=kb_root,
                ),
            )
            done = sum(1 for r in results if r.status.value == "done")
            failed = sum(1 for r in results if r.status.value == "failed")
            if _active_run:
                _active_run["status"] = "completed" if failed == 0 else "failed"
                _active_run["stages_done"] = done
                _active_run["stages_failed"] = failed
        except Exception as exc:
            logger.exception("Pipeline run failed")
            if _active_run:
                _active_run["status"] = "failed"
                _active_run["error"] = str(exc)

    _run_task = asyncio.create_task(_run_in_background())

    return PipelineStartResponse(
        run_id=run_id,
        status="running",
        output_dir=str(run_dir),
    )


@router.post("/pipeline/stop")
async def stop_pipeline() -> dict[str, str]:
    """Stop the currently running pipeline."""
    global _active_run, _run_task

    if not _run_task or not _active_run:
        raise HTTPException(status_code=404, detail="No pipeline is running")

    _run_task.cancel()
    _active_run["status"] = "stopped"
    return {"status": "stopped"}


@router.get("/pipeline/status")
async def pipeline_status() -> dict[str, Any]:
    """Get current pipeline run status."""
    if not _active_run:
        return {"status": "idle"}
    return _active_run


@router.get("/pipeline/stages")
async def pipeline_stages() -> dict[str, Any]:
    """Get the 23-stage pipeline definition."""
    from researchclaw.pipeline.stages import Stage

    stages = []
    for s in Stage:
        stages.append({
            "number": int(s),
            "name": s.name,
            "label": getattr(s, "label", s.name.replace("_", " ").title()),
            "phase": getattr(s, "phase", ""),
        })
    return {"stages": stages}


@router.get("/runs")
async def list_runs() -> dict[str, Any]:
    """List historical pipeline runs from artifacts/ directory."""
    artifacts = Path("artifacts")
    runs: list[dict[str, Any]] = []
    if artifacts.exists():
        for d in sorted(artifacts.iterdir(), reverse=True):
            if d.is_dir() and d.name.startswith("rc-"):
                info: dict[str, Any] = {"run_id": d.name, "path": str(d)}
                # Try reading checkpoint
                ckpt = d / "checkpoint.json"
                if ckpt.exists():
                    try:
                        with ckpt.open() as f:
                            info["checkpoint"] = json.load(f)
                    except Exception:
                        pass
                runs.append(info)
    return {"runs": runs[:50]}  # limit to 50 most recent


@router.get("/runs/{run_id}")
async def get_run(run_id: str) -> dict[str, Any]:
    """Get details for a specific run."""
    run_dir = _validated_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    info: dict[str, Any] = {"run_id": run_id, "path": str(run_dir)}

    ckpt = run_dir / "checkpoint.json"
    if ckpt.exists():
        try:
            with ckpt.open() as f:
                info["checkpoint"] = json.load(f)
        except Exception:
            pass

    # List stage directories
    stage_dirs = sorted(
        [d.name for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("stage-")]
    )
    info["stages_completed"] = stage_dirs

    # Check for paper
    for pattern in ["paper.md", "paper.tex", "paper.pdf"]:
        found = list(run_dir.rglob(pattern))
        if found:
            info[f"has_{pattern.split('.')[1]}"] = True

    info["literature"] = _read_literature(run_dir)

    return info


@router.get("/runs/{run_id}/metrics")
async def get_run_metrics(run_id: str) -> dict[str, Any]:
    """Get experiment metrics for a run."""
    run_dir = _validated_run_dir(run_id)
    if not run_dir.exists():
        raise HTTPException(status_code=404, detail=f"Run not found: {run_id}")

    metrics: dict[str, Any] = {}
    results_file = run_dir / "results.json"
    if results_file.exists():
        try:
            with results_file.open() as f:
                metrics = json.load(f)
        except Exception:
            pass

    return {"run_id": run_id, "metrics": metrics}
