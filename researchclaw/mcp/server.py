"""ResearchClaw MCP Server: expose pipeline capabilities to external agents.

Speaks JSON-RPC 2.0 over stdio with newline-delimited framing (the MCP stdio
spec), so pi, Claude Code, Codex and any other MCP client can register

    researchclaw mcp --start -c config.yaml

and call the tools below. ``initialize``, ``ping``, ``tools/list`` and
``tools/call`` are implemented; ``resources/*`` and ``prompts/*`` answer with
empty lists instead of hanging.

Tool handlers are real: ``run_pipeline`` starts a background thread that runs
:func:`researchclaw.pipeline.runner.execute_pipeline`, ``get_pipeline_status``
reads the run directory (``checkpoint.json``, ``heartbeat.json``, per-stage
``stage_health.json``), ``search_literature`` calls the literature layer and
``review_paper`` asks the configured LLM for a review.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import threading
import time
import traceback
from dataclasses import replace
from pathlib import Path
from typing import Any

from researchclaw import __version__
from researchclaw.mcp.tools import TOOL_DEFINITIONS, list_tool_names

logger = logging.getLogger(__name__)

_VALID_RUN_ID = re.compile(r"^[a-zA-Z0-9_\-]+$")
_PROTOCOL_VERSION = "2024-11-05"

#: Files a run may leave results in, most specific first.
_RESULT_FILES = (
    "stage-14/experiment_summary.json",
    "stage-12/experiment_history.json",
    "stage-12/runs/results.json",
    "experiment_results.json",
)


def _artifacts_root() -> Path:
    """Default artifact root — relative to the process working directory."""
    return Path("artifacts")


def _validated_run_dir(run_id: str, root: Path | None = None) -> Path:
    """Validate run_id to prevent path traversal and return the run directory."""
    if not _VALID_RUN_ID.match(run_id):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    base = (root or _artifacts_root()).resolve()
    run_dir = (base / run_id).resolve()
    if not run_dir.is_relative_to(base):
        raise ValueError(f"Invalid run_id: {run_id!r}")
    return run_dir


def _slug(text: str, limit: int = 32) -> str:
    """Filesystem-safe run-id suffix."""
    slug = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return (slug or "run")[:limit].strip("-") or "run"


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class ResearchClawMCPServer:
    """MCP server exposing AutoResearchClaw capabilities as tools.

    ``config`` may be an :class:`~researchclaw.config.RCConfig`; otherwise the
    server loads ``config_path`` or falls back to ``config.arc.yaml`` /
    ``config.yaml`` in the working directory.
    """

    def __init__(
        self,
        config: Any = None,
        *,
        config_path: str | Path = "",
        artifacts_root: Path | str | None = None,
    ) -> None:
        self.config = config
        self._config_path = str(config_path or "")
        self._artifacts_root = (
            Path(artifacts_root) if artifacts_root is not None else _artifacts_root()
        )
        self._running = False
        self._runs: dict[str, dict[str, Any]] = {}

    # ── configuration ─────────────────────────────────────────────

    def _load_config(self, config_path: str | Path | None = None) -> Any:
        """Resolve the pipeline config for handlers that need it."""
        if self.config is not None and not config_path:
            return self.config
        from researchclaw.config import RCConfig

        candidates = [config_path, self._config_path, "config.arc.yaml", "config.yaml"]
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(str(candidate))
            if path.is_file():
                return RCConfig.load(path, check_paths=False)
        raise FileNotFoundError(
            "No config file found — pass config_path or create config.arc.yaml"
        )

    # ── tools ─────────────────────────────────────────────────────

    def get_tools(self) -> list[dict[str, Any]]:
        """Return the list of available MCP tools."""
        return TOOL_DEFINITIONS

    async def handle_tool_call(
        self, name: str, arguments: dict[str, Any]
    ) -> dict[str, Any]:
        """Handle an incoming MCP tool call and return a plain result dict."""
        if name not in list_tool_names():
            return {"error": f"Unknown tool: {name}", "success": False}

        arguments = arguments or {}
        logger.info(
            "MCP tool call: %s(%s)", name, json.dumps(arguments, default=str)[:200]
        )

        try:
            if name == "run_pipeline":
                return await self._handle_run_pipeline(arguments)
            if name == "get_pipeline_status":
                return await self._handle_get_status(arguments)
            if name == "get_experiment_results":
                return await self._handle_get_results(arguments)
            if name == "search_literature":
                return await self._handle_search_literature(arguments)
            if name == "review_paper":
                return await self._handle_review_paper(arguments)
            if name == "get_paper":
                return await self._handle_get_paper(arguments)
            return {"error": f"Handler not implemented: {name}", "success": False}
        except Exception as exc:  # noqa: BLE001 - surfaced to the MCP client
            logger.error("MCP tool call %s failed: %s", name, exc)
            return {"error": str(exc), "success": False}

    # ── tool handlers ─────────────────────────────────────────────

    async def _handle_run_pipeline(self, args: dict[str, Any]) -> dict[str, Any]:
        """Start a real pipeline run in a background thread."""
        topic = str(args.get("topic") or "").strip()
        if not topic:
            return {"success": False, "error": "topic is required"}
        try:
            config = self._load_config(args.get("config_path"))
        except (FileNotFoundError, OSError, ValueError) as exc:
            return {"success": False, "error": str(exc)}

        config = replace(config, research=replace(config.research, topic=topic))
        run_id = f"mcp-{time.strftime('%Y%m%d-%H%M%S')}-{_slug(topic)}"
        run_dir = self._artifacts_root / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        auto_approve = bool(args.get("auto_approve", False))

        thread = threading.Thread(
            target=self._run_pipeline_sync,
            args=(run_id, run_dir, config, auto_approve),
            name=f"rc-{run_id}",
            daemon=True,
        )
        self._runs[run_id] = {
            "thread": thread,
            "run_dir": run_dir,
            "topic": topic,
            "auto_approve": auto_approve,
            "started_at": time.time(),
        }
        thread.start()
        return {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
            "auto_approve": auto_approve,
            "message": f"Pipeline started for topic: {topic}",
        }

    @staticmethod
    def _run_pipeline_sync(
        run_id: str, run_dir: Path, config: Any, auto_approve: bool
    ) -> None:
        """Thread body: run the pipeline and log the outcome next to it."""
        from researchclaw.adapters import AdapterBundle
        from researchclaw.pipeline.runner import execute_pipeline

        try:
            execute_pipeline(
                run_dir=run_dir,
                run_id=run_id,
                config=config,
                adapters=AdapterBundle(),
                auto_approve_gates=auto_approve,
            )
        except Exception:  # noqa: BLE001 - background thread must not die silently
            logger.exception("MCP pipeline run failed: %s", run_id)
            try:
                (run_dir / "mcp_run_error.log").write_text(
                    traceback.format_exc(), encoding="utf-8"
                )
            except OSError:
                pass

    async def _handle_get_status(self, args: dict[str, Any]) -> dict[str, Any]:
        """Read a run directory: checkpoint, heartbeat and per-stage health."""
        run_id = str(args["run_id"])
        run_dir = _validated_run_dir(run_id, self._artifacts_root)
        if not run_dir.is_dir():
            return {"success": False, "error": f"Run not found: {run_id}"}

        status: dict[str, Any] = {
            "success": True,
            "run_id": run_id,
            "run_dir": str(run_dir),
        }
        checkpoint = _read_json(run_dir / "checkpoint.json")
        if checkpoint is not None:
            status["checkpoint"] = checkpoint
        heartbeat = _read_json(run_dir / "heartbeat.json")
        if heartbeat is not None:
            status["heartbeat"] = heartbeat

        stages: list[dict[str, Any]] = []
        for stage_dir in sorted(run_dir.glob("stage-*")):
            health = _read_json(stage_dir / "stage_health.json") or {}
            entry: dict[str, Any] = {"stage": stage_dir.name}
            if health:
                entry["status"] = health.get("status")
                if health.get("error"):
                    entry["error"] = health["error"]
            stages.append(entry)
        status["stages"] = stages

        record = self._runs.get(run_id)
        if record is not None:
            status["running"] = bool(record["thread"].is_alive())
        return status

    async def _handle_get_results(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return whatever structured experiment results the run has produced."""
        run_id = str(args["run_id"])
        run_dir = _validated_run_dir(run_id, self._artifacts_root)
        if not run_dir.is_dir():
            return {"success": False, "error": f"Run not found: {run_id}"}

        candidates = list(_RESULT_FILES)
        stage = str(args.get("stage") or "").strip()
        if stage:
            candidates = [
                f"stage-{stage}/{name}"
                for name in (
                    "experiment_summary.json",
                    "results.json",
                    "experiment_history.json",
                )
            ] + candidates

        for relative in candidates:
            payload = _read_json(run_dir / relative)
            if payload is not None:
                return {
                    "success": True,
                    "run_id": run_id,
                    "source": relative,
                    "results": payload,
                }

        for path in sorted(run_dir.glob("stage-12/runs/*/results.json")):
            payload = _read_json(path)
            if payload is not None:
                return {
                    "success": True,
                    "run_id": run_id,
                    "source": str(path.relative_to(run_dir)).replace("\\", "/"),
                    "results": payload,
                }
        return {
            "success": False,
            "error": f"No experiment results found in {run_dir}",
        }

    async def _handle_search_literature(self, args: dict[str, Any]) -> dict[str, Any]:
        """Query the literature backends configured for the project."""
        query = str(args.get("query") or "").strip()
        if not query:
            return {"success": False, "error": "query is required"}
        limit = int(args.get("max_results") or 10)
        sources = args.get("sources") or None

        from researchclaw.literature.search import SourceKeys, search_papers

        keys = SourceKeys()
        try:
            literature = self._load_config().literature_search
            keys = SourceKeys(
                s2_api_key=literature.s2_api_key,
                openalex_email=literature.openalex_email,
                openalex_api_key=literature.openalex_api_key,
                pubmed_api_key=literature.pubmed_api_key,
                core_api_key=literature.core_api_key,
            )
        except (FileNotFoundError, OSError, ValueError):
            logger.debug("No config for search_literature — using keyless defaults")

        kwargs: dict[str, Any] = {"limit": limit, "keys": keys}
        if sources:
            kwargs["sources"] = tuple(str(s) for s in sources)
        try:
            papers = await asyncio.to_thread(search_papers, query, **kwargs)
        except Exception as exc:  # noqa: BLE001 - network source failures
            return {"success": False, "error": f"Literature search failed: {exc}"}

        return {
            "success": True,
            "query": query,
            "count": len(papers),
            "results": [
                {
                    "title": paper.title,
                    "authors": [author.name for author in paper.authors],
                    "year": paper.year,
                    "venue": paper.venue,
                    "doi": paper.doi,
                    "url": paper.url,
                    "pdf_url": paper.pdf_url,
                    "source": paper.source,
                    "abstract": (paper.abstract or "")[:1500],
                }
                for paper in papers[:limit]
            ],
        }

    async def _handle_review_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        """Ask the configured LLM for a review of a paper file."""
        paper_path = Path(str(args.get("paper_path") or ""))
        if not paper_path.is_file():
            return {"success": False, "error": f"Paper not found: {paper_path}"}
        text = paper_path.read_text(encoding="utf-8", errors="replace")
        if not text.strip():
            return {"success": False, "error": f"Paper is empty: {paper_path}"}

        try:
            config = self._load_config()
            from researchclaw.llm import create_llm_client

            llm = create_llm_client(config)
        except Exception as exc:  # noqa: BLE001 - report, don't crash the server
            return {"success": False, "error": f"LLM unavailable: {exc}"}

        prompt = (
            "Review this research paper draft. Report: (1) contribution clarity, "
            "(2) methodological soundness, (3) whether claims match the evidence, "
            "(4) missing baselines or ablations, (5) concrete fixes. Be specific "
            "and terse; no praise.\n\nPAPER:\n" + text[:60_000]
        )
        try:
            response = await asyncio.to_thread(
                llm.chat,
                [{"role": "user", "content": prompt}],
                system="You are a demanding conference reviewer.",
            )
        except Exception as exc:  # noqa: BLE001
            return {"success": False, "error": f"Review call failed: {exc}"}
        return {
            "success": True,
            "paper_path": str(paper_path),
            "review": getattr(response, "content", str(response)),
        }

    async def _handle_get_paper(self, args: dict[str, Any]) -> dict[str, Any]:
        """Return the paper source (markdown or LaTeX) from the run directory."""
        run_id = str(args["run_id"])
        fmt = str(args.get("format") or "markdown")
        run_dir = _validated_run_dir(run_id, self._artifacts_root)
        if not run_dir.is_dir():
            return {"success": False, "error": f"Run not found: {run_id}"}

        names = ("paper.tex",) if fmt == "latex" else ("paper_draft.md",)
        search_paths = [run_dir / name for name in names]
        for stage_dir in sorted(run_dir.glob("stage-*")):
            search_paths.extend(stage_dir / name for name in names)
        for path in search_paths:
            if path.is_file():
                return {
                    "success": True,
                    "run_id": run_id,
                    "path": str(path),
                    "content": path.read_text(encoding="utf-8", errors="replace"),
                }
        return {
            "success": False,
            "error": f"Paper ({fmt}) not found in {run_dir}",
        }

    # ── JSON-RPC ──────────────────────────────────────────────────

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        """Dispatch one JSON-RPC message; None for notifications."""
        method = message.get("method")
        request_id = message.get("id")
        if request_id is None:
            # Notifications (initialized, cancelled, ...) need no response.
            return None

        if method == "initialize":
            result: dict[str, Any] = {
                "protocolVersion": _PROTOCOL_VERSION,
                "capabilities": {"tools": {}},
                "serverInfo": {"name": "researchclaw", "version": __version__},
            }
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": TOOL_DEFINITIONS}
        elif method == "tools/call":
            params = message.get("params") or {}
            payload = await self.handle_tool_call(
                str(params.get("name") or ""), params.get("arguments") or {}
            )
            result = {
                "content": [
                    {
                        "type": "text",
                        "text": json.dumps(payload, ensure_ascii=False, default=str),
                    }
                ],
                "isError": not bool(payload.get("success", True)),
            }
        elif method == "resources/list":
            result = {"resources": []}
        elif method == "prompts/list":
            result = {"prompts": []}
        else:
            return {
                "jsonrpc": "2.0",
                "id": request_id,
                "error": {
                    "code": -32601,
                    "message": f"Method not found: {method}",
                },
            }

        return {"jsonrpc": "2.0", "id": request_id, "result": result}

    async def serve(self, io: Any) -> None:
        """Read JSON-RPC messages from *io* until EOF."""
        from researchclaw.mcp.transport import StdioTransport

        transport: StdioTransport = io
        await transport.start()
        self._running = True
        try:
            while True:
                try:
                    message = await transport.receive()
                except EOFError:
                    break
                except ValueError as exc:
                    await transport.send(
                        {
                            "jsonrpc": "2.0",
                            "id": None,
                            "error": {"code": -32700, "message": str(exc)},
                        }
                    )
                    continue
                response = await self.handle_message(message)
                if response is not None:
                    await transport.send(response)
        finally:
            await transport.close()
            self._running = False

    # ── server lifecycle ──────────────────────────────────────────

    async def start(self, transport: str = "stdio") -> None:
        """Serve MCP over stdio until the client closes the stream."""
        if transport != "stdio":
            raise ValueError(
                f"Unsupported MCP transport: {transport!r} (only 'stdio' is implemented)"
            )
        from researchclaw.mcp.transport import StdioTransport

        await self.serve(StdioTransport())

    async def stop(self) -> None:
        """Stop the MCP server (the stdio loop exits on EOF)."""
        self._running = False
        logger.info("MCP server stopped")

    @property
    def is_running(self) -> bool:
        return self._running
