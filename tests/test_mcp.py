"""Tests for MCP integration (C3): Server, Client, Tools, Transport, Registry."""

from __future__ import annotations

import asyncio
import io
import json
from pathlib import Path
from typing import Any

import pytest

from researchclaw.mcp.tools import TOOL_DEFINITIONS, get_tool_schema, list_tool_names
from researchclaw.mcp.server import ResearchClawMCPServer
from researchclaw.mcp.client import MCPClient
from researchclaw.mcp.registry import MCPServerRegistry
from researchclaw.mcp.transport import SSETransport, StdioTransport


# ══════════════════════════════════════════════════════════════════
# MCP Tools tests
# ══════════════════════════════════════════════════════════════════


class TestMCPTools:
    def test_tool_definitions_not_empty(self) -> None:
        assert len(TOOL_DEFINITIONS) >= 6

    def test_all_tools_have_required_fields(self) -> None:
        for tool in TOOL_DEFINITIONS:
            assert "name" in tool
            assert "description" in tool
            assert "inputSchema" in tool
            assert tool["inputSchema"]["type"] == "object"

    def test_get_tool_schema_exists(self) -> None:
        schema = get_tool_schema("run_pipeline")
        assert schema is not None
        assert schema["name"] == "run_pipeline"

    def test_get_tool_schema_missing(self) -> None:
        assert get_tool_schema("nonexistent") is None

    def test_list_tool_names(self) -> None:
        names = list_tool_names()
        assert "run_pipeline" in names
        assert "get_pipeline_status" in names
        assert "search_literature" in names

    def test_run_pipeline_requires_topic(self) -> None:
        schema = get_tool_schema("run_pipeline")
        assert schema is not None
        assert "topic" in schema["inputSchema"]["required"]

    def test_get_paper_has_format_enum(self) -> None:
        schema = get_tool_schema("get_paper")
        assert schema is not None
        props = schema["inputSchema"]["properties"]
        assert "format" in props
        assert "enum" in props["format"]


# ══════════════════════════════════════════════════════════════════
# MCP Server tests
# ══════════════════════════════════════════════════════════════════


class TestMCPServer:
    @staticmethod
    def _server(tmp_path: Path) -> ResearchClawMCPServer:
        from researchclaw.config import RCConfig

        config = RCConfig.from_dict(
            {
                "project": {"name": "mcp-test", "mode": "docs-first"},
                "research": {"topic": "mcp testing"},
                "runtime": {"timezone": "UTC"},
                "notifications": {"channel": "local"},
                "knowledge_base": {"root": str(tmp_path / "kb")},
                "llm": {
                    "provider": "openai-compatible",
                    "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY",
                    "api_key": "inline",
                },
            },
            project_root=tmp_path,
            check_paths=False,
        )
        return ResearchClawMCPServer(
            config=config, artifacts_root=tmp_path / "artifacts"
        )

    def test_get_tools(self) -> None:
        server = ResearchClawMCPServer()
        tools = server.get_tools()
        assert len(tools) >= 6
        names = [t["name"] for t in tools]
        assert "run_pipeline" in names

    def test_handle_unknown_tool(self) -> None:
        server = ResearchClawMCPServer()
        result = asyncio.run(server.handle_tool_call("nonexistent", {}))
        assert result["success"] is False
        assert "Unknown tool" in result["error"]

    def test_handle_run_pipeline_starts_background_run(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}

        def fake_run(
            run_id: str, run_dir: Path, config: Any, auto: bool, to_stage: str = ""
        ) -> None:
            captured.update(
                run_id=run_id,
                run_dir=Path(run_dir),
                topic=config.research.topic,
                auto=auto,
                to_stage=to_stage,
            )

        monkeypatch.setattr(server, "_run_pipeline_sync", fake_run)
        result = asyncio.run(
            server.handle_tool_call(
                "run_pipeline",
                {"topic": "GNN drug discovery", "auto_approve": True},
            )
        )

        assert result["success"] is True
        assert "GNN drug discovery" in result["message"]
        assert str(result["run_id"]).startswith("mcp-")
        assert captured["topic"] == "GNN drug discovery"
        assert captured["auto"] is True
        assert captured["to_stage"] == ""
        assert Path(result["run_dir"]).is_dir()

    def test_run_pipeline_accepts_to_stage(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}

        def fake_run(
            run_id: str, run_dir: Path, config: Any, auto: bool, to_stage: str = ""
        ) -> None:
            captured["to_stage"] = to_stage

        monkeypatch.setattr(server, "_run_pipeline_sync", fake_run)
        result = asyncio.run(
            server.handle_tool_call(
                "run_pipeline", {"topic": "literature only", "to_stage": "SYNTHESIS"}
            )
        )
        assert result["success"] is True
        assert result["to_stage"] == "SYNTHESIS"
        assert captured["to_stage"] == "SYNTHESIS"

    def test_run_pipeline_rejects_unknown_to_stage(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        result = asyncio.run(
            server.handle_tool_call(
                "run_pipeline", {"topic": "x", "to_stage": "NOT_A_STAGE"}
            )
        )
        assert result["success"] is False
        assert "Unknown to_stage" in result["error"]

    def test_handle_run_pipeline_without_config(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The tool searches the cwd for config.arc.yaml / config.yaml; a
        # developer's gitignored local config used to turn this "no config"
        # assertion into a real 3-minute pipeline run.
        monkeypatch.chdir(tmp_path)
        server = ResearchClawMCPServer(artifacts_root=tmp_path / "artifacts")
        result = asyncio.run(server.handle_tool_call("run_pipeline", {"topic": "GNN"}))
        assert result["success"] is False
        assert "config" in result["error"].lower()

    def test_handle_run_pipeline_requires_topic(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        result = asyncio.run(server.handle_tool_call("run_pipeline", {}))
        assert result["success"] is False
        assert "topic" in result["error"]

    def test_handle_get_status_missing_run(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        result = asyncio.run(
            server.handle_tool_call("get_pipeline_status", {"run_id": "nonexistent"})
        )
        assert result["success"] is False

    def test_handle_get_status_reads_run_dir(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        run_dir = tmp_path / "artifacts" / "rc-20260101-000000"
        stage_dir = run_dir / "stage-01"
        stage_dir.mkdir(parents=True)
        (run_dir / "checkpoint.json").write_text(
            json.dumps({"last_completed_stage": 1}), encoding="utf-8"
        )
        (run_dir / "heartbeat.json").write_text(
            json.dumps({"last_stage": "1"}), encoding="utf-8"
        )
        (stage_dir / "stage_health.json").write_text(
            json.dumps({"status": "done"}), encoding="utf-8"
        )

        result = asyncio.run(
            server.handle_tool_call(
                "get_pipeline_status", {"run_id": "rc-20260101-000000"}
            )
        )

        assert result["success"] is True
        assert result["checkpoint"]["last_completed_stage"] == 1
        assert result["heartbeat"]["last_stage"] == "1"
        assert result["stages"] == [{"stage": "stage-01", "status": "done"}]

    def test_handle_get_results_missing(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        result = asyncio.run(
            server.handle_tool_call("get_experiment_results", {"run_id": "missing"})
        )
        assert result["success"] is False

    def test_handle_get_results_reads_summary(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        stage_dir = tmp_path / "artifacts" / "rc-results" / "stage-14"
        stage_dir.mkdir(parents=True)
        (stage_dir / "experiment_summary.json").write_text(
            json.dumps({"metrics": {"accuracy": 0.91}}), encoding="utf-8"
        )

        result = asyncio.run(
            server.handle_tool_call("get_experiment_results", {"run_id": "rc-results"})
        )

        assert result["success"] is True
        assert result["source"].endswith("experiment_summary.json")
        assert result["results"]["metrics"]["accuracy"] == 0.91

    def test_handle_search_literature(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.literature.models import Paper

        server = self._server(tmp_path)
        captured: dict[str, Any] = {}

        def fake_search(query: str, **kwargs: Any) -> list[Paper]:
            captured["query"] = query
            captured.update(kwargs)
            return [
                Paper(
                    paper_id="p1",
                    title="Real Paper",
                    year=2024,
                    source="openalex",
                    pdf_url="https://example.org/p1.pdf",
                )
            ]

        monkeypatch.setattr("researchclaw.literature.search.search_papers", fake_search)
        result = asyncio.run(
            server.handle_tool_call(
                "search_literature", {"query": "transformers", "max_results": 5}
            )
        )

        assert result["success"] is True
        assert captured["query"] == "transformers"
        assert result["results"][0]["title"] == "Real Paper"
        assert result["results"][0]["pdf_url"].endswith(".pdf")

    def test_handle_search_literature_reports_failure(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)

        def boom(query: str, **kwargs: Any) -> list[Any]:
            raise RuntimeError("rate limited")

        monkeypatch.setattr("researchclaw.literature.search.search_papers", boom)
        result = asyncio.run(server.handle_tool_call("search_literature", {"query": "x"}))
        assert result["success"] is False
        assert "rate limited" in result["error"]

    # ── search_literature: evidence fields, limits, sources ──────────

    @staticmethod
    def _papers(count: int, **overrides: Any) -> list[Any]:
        from researchclaw.literature.models import Author, Paper

        return [
            Paper(
                paper_id=f"p{i}",
                title=f"Paper {i}",
                authors=(Author(name="Ada Lovelace"),),
                year=2024,
                source="pubmed",
                **overrides,
            )
            for i in range(count)
        ]

    @staticmethod
    def _patch_search(
        monkeypatch: pytest.MonkeyPatch, papers: list[Any], captured: dict[str, Any]
    ) -> None:
        def fake_search(query: str, **kwargs: Any) -> list[Any]:
            captured["query"] = query
            captured.update(kwargs)
            return papers

        monkeypatch.setattr("researchclaw.literature.search.search_papers", fake_search)

    def test_search_literature_carries_evidence_fields(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.literature.models import Paper

        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        papers = [
            Paper(
                paper_id="retracted-1",
                title="Retracted Study",
                year=2021,
                source="pubmed",
                pmid="12345678",
                doi="10.1000/xyz",
                arxiv_id="2101.00001",
                venue="Journal of Things",
                citation_count=42,
                pub_types=("Journal Article", "Retraction of Publication"),
                is_preprint=False,
                retracted=True,
                concern=True,
            )
        ]
        self._patch_search(monkeypatch, papers, captured)

        result = asyncio.run(
            server.handle_tool_call("search_literature", {"query": "retractions"})
        )

        assert result["success"] is True
        row = result["results"][0]
        assert row["retracted"] is True
        assert row["concern"] is True
        assert row["is_preprint"] is False
        assert row["pmid"] == "12345678"
        assert isinstance(row["pub_types"], list)
        assert "Retraction of Publication" in row["pub_types"]
        assert row["citation_count"] == 42
        assert row["venue"] == "Journal of Things"
        assert row["source"] == "pubmed"
        assert row["doi"] == "10.1000/xyz"
        assert row["arxiv_id"] == "2101.00001"
        assert "pdf_url" in row
        # The retraction is visible in the title too, not only in a flag.
        assert row["title"].startswith("[RETRACTED]")

    def test_search_literature_evidence_fields_absent_flags_default_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(1), captured)

        result = asyncio.run(
            server.handle_tool_call("search_literature", {"query": "clean"})
        )

        row = result["results"][0]
        assert row["retracted"] is False
        assert row["concern"] is False
        assert row["is_preprint"] is False
        assert row["pub_types"] == []

    @pytest.mark.parametrize("bad_limit", [-5, 0])
    def test_search_literature_non_positive_max_results_defaults_to_ten(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_limit: int
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(12), captured)

        result = asyncio.run(
            server.handle_tool_call(
                "search_literature", {"query": "x", "max_results": bad_limit}
            )
        )

        assert captured["limit"] == 10
        assert len(result["results"]) == 10
        assert result["count"] == len(result["results"])

    def test_search_literature_clamps_huge_max_results(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(150), captured)

        result = asyncio.run(
            server.handle_tool_call(
                "search_literature", {"query": "x", "max_results": 500}
            )
        )

        assert captured["limit"] == 100
        assert len(result["results"]) == 100
        assert result["count"] == len(result["results"])

    @pytest.mark.parametrize("bad_sources", [{"openalex": 1}, [1, 2], 5, [["arxiv"]]])
    def test_search_literature_rejects_bad_sources(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bad_sources: Any
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(1), captured)

        result = asyncio.run(
            server.handle_tool_call(
                "search_literature", {"query": "x", "sources": bad_sources}
            )
        )

        assert "error" in result
        assert result["success"] is False
        assert "sources" in result["error"]
        assert "search_papers" not in captured  # never reached the backends

    def test_search_literature_accepts_bare_string_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(1), captured)

        result = asyncio.run(
            server.handle_tool_call(
                "search_literature", {"query": "x", "sources": "pubmed"}
            )
        )

        assert result["success"] is True
        assert captured["sources"] == ("pubmed",)
        assert "unknown_sources" not in result

    def test_search_literature_reports_unknown_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(0), captured)

        result = asyncio.run(
            server.handle_tool_call(
                "search_literature", {"query": "x", "sources": ["pubmedd"]}
            )
        )

        assert result["success"] is True
        assert result["unknown_sources"] == ["pubmedd"]
        assert "pubmed" in result["valid_sources"]

    def test_search_literature_uses_config_sources_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.config import RCConfig

        config = RCConfig.from_dict(
            {
                "project": {"name": "mcp-test", "mode": "docs-first"},
                "research": {"topic": "mcp testing", "min_year": 2019},
                "runtime": {"timezone": "UTC"},
                "notifications": {"channel": "local"},
                "knowledge_base": {"root": str(tmp_path / "kb")},
                "literature_search": {"sources": ["pubmed"]},
                "llm": {
                    "provider": "openai-compatible",
                    "base_url": "http://localhost:1234/v1",
                    "api_key_env": "RC_TEST_KEY",
                    "api_key": "inline",
                },
            },
            project_root=tmp_path,
            check_paths=False,
        )
        server = ResearchClawMCPServer(
            config=config, artifacts_root=tmp_path / "artifacts"
        )
        captured: dict[str, Any] = {}
        self._patch_search(monkeypatch, self._papers(1), captured)

        result = asyncio.run(
            server.handle_tool_call("search_literature", {"query": "x"})
        )

        assert result["success"] is True
        assert captured["sources"] == ("pubmed",)
        assert captured["year_min"] == 2019

    def test_handle_review_paper(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = self._server(tmp_path)
        run_dir = tmp_path / "artifacts" / "rc-test" / "stage-22"
        run_dir.mkdir(parents=True)
        paper = run_dir / "paper_final.md"
        paper.write_text("# Title\nbody", encoding="utf-8")

        class _Response:
            content = "REVIEW: needs more baselines"

        class _LLM:
            def chat(self, messages: Any, **kwargs: Any) -> _Response:
                return _Response()

        monkeypatch.setattr(
            "researchclaw.llm.create_llm_client", lambda config: _LLM()
        )
        result = asyncio.run(
            server.handle_tool_call("review_paper", {"paper_path": str(paper)})
        )

        assert result["success"] is True
        assert "baselines" in result["review"]

    def test_handle_review_paper_missing(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        result = asyncio.run(
            server.handle_tool_call(
                "review_paper", {"paper_path": str(tmp_path / "nope.md")}
            )
        )
        assert result["success"] is False

    def test_handle_review_paper_rejects_outside_path(self, tmp_path: Path) -> None:
        """A client-supplied absolute path must not be read from anywhere."""
        server = self._server(tmp_path)
        outside = tmp_path.parent / "secret.md"
        outside.write_text("private", encoding="utf-8")

        result = asyncio.run(
            server.handle_tool_call("review_paper", {"paper_path": str(outside)})
        )

        assert result["success"] is False
        assert "must live under" in result["error"]

    def test_handle_get_paper_missing(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        result = asyncio.run(server.handle_tool_call("get_paper", {"run_id": "missing"}))
        assert result["success"] is False

    def test_handle_get_paper_found(self, tmp_path: Path) -> None:
        server = self._server(tmp_path)
        stage_dir = tmp_path / "artifacts" / "rc-paper" / "stage-22"
        stage_dir.mkdir(parents=True)
        (stage_dir / "paper.tex").write_text(
            "\\documentclass{article}", encoding="utf-8"
        )

        result = asyncio.run(
            server.handle_tool_call(
                "get_paper", {"run_id": "rc-paper", "format": "latex"}
            )
        )

        assert result["success"] is True
        assert "documentclass" in result["content"]


# ══════════════════════════════════════════════════════════════════
# MCP JSON-RPC protocol tests
# ══════════════════════════════════════════════════════════════════


def _serve(
    server: ResearchClawMCPServer, messages: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    stdin = io.BytesIO(b"".join(json.dumps(m).encode() + b"\n" for m in messages))
    stdout = io.StringIO()

    async def _run() -> None:
        await server.serve(StdioTransport(reader=stdin, writer=stdout))

    asyncio.run(_run())
    return [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]


class TestMCPProtocol:
    def test_initialize_and_tools_list(self) -> None:
        responses = _serve(
            ResearchClawMCPServer(),
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2024-11-05"},
                },
                {"jsonrpc": "2.0", "method": "notifications/initialized"},
                {"jsonrpc": "2.0", "id": 2, "method": "tools/list"},
            ],
        )

        assert len(responses) == 2  # notifications produce no response
        assert responses[0]["result"]["serverInfo"]["name"] == "researchclaw"
        assert responses[0]["result"]["protocolVersion"] == "2024-11-05"
        assert any(
            tool["name"] == "run_pipeline" for tool in responses[1]["result"]["tools"]
        )

    def test_tools_call_wraps_result_as_content(self, tmp_path: Path) -> None:
        server = ResearchClawMCPServer(artifacts_root=tmp_path / "artifacts")
        responses = _serve(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {
                        "name": "get_paper",
                        "arguments": {"run_id": "missing"},
                    },
                }
            ],
        )

        result = responses[0]["result"]
        assert result["isError"] is True
        payload = json.loads(result["content"][0]["text"])
        assert payload["success"] is False

    def test_unknown_method_returns_jsonrpc_error(self) -> None:
        responses = _serve(
            ResearchClawMCPServer(),
            [{"jsonrpc": "2.0", "id": 3, "method": "does/not/exist"}],
        )
        assert responses[0]["error"]["code"] == -32601

    def test_malformed_line_returns_parse_error(self) -> None:
        stdin = io.BytesIO(b"not json\n")
        stdout = io.StringIO()

        async def _run() -> None:
            await ResearchClawMCPServer().serve(
                StdioTransport(reader=stdin, writer=stdout)
            )

        asyncio.run(_run())
        response = json.loads(stdout.getvalue().strip())
        assert response["error"]["code"] == -32700

    def test_prints_do_not_corrupt_the_protocol_stream(self, capsys) -> None:
        server = ResearchClawMCPServer()
        original = server.handle_tool_call

        async def noisy(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
            print("pipeline progress line")
            return await original(name, arguments)

        server.handle_tool_call = noisy  # type: ignore[method-assign]
        responses = _serve(
            server,
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "health_check", "arguments": {}},
                }
            ],
        )

        assert len(responses) == 1
        assert responses[0]["id"] == 1
        assert "pipeline progress line" in capsys.readouterr().err

    def test_non_object_params_get_invalid_params(self) -> None:
        responses = _serve(
            ResearchClawMCPServer(),
            [
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": [1],
                },
                {"jsonrpc": "2.0", "id": 2, "method": "ping"},
            ],
        )

        assert responses[0]["error"]["code"] == -32602
        assert responses[1]["id"] == 2
        assert responses[1]["result"] == {}

    def test_to_stage_full_is_accepted(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        server = TestMCPServer._server(tmp_path)
        monkeypatch.setattr(
            ResearchClawMCPServer,
            "_run_pipeline_sync",
            staticmethod(lambda *args, **kwargs: None),
        )

        result = asyncio.run(
            server.handle_tool_call(
                "run_pipeline", {"topic": "x", "to_stage": "FULL"}
            )
        )

        assert result["success"] is True
        assert result["to_stage"] == "full"

    def test_invalid_to_stage_leaves_no_artifacts(self, tmp_path: Path) -> None:
        server = ResearchClawMCPServer(artifacts_root=tmp_path / "artifacts")

        result = asyncio.run(
            server.handle_tool_call(
                "run_pipeline", {"topic": "x", "to_stage": "NOPE"}
            )
        )

        assert result["success"] is False
        assert not (tmp_path / "artifacts").exists()

    def test_explicit_missing_config_does_not_fall_back(self, tmp_path: Path) -> None:
        server = ResearchClawMCPServer(artifacts_root=tmp_path / "artifacts")

        with pytest.raises(FileNotFoundError, match="Config file not found"):
            server._load_config(str(tmp_path / "typo.yaml"))

    def test_stage_traversal_is_rejected(self, tmp_path: Path) -> None:
        run_dir = tmp_path / "artifacts" / "rc-x"
        run_dir.mkdir(parents=True)
        (tmp_path / "evil").mkdir()
        (tmp_path / "evil" / "results.json").write_text("{}", encoding="utf-8")
        server = ResearchClawMCPServer(artifacts_root=tmp_path / "artifacts")

        result = asyncio.run(
            server.handle_tool_call(
                "get_experiment_results",
                {"run_id": "rc-x", "stage": "/../../../evil"},
            )
        )

        assert result["success"] is False
        assert "Invalid stage" in result["error"]

    def test_serve_exits_on_eof(self) -> None:
        server = ResearchClawMCPServer()

        async def _run() -> None:
            transport = StdioTransport(reader=io.BytesIO(b""), writer=io.StringIO())
            await server.serve(transport)
            assert not server.is_running

        asyncio.run(_run())


# ══════════════════════════════════════════════════════════════════
# Stdio transport framing
# ══════════════════════════════════════════════════════════════════


class TestStdioTransport:
    def test_send_uses_newline_json_not_content_length(self) -> None:
        writer = io.StringIO()
        transport = StdioTransport(reader=io.BytesIO(b""), writer=writer)

        async def _run() -> None:
            await transport.start()
            await transport.send({"jsonrpc": "2.0", "id": 1, "result": {}})

        asyncio.run(_run())
        output = writer.getvalue()
        assert "Content-Length" not in output
        assert output.endswith("\n")
        assert json.loads(output)["id"] == 1

    def test_receive_skips_blank_lines(self) -> None:
        reader = io.BytesIO(b'\n\n{"jsonrpc": "2.0", "id": 2}\n')
        transport = StdioTransport(reader=reader, writer=io.StringIO())

        async def _run() -> dict[str, Any]:
            await transport.start()
            return await transport.receive()

        assert asyncio.run(_run())["id"] == 2
# ══════════════════════════════════════════════════════════════════
# MCP Client tests
# ══════════════════════════════════════════════════════════════════


class TestMCPClient:
    def test_init(self) -> None:
        client = MCPClient("http://localhost:3000")
        assert client.uri == "http://localhost:3000"
        assert not client.is_connected

    def test_connect_disconnect(self) -> None:
        client = MCPClient("http://localhost:3000")

        async def _run() -> None:
            await client.connect()
            assert client.is_connected
            await client.disconnect()
            assert not client.is_connected

        asyncio.run(_run())

    def test_list_tools_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.list_tools())

    def test_call_tool_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.call_tool("test", {}))

    def test_list_resources_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.list_resources())

    def test_read_resource_not_connected(self) -> None:
        client = MCPClient("http://localhost:3000")
        with pytest.raises(ConnectionError):
            asyncio.run(client.read_resource("test://resource"))

    def test_list_tools_connected(self) -> None:
        client = MCPClient("http://localhost:3000")

        async def _run() -> list:
            await client.connect()
            return await client.list_tools()

        tools = asyncio.run(_run())
        assert isinstance(tools, list)

    def test_tools_cached(self) -> None:
        client = MCPClient("http://localhost:3000")

        async def _run() -> tuple:
            await client.connect()
            t1 = await client.list_tools()
            t2 = await client.list_tools()
            return t1, t2

        t1, t2 = asyncio.run(_run())
        assert t1 is t2


# ══════════════════════════════════════════════════════════════════
# MCP Server Registry tests
# ══════════════════════════════════════════════════════════════════


class TestMCPServerRegistry:
    def test_register_and_list(self) -> None:
        async def _run() -> list:
            reg = MCPServerRegistry()
            await reg.register("test", "http://localhost:3000")
            return reg.list_all()

        servers = asyncio.run(_run())
        assert len(servers) == 1
        assert servers[0]["name"] == "test"
        assert servers[0]["connected"] is True

    def test_unregister(self) -> None:
        async def _run() -> int:
            reg = MCPServerRegistry()
            await reg.register("test", "http://localhost:3000")
            await reg.unregister("test")
            return reg.count

        count = asyncio.run(_run())
        assert count == 0

    def test_get(self) -> None:
        async def _run() -> MCPClient | None:
            reg = MCPServerRegistry()
            await reg.register("test", "http://localhost:3000")
            return reg.get("test")

        client = asyncio.run(_run())
        assert client is not None
        assert client.is_connected

    def test_get_missing(self) -> None:
        reg = MCPServerRegistry()
        assert reg.get("nonexistent") is None

    def test_close_all(self) -> None:
        async def _run() -> int:
            reg = MCPServerRegistry()
            await reg.register("a", "http://a:3000")
            await reg.register("b", "http://b:3000")
            await reg.close_all()
            return reg.count

        count = asyncio.run(_run())
        assert count == 0


# ══════════════════════════════════════════════════════════════════
# Transport tests
# ══════════════════════════════════════════════════════════════════


class TestSSETransport:
    def test_start_stop(self) -> None:
        transport = SSETransport(port=9999)

        async def _run() -> None:
            await transport.start()
            assert transport._running is True
            await transport.close()
            assert transport._running is False

        asyncio.run(_run())

    def test_receive_not_implemented(self) -> None:
        transport = SSETransport()
        with pytest.raises(NotImplementedError):
            asyncio.run(transport.receive())
