"""Tests for the two-pass (hybrid) literature search and cache-policy tracking."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.literature.models import Paper
from researchclaw.pipeline.stages import StageStatus


def _config(tmp_path: Path, **research: Any) -> RCConfig:
    config = RCConfig.from_dict(
        {
            "project": {"name": "hybrid-test", "mode": "docs-first"},
            "research": {"topic": "transformer attention mechanism", **research},
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
    return replace(config, web_search=replace(config.web_search, enabled=False))


def _paper(paper_id: str, year: int, **kwargs: Any) -> Paper:
    return Paper(paper_id=paper_id, title=f"Paper {paper_id}", year=year,
                 source="openalex", **kwargs)


def _run_stage4(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    config: RCConfig,
    search,
    *,
    year_min: int = 2020,
) -> tuple[Any, dict[str, Any], Path]:
    from researchclaw.pipeline.stage_impls import _literature as lit

    run_dir = tmp_path / "run"
    (run_dir / "stage-03").mkdir(parents=True)
    (run_dir / "stage-03" / "queries.json").write_text(
        json.dumps({"queries": ["transformer attention"], "year_min": year_min}),
        encoding="utf-8",
    )
    stage_dir = run_dir / "stage-04"
    stage_dir.mkdir()

    calls: list[dict[str, Any]] = []

    def fake_search(queries: list[str], **kwargs: Any) -> list[Paper]:
        calls.append({"queries": queries, **kwargs})
        return search(int(kwargs.get("year_min", 0)))

    monkeypatch.setattr(
        "researchclaw.literature.search.search_papers_multi_query", fake_search
    )
    monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda _t: [])

    result = lit._execute_literature_collect(
        stage_dir, run_dir, config, AdapterBundle(), llm=None
    )
    meta = json.loads((stage_dir / "search_meta.json").read_text(encoding="utf-8"))
    return result, meta, stage_dir


class TestHybridSearch:
    def test_second_pass_adds_foundational_papers(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def search(year_min: int) -> list[Paper]:
            if year_min > 0:  # recent pass
                return [_paper("recent-1", 2022, doi="10.1/c"), _paper("recent-2", 2024)]
            # foundational pass: old + duplicates + unknown year
            return [
                _paper("old-1", 2015, doi="10.1/a", citation_count=500),
                _paper("old-2", 2016, doi="10.1/b"),
                _paper("dup-recent", 2022, doi="10.1/c"),
                _paper("no-year", 0),
            ]

        result, meta, stage_dir = _run_stage4(
            tmp_path, monkeypatch, _config(tmp_path), search
        )

        assert result.status == StageStatus.DONE
        rows = [
            json.loads(line)
            for line in (stage_dir / "candidates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        bands = {row["paper_id"]: row.get("year_band") for row in rows}
        assert bands["recent-1"] == "recent"
        assert bands["old-1"] == "foundational"
        assert bands["old-2"] == "foundational"
        # the duplicate from the foundational pass is not added twice
        assert sum(1 for row in rows if row["paper_id"] == "recent-1") == 1
        # unknown-year papers are not treated as foundations
        assert "no-year" not in bands
        assert meta["hybrid_search"]["enabled"] is True
        assert meta["hybrid_search"]["foundational_added"] == 2

    def test_foundational_cap(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def search(year_min: int) -> list[Paper]:
            if year_min > 0:
                return [_paper("recent", 2023)]
            return [_paper(f"old-{i}", 2010 + i) for i in range(10)]

        config = _config(
            tmp_path, hybrid_search=True, foundational_max_papers=3
        )
        _, meta, _ = _run_stage4(tmp_path, monkeypatch, config, search)
        assert meta["hybrid_search"]["foundational_added"] == 3

    def test_hybrid_disabled_runs_one_pass(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def search(year_min: int) -> list[Paper]:
            calls.append(year_min)
            return [_paper("p", 2022)]

        config = _config(tmp_path, hybrid_search=False)
        _, meta, _ = _run_stage4(tmp_path, monkeypatch, config, search)

        assert calls == [2020]
        assert meta["hybrid_search"]["enabled"] is False

    def test_no_floor_disables_hybrid(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[int] = []

        def search(year_min: int) -> list[Paper]:
            calls.append(year_min)
            return [_paper("p", 2015)]

        _, meta, _ = _run_stage4(
            tmp_path, monkeypatch, _config(tmp_path), search, year_min=0
        )

        assert calls == [0]
        assert meta["hybrid_search"]["enabled"] is False
        assert meta["year_filter"]["active"] is False

    def test_cache_policy_is_recorded(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def search(year_min: int) -> list[Paper]:
            return [_paper("p", 2022, doi="10.1/x")]

        config = _config(tmp_path, foundational_max_papers=0)
        _, meta, _ = _run_stage4(tmp_path, monkeypatch, config, search)

        from researchclaw.literature.cache import inspect_search_policy

        info = inspect_search_policy(min_year=2020, hybrid=True)
        assert info["policy"]["min_year"] == 2020
        assert info["policy"]["hybrid"] is True
        assert meta["cache"]["previous_policy"] is None
        assert meta["cache"]["legacy_entries"] is False
