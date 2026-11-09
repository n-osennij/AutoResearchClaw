"""Tests for the Stage-4 year reminder and the configurable screening budget."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.literature.models import Paper
from researchclaw.pipeline.stages import StageStatus


def _config(tmp_path: Path, **lit: Any) -> RCConfig:
    research: dict[str, Any] = {"topic": "transformer attention mechanism"}
    if "min_year" in lit:
        research["min_year"] = lit.pop("min_year")
    config = RCConfig.from_dict(
        {
            "project": {"name": "screening-test", "mode": "docs-first"},
            "research": research,
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"root": str(tmp_path / "kb")},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline",
            },
            "literature_search": lit,
        },
        project_root=tmp_path,
        check_paths=False,
    )
    return replace(
        config,
        web_search=replace(config.web_search, enabled=False),
    )


class TestScreeningConfig:
    def test_defaults(self, tmp_path: Path) -> None:
        lit = _config(tmp_path).literature_search
        assert lit.max_screen_chars == 100_000
        assert lit.max_abstract_chars == 800
        assert lit.min_shortlist == 15

    def test_overrides_are_parsed(self, tmp_path: Path) -> None:
        lit = _config(
            tmp_path,
            max_screen_chars=400_000,
            max_abstract_chars=1500,
            min_shortlist=30,
        ).literature_search
        assert lit.max_screen_chars == 400_000
        assert lit.max_abstract_chars == 1500
        assert lit.min_shortlist == 30

    def test_values_are_clamped(self, tmp_path: Path) -> None:
        lit = _config(
            tmp_path,
            max_screen_chars=1,
            max_abstract_chars=1,
            min_shortlist=0,
        ).literature_search
        assert lit.max_screen_chars == 10_000
        assert lit.max_abstract_chars == 100
        assert lit.min_shortlist == 1


class TestScreeningBudget:
    def test_stage5_respects_configured_caps(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline.stage_impls import _literature as lit

        run_dir = tmp_path / "run"
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True)
        (run_dir / "stage-04").mkdir()
        rows = [
            {
                "paper_id": f"p{i}",
                "title": f"Transformer attention paper {i}",
                "abstract": "transformer attention " * 60,  # ~1200 chars
                "year": 2021,
                "source": "openalex",
            }
            for i in range(20)
        ]
        (run_dir / "stage-04" / "candidates.jsonl").write_text(
            "\n".join(json.dumps(r) for r in rows), encoding="utf-8"
        )

        config = _config(
            tmp_path,
            max_screen_chars=5000,
            max_abstract_chars=100,
            min_shortlist=3,
        )
        captured: dict[str, Any] = {}

        def fake_chat(llm: Any, system: str, user: str, **kwargs: Any) -> Any:
            captured["user"] = user
            shortlist = [
                {"title": f"Transformer attention paper {i}", "relevance_score": 0.9}
                for i in range(3)
            ]
            return SimpleNamespace(content=json.dumps({"shortlist": shortlist}))

        monkeypatch.setattr(lit, "_chat_with_prompt", fake_chat)

        result = lit._execute_literature_screen(
            stage_dir, run_dir, config, AdapterBundle(), llm=object()
        )

        assert result.status == StageStatus.DONE
        prompt = captured["user"]
        candidate_block = prompt.split("Topic:", 1)[0]
        # The screening blob respects the configured char cap …
        assert len(candidate_block) <= 5000
        # … and abstracts are cut to max_abstract_chars before it.
        for line in candidate_block.splitlines():
            if line.startswith("{"):
                row = json.loads(line)
                assert len(row["abstract"]) <= 100 + len("...")
        # 20 rows would not fit, so the model saw fewer than all of them.
        assert candidate_block.count('"paper_id"') < 20
        assert (stage_dir / "shortlist.jsonl").exists()


class TestYearReminder:
    @pytest.mark.parametrize(("min_year", "active"), [(2015, True), (0, False)])
    def test_stage4_search_meta_records_year_filter(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_year: int, active: bool
    ) -> None:
        from researchclaw.pipeline.stage_impls import _literature as lit

        run_dir = tmp_path / "run"
        (run_dir / "stage-03").mkdir(parents=True)
        (run_dir / "stage-03" / "queries.json").write_text(
            json.dumps({"queries": ["transformer attention"], "year_min": min_year}),
            encoding="utf-8",
        )
        stage_dir = run_dir / "stage-04"
        stage_dir.mkdir()

        monkeypatch.setattr(
            "researchclaw.literature.search.search_papers_multi_query",
            lambda *a, **k: [
                Paper(paper_id="p1", title="T", year=2016, source="openalex")
            ],
        )
        monkeypatch.setattr("researchclaw.data.load_seminal_papers", lambda _t: [])

        result = lit._execute_literature_collect(
            stage_dir, run_dir, _config(tmp_path), AdapterBundle(), llm=None
        )

        assert result.status == StageStatus.DONE
        meta = json.loads((stage_dir / "search_meta.json").read_text(encoding="utf-8"))
        assert meta["year_min"] == min_year
        assert meta["year_filter"]["active"] is active
        if active:
            assert str(min_year) in meta["year_filter"]["hint"]
        else:
            assert meta["year_filter"]["hint"] == "no publication-year cutoff"
