"""Tests for ``research.min_year`` — the Stage-4 publication-year floor."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stages import StageStatus


def _config(tmp_path: Path, **research: Any) -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "min-year-test", "mode": "docs-first"},
            "research": {"topic": "transformer architectures", **research},
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


class TestMinYearConfig:
    def test_default_is_2020(self, tmp_path: Path) -> None:
        assert _config(tmp_path).research.min_year == 2020

    def test_explicit_year_is_parsed(self, tmp_path: Path) -> None:
        assert _config(tmp_path, min_year=2015).research.min_year == 2015

    def test_zero_disables_the_filter(self, tmp_path: Path) -> None:
        assert _config(tmp_path, min_year=0).research.min_year == 0

    def test_negative_year_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid research.min_year"):
            _config(tmp_path, min_year=-1)

    def test_future_year_is_rejected(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid research.min_year"):
            _config(tmp_path, min_year=datetime.now().year + 5)


class TestSearchStrategyYearMin:
    @pytest.mark.parametrize("min_year", [2020, 2015, 0])
    def test_stage3_writes_config_value(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, min_year: int
    ) -> None:
        from researchclaw.pipeline.stage_impls import _literature as lit

        config = _config(tmp_path, min_year=min_year)
        run_dir = tmp_path / "run"
        stage_dir = run_dir / "stage-03"
        stage_dir.mkdir(parents=True)

        plan = {
            "search_strategies": [
                {"queries": ["transformer attention mechanism", "self-attention survey"]}
            ],
            "sources": [
                {
                    "id": "openalex",
                    "name": "OpenAlex",
                    "type": "api",
                    "url": "https://api.openalex.org",
                    "status": "active",
                    "query": "transformer",
                    "verified_at": "2026-01-01T00:00:00Z",
                }
            ],
        }
        monkeypatch.setattr(
            lit,
            "_chat_with_prompt",
            lambda *args, **kwargs: SimpleNamespace(content=json.dumps(plan)),
        )

        result = lit._execute_search_strategy(
            stage_dir, run_dir, config, AdapterBundle(), llm=object()
        )

        assert result.status == StageStatus.DONE
        meta = json.loads((stage_dir / "queries.json").read_text(encoding="utf-8"))
        assert meta["year_min"] == min_year
        assert meta["queries"]
