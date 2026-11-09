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
            for i in range(60)
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
        # The candidates are the JSON lines appended at the end of the prompt
        # (the old split at "Topic:" measured the instruction prefix instead).
        candidate_lines = [
            line for line in prompt.splitlines() if line.startswith("{")
        ]
        shown = [json.loads(line) for line in candidate_lines]
        # max_screen_chars=5000 is clamped to the 10 000 floor; 60 rows of
        # ~250 chars overflow it, so the model must see fewer than all rows.
        assert len(candidate_lines) < 60
        assert len(candidate_lines) > 0
        assert sum(len(line) for line in candidate_lines) <= 10_000
        # … and abstracts are cut to max_abstract_chars before it.
        assert all(len(row["abstract"]) <= 100 + len("...") for row in shown)
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


class TestScreenerRetry:
    """A reasoning model can answer "" with finish_reason="length"."""

    def _setup(self, tmp_path: Path, rows: int = 5) -> tuple[Path, Path, RCConfig]:
        from researchclaw.pipeline.stage_impls import _literature as lit  # noqa: F401

        run_dir = tmp_path / "run"
        (run_dir / "stage-04").mkdir(parents=True)
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir()
        candidates = [
            {"paper_id": f"pmid:{i}", "title": f"Transformer attention paper {i}", "pmid": str(i)}
            for i in range(rows)
        ]
        (run_dir / "stage-04" / "candidates.jsonl").write_text(
            "\n".join(json.dumps(r) for r in candidates), encoding="utf-8"
        )
        return stage_dir, run_dir, _config(tmp_path, min_shortlist=2)

    def test_empty_reply_is_retried_with_a_bigger_budget(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline.stage_impls import _literature as lit

        stage_dir, run_dir, config = self._setup(tmp_path)
        budgets: list[int] = []

        def fake_chat(llm: Any, system: str, user: str, **kwargs: Any) -> Any:
            budgets.append(kwargs.get("max_tokens") or 0)
            if len(budgets) == 1:  # reasoning ate the whole budget
                return SimpleNamespace(content="", finish_reason="length", truncated=True)
            return SimpleNamespace(
                content=json.dumps(
                    {"shortlist": [{"index": 1, "relevance_score": 0.9}]}
                ),
                finish_reason="stop",
                truncated=False,
            )

        monkeypatch.setattr(lit, "_chat_with_prompt", fake_chat)
        result = lit._execute_literature_screen(
            stage_dir, run_dir, config, AdapterBundle(), llm=object()
        )
        assert result.status == StageStatus.DONE
        assert len(budgets) == 2 and budgets[1] > budgets[0]
        meta = json.loads((stage_dir / "screen_meta.json").read_text(encoding="utf-8"))
        assert meta["parse_failed"] is False
        shortlist = [
            json.loads(line)
            for line in (stage_dir / "shortlist.jsonl").read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert shortlist[0]["pmid"] == "0"

    def test_unparsable_reply_keeps_the_raw_response(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from researchclaw.pipeline.stage_impls import _literature as lit

        stage_dir, run_dir, config = self._setup(tmp_path)

        def fake_chat(llm: Any, system: str, user: str, **kwargs: Any) -> Any:
            return SimpleNamespace(content="Sure! Here is my screen:", finish_reason="stop")

        monkeypatch.setattr(lit, "_chat_with_prompt", fake_chat)
        lit._execute_literature_screen(stage_dir, run_dir, config, AdapterBundle(), llm=object())
        raw = (stage_dir / "screen_raw.txt").read_text(encoding="utf-8")
        assert raw.startswith("Sure! Here is my screen")
