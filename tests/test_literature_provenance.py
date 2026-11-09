"""Provenance contract between Stage 5 (screen) and Stage 6 (cards).

The screen lets the model rewrite the shortlist, so evidence metadata
(``pmid``, ``is_preprint``, ``pub_types``, ``concern``, ``retracted``) has to be
re-applied from the candidate rows — a model that drops a field is not lying,
it is summarising. These tests pin the re-application, the count bookkeeping in
``screen_meta.json`` and the knowledge-card fallback when the model rewords a
title instead of echoing it.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from researchclaw.adapters import AdapterBundle
from researchclaw.config import RCConfig
from researchclaw.pipeline.stages import StageStatus


def _config(tmp_path: Path, **lit: Any) -> RCConfig:
    config = RCConfig.from_dict(
        {
            "project": {"name": "provenance-test", "mode": "docs-first"},
            "research": {"topic": "transformer attention mechanism"},
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
    return replace(config, web_search=replace(config.web_search, enabled=False))


def _rows() -> list[dict[str, Any]]:
    return [
        {
            "paper_id": "pmid:30000001",
            "title": "Attention improves sequence transduction.",
            "abstract": "transformer attention mechanism study",
            "year": 2021,
            "source": "europepmc",
            "pmid": "30000001",
            "is_preprint": True,
            "pub_types": ("Preprint",),
            "concern": "small cohort, n=12",
        },
        {
            "paper_id": "doi:10.1/peer",
            "title": "Peer-reviewed transformer benchmark",
            "abstract": "transformer attention mechanism benchmark",
            "year": 2020,
            "source": "openalex",
            "retracted": True,
        },
    ]


class TestTitleKeys:
    def test_model_restated_titles_still_match(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _row_keys

        base = {"paper_id": "", "title": "Attention improves sequence transduction."}
        variants = [
            "attention improves sequence transduction",
            "Attention improves sequence transduction",
            "ATTENTION IMPROVES SEQUENCE TRANSDUCTION.",
            "Attention improves sequence transduction, et al.",
            "Attention improves sequence transduction et al",
        ]
        keys = {tuple(_row_keys({**base, "title": v})) for v in variants}
        assert len(keys) == 1, keys

    def test_doi_key_is_case_insensitive(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _row_keys

        assert _row_keys({"doi": "10.1000/ABC"}) == _row_keys({"doi": "10.1000/abc"})


class TestReapplyProvenance:
    def test_fields_are_restored_and_unmatched_counted(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _reapply_provenance

        shortlist = [
            {"title": "attention improves sequence transduction"},  # dropped pmid etc.
            {"title": "a paper the model invented"},  # no source row
        ]
        merged, dropped, unmatched = _reapply_provenance(shortlist, _rows())
        assert dropped == 0
        assert unmatched == 1
        assert len(merged) == 2
        assert merged[0]["pmid"] == "30000001"
        assert merged[0]["is_preprint"] is True
        assert merged[0]["concern"] == "small cohort, n=12"

    def test_retracted_source_row_is_dropped(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _reapply_provenance

        shortlist = [{"title": "Peer-reviewed transformer benchmark"}]
        merged, dropped, _ = _reapply_provenance(shortlist, _rows())
        assert merged == []
        assert dropped == 1

    def test_model_found_retraction_is_not_erased(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _reapply_provenance

        shortlist = [
            {"title": "attention improves sequence transduction", "retracted": True}
        ]
        merged, dropped, _ = _reapply_provenance(shortlist, _rows())
        assert dropped == 1 and merged == []

    def test_missing_source_fields_do_not_clear_model_values(self) -> None:
        from researchclaw.pipeline.stage_impls._literature import _reapply_provenance

        sources = [{"title": "attention improves sequence transduction"}]
        shortlist = [{"title": "attention improves sequence transduction", "concern": "x"}]
        merged, _, _ = _reapply_provenance(shortlist, sources)
        assert merged[0]["concern"] == "x"


class TestScreeningMeta:
    def _run(self, tmp_path: Path, rows: list[dict[str, Any]], reply: Any) -> tuple[Path, Any]:
        from researchclaw.pipeline.stage_impls import _literature as lit

        run_dir = tmp_path / "run"
        stage_dir = run_dir / "stage-05"
        stage_dir.mkdir(parents=True)
        s4 = run_dir / "stage-04"
        s4.mkdir()
        (s4 / "candidates.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
        )

        def fake_chat(llm: Any, system: str, user: str, **kwargs: Any) -> Any:
            return SimpleNamespace(content=json.dumps({"shortlist": reply}))

        original = lit._chat_with_prompt
        lit._chat_with_prompt = fake_chat  # type: ignore[assignment]
        try:
            result = lit._execute_literature_screen(
                stage_dir, run_dir, _config(tmp_path, min_shortlist=1),
                AdapterBundle(), llm=object(),
            )
        finally:
            lit._chat_with_prompt = original  # type: ignore[assignment]
        meta = json.loads((stage_dir / "screen_meta.json").read_text())
        return stage_dir, (result, meta)

    def test_safety_valve_reports_the_rows_it_screened(self, tmp_path: Path) -> None:
        """Pre-filter drops everything → the valve screens raw rows, and the
        metadata must say how many, not 0."""
        rows = [
            {"paper_id": f"p{i}", "title": f"Unrelated botany {i}", "abstract": "x"}
            for i in range(4)
        ]
        _, (result, meta) = self._run(
            tmp_path, rows, [{"title": "Unrelated botany 0", "relevance_score": 0.9}]
        )
        assert result.status == StageStatus.DONE
        assert meta["safety_valve"] is True
        assert meta["candidates_screened"] == 4
        assert meta["candidates_after_prefilter"] == 4
        assert meta["candidates_seen"] == 4

    def test_retracted_rows_never_reach_the_shortlist(self, tmp_path: Path) -> None:
        rows = [
            {
                "paper_id": "doi:ret",
                "title": "transformer attention retracted study",
                "abstract": "transformer attention",
                "year": 2019,
                "retracted": True,
            },
            {
                "paper_id": "doi:ok",
                "title": "transformer attention valid study",
                "abstract": "transformer attention",
                "year": 2021,
            },
        ]
        stage_dir, (result, meta) = self._run(
            tmp_path,
            rows,
            [
                {"title": "transformer attention retracted study", "relevance_score": 0.9},
                {"title": "transformer attention valid study", "relevance_score": 0.9},
            ],
        )
        assert result.status == StageStatus.DONE
        kept = [
            json.loads(line)
            for line in (stage_dir / "shortlist.jsonl").read_text().splitlines()
            if line.strip()
        ]
        assert [row["title"] for row in kept] == ["transformer attention valid study"]
        assert meta["candidates_retracted_dropped"] == 1

    def test_shortlist_unmatched_is_recorded(self, tmp_path: Path) -> None:
        _, (_, meta) = self._run(
            tmp_path,
            [
                {
                    "paper_id": "p1",
                    "title": "transformer attention known paper",
                    "abstract": "transformer attention",
                    "year": 2021,
                }
            ],
            [{"title": "a title that shares nothing", "relevance_score": 0.9}],
        )
        assert meta["shortlist_unmatched"] == 1


class TestCardProvenance:
    def _run(self, tmp_path: Path, shortlist: list[dict[str, Any]], cards: list[dict[str, Any]]) -> Path:
        from researchclaw.pipeline.stage_impls import _literature as lit

        run_dir = tmp_path / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        s5 = run_dir / "stage-05"
        s5.mkdir()
        (s5 / "shortlist.jsonl").write_text(
            "\n".join(json.dumps(row) for row in shortlist), encoding="utf-8"
        )
        stage_dir = run_dir / "stage-06"
        stage_dir.mkdir()

        def fake_chat(llm: Any, system: str, user: str, **kwargs: Any) -> Any:
            return SimpleNamespace(content=json.dumps({"cards": cards}))

        original = lit._chat_with_prompt
        lit._chat_with_prompt = fake_chat  # type: ignore[assignment]
        try:
            result = lit._execute_knowledge_extract(
                stage_dir, run_dir, _config(tmp_path), AdapterBundle(), llm=object()
            )
        finally:
            lit._chat_with_prompt = original  # type: ignore[assignment]
        assert result.status == StageStatus.DONE
        return stage_dir

    def test_reworded_title_falls_back_to_position(self, tmp_path: Path) -> None:
        shortlist = [
            {
                "paper_id": "pmid:42",
                "title": "Original title one",
                "pmid": "42",
                "is_preprint": True,
            },
            {"paper_id": "pmid:43", "title": "Original title two", "pmid": "43"},
        ]
        cards = [
            {"card_id": "card-1", "title": "A reworded paraphrase"},
            {"card_id": "card-2", "title": "Another paraphrase"},
        ]
        stage_dir = self._run(tmp_path, shortlist, cards)
        first = (stage_dir / "cards" / "card-1.md").read_text(encoding="utf-8")
        second = (stage_dir / "cards" / "card-2.md").read_text(encoding="utf-8")
        assert "pmid: 42" in first
        assert "is_preprint: True" in first
        assert "pmid: 43" in second

    def test_matching_title_wins_over_position(self, tmp_path: Path) -> None:
        shortlist = [
            {"paper_id": "pmid:42", "title": "First paper", "pmid": "42"},
            {"paper_id": "pmid:43", "title": "Second paper", "pmid": "43"},
        ]
        # card-1 describes the *second* paper: the key must beat the slot.
        cards = [{"card_id": "card-1", "title": "Second paper"}]
        stage_dir = self._run(tmp_path, shortlist, cards)
        body = (stage_dir / "cards" / "card-1.md").read_text(encoding="utf-8")
        assert "pmid: 43" in body

    def test_no_provenance_block_when_nothing_matches(self, tmp_path: Path) -> None:
        shortlist = [{"paper_id": "pmid:42", "title": "First paper", "pmid": "42"}]
        cards = [
            {"card_id": "card-1", "title": "One"},
            {"card_id": "card-2", "title": "Two"},
            {"card_id": "card-3", "title": "Three"},
        ]
        stage_dir = self._run(tmp_path, shortlist, cards)
        assert "## Provenance" in (stage_dir / "cards" / "card-1.md").read_text()
        # Beyond the shortlist length there is no position to borrow from.
        assert "## Provenance" not in (stage_dir / "cards" / "card-2.md").read_text()
        assert "## Provenance" not in (stage_dir / "cards" / "card-3.md").read_text()

    def test_retracted_false_is_not_written(self, tmp_path: Path) -> None:
        shortlist = [{"paper_id": "pmid:42", "title": "First paper", "pmid": "42"}]
        stage_dir = self._run(
            tmp_path, shortlist, [{"card_id": "card-1", "title": "First paper"}]
        )
        body = (stage_dir / "cards" / "card-1.md").read_text(encoding="utf-8")
        assert "retracted" not in body
        assert "pmid: 42" in body
