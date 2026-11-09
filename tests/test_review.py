"""Tests for the standalone literature-review renderer."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from researchclaw.review import generate_literature_review, write_literature_review


def _run_dir(tmp_path: Path) -> Path:
    run_dir = tmp_path / "run"
    (run_dir / "stage-01").mkdir(parents=True)
    (run_dir / "stage-03").mkdir()
    (run_dir / "stage-04").mkdir()
    (run_dir / "stage-05").mkdir()
    (run_dir / "stage-06" / "cards").mkdir(parents=True)
    (run_dir / "stage-07").mkdir()

    (run_dir / "stage-01" / "goal.md").write_text(
        "# Research Goal\n\n## Topic\nKV cache compression\n\n"
        "## Scope\nMake long-context inference cheaper.\n",
        encoding="utf-8",
    )
    (run_dir / "stage-03" / "queries.json").write_text(
        json.dumps({"queries": ["kv cache", "long context"]}), encoding="utf-8"
    )
    (run_dir / "stage-04" / "search_meta.json").write_text(
        json.dumps(
            {
                "total_candidates": 42,
                "year_min": 2020,
                "year_filter": {
                    "min_year": 2020,
                    "active": True,
                    "hybrid": True,
                    "hint": "primary pass excludes before 2020",
                },
                "hybrid_search": {
                    "enabled": True,
                    "foundational_added": 3,
                    "foundational_max_papers": 40,
                },
            }
        ),
        encoding="utf-8",
    )
    rows = [
        {
            "title": "KV Cache Compression",
            "year": 2023,
            "venue": "NeurIPS",
            "citation_count": 120,
            "relevance_score": 0.9,
            "source": "openalex",
            "url": "https://example.org/1",
        },
        {
            "title": "Cache | Pipes",  # pipe must be escaped in the table
            "year": 2018,
            "venue": "ICML",
            "citation_count": 900,
            "source": "arxiv",
        },
    ]
    (run_dir / "stage-05" / "shortlist.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows), encoding="utf-8"
    )
    (run_dir / "stage-05" / "screen_meta.json").write_text(
        json.dumps({"candidates_screened": 30}), encoding="utf-8"
    )
    (run_dir / "stage-06" / "cards" / "card-1.md").write_text(
        "# Quantization card\n\n## Method\n4-bit quantization.\n", encoding="utf-8"
    )
    (run_dir / "stage-07" / "synthesis.md").write_text(
        "## Cluster Overview\n\nKV cache dominates memory.\n", encoding="utf-8"
    )
    return run_dir


class TestLiteratureReview:
    def test_renders_all_sections(self, tmp_path: Path) -> None:
        review = generate_literature_review(_run_dir(tmp_path))

        assert review.startswith("# Literature review: KV cache compression")
        assert "## Scope\n\nMake long-context inference cheaper." in review
        assert "## Search strategy" in review
        assert "**Hybrid pass:** on" in review
        assert "## Screening" in review
        assert "- **Collected candidates:** 42" in review
        assert "| 1 | KV Cache Compression | 2023 | NeurIPS |  | 120 | 0.9 | openalex |" in review
        assert "Cache \\| Pipes" in review
        assert "### Quantization card" in review
        assert "#### Method" in review
        assert "### Cluster Overview" in review
        assert "## References" in review
        assert "1. KV Cache Compression. *NeurIPS*. 2023." in review

    def test_topic_override(self, tmp_path: Path) -> None:
        review = generate_literature_review(
            _run_dir(tmp_path), topic="Custom topic"
        )
        assert review.startswith("# Literature review: Custom topic")

    def test_survives_missing_artifacts(self, tmp_path: Path) -> None:
        empty = tmp_path / "empty-run"
        empty.mkdir()
        review = generate_literature_review(empty)
        assert "# Literature review:" in review
        assert "## References" not in review

    def test_write_helper(self, tmp_path: Path) -> None:
        run_dir = _run_dir(tmp_path)
        path = write_literature_review(run_dir)
        assert path == run_dir / "review.md"
        assert "## Synthesis and gaps" in path.read_text(encoding="utf-8")

    def test_cli_renders_from_run_dir(self, tmp_path: Path) -> None:
        from researchclaw.cli import main

        run_dir = _run_dir(tmp_path)
        exit_code = main(["review", "--run-dir", str(run_dir)])

        assert exit_code == 0
        review = (run_dir / "review.md").read_text(encoding="utf-8")
        assert review.startswith("# Literature review: KV cache compression")

    def test_cli_rejects_missing_run_dir(self, tmp_path: Path) -> None:
        from researchclaw.cli import main

        exit_code = main(["review", "--run-dir", str(tmp_path / "nope")])
        assert exit_code == 1

    def test_warns_when_candidates_are_llm_synthesized(self, tmp_path: Path) -> None:
        run_dir = _run_dir(tmp_path)
        meta = json.loads(
            (run_dir / "stage-04" / "search_meta.json").read_text(encoding="utf-8")
        )
        meta["real_search"] = False
        (run_dir / "stage-04" / "search_meta.json").write_text(
            json.dumps(meta), encoding="utf-8"
        )

        review = generate_literature_review(run_dir)

        assert "## Warnings" in review
        assert "Candidates were synthesized by the LLM" in review

    def test_no_candidates_uses_search_meta_total(self, tmp_path: Path) -> None:
        run_dir = _run_dir(tmp_path)
        review = generate_literature_review(run_dir)
        assert "- **Collected candidates:** 42" in review
        assert "- **Passed keyword prefilter:** 30" in review

class TestEvidenceTypeColumn:
    def test_preprint_and_retraction_are_visible(self, tmp_path: Path) -> None:
        run = _run_dir(tmp_path)
        shortlist = run / "stage-05" / "shortlist.jsonl"
        rows = [
            json.loads(line)
            for line in shortlist.read_text(encoding="utf-8").splitlines()
        ]
        rows[0]["is_preprint"] = True
        rows[0]["retracted"] = True
        rows[0]["pub_types"] = ["Preprint"]
        rows[1]["pub_types"] = ["Randomized Controlled Trial", "Journal Article"]
        body = ("\n".join(json.dumps(r) for r in rows)) + "\n"
        shortlist.write_text(body, encoding="utf-8")
        review = generate_literature_review(run)
        assert "preprint" in review
        assert "retracted" in review
        assert "Randomized Controlled Trial" in review
