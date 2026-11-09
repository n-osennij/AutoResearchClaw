"""The checked-in prompt template must mirror the built-in bank exactly."""

from __future__ import annotations

from pathlib import Path

import yaml

from researchclaw.prompts.export import export_bank, write_default_prompts
from researchclaw.prompts.manager import PromptManager

_REPO_TEMPLATE = Path(__file__).parent.parent / "prompts.default.yaml"


class TestPromptsDefaultYaml:
    def test_matches_the_ml_bank(self) -> None:
        data = yaml.safe_load(_REPO_TEMPLATE.read_text(encoding="utf-8"))
        expected = export_bank("ml")

        assert data["stages"] == expected["stages"], (
            "prompts.default.yaml drifted from researchclaw/prompts/ml.py — "
            "regenerate it with `python scripts/export_prompts.py`"
        )
        assert data["blocks"] == expected["blocks"]
        assert data["sub_prompts"] == expected["sub_prompts"]

    def test_regeneration_round_trips(self, tmp_path: Path) -> None:
        target = write_default_prompts(tmp_path / "prompts.default.yaml", domain="ml")
        assert target.read_text(encoding="utf-8") == _REPO_TEMPLATE.read_text(
            encoding="utf-8"
        )

    def test_template_is_a_noop_override(self) -> None:
        baseline = PromptManager()
        overridden = PromptManager(_REPO_TEMPLATE)

        assert overridden._stages == baseline._stages
        assert overridden._blocks == baseline._blocks
        assert overridden._sub_prompts == baseline._sub_prompts

        rendered = overridden.for_stage("literature_screen", topic="t")
        assert rendered.user == baseline.for_stage("literature_screen", topic="t").user
