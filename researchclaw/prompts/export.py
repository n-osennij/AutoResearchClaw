"""Export a prompt bank to the YAML override format.

``prompts.default.yaml`` is generated from this module, so the checked-in
template can never drift from the built-in bank: copying it and pointing
``prompts.custom_file`` at the copy is a no-op until the user edits something.

Only the parts ``PromptManager._load_overrides`` understands are exported —
``stages``, ``blocks`` and ``sub_prompts``. Debate role prompts are not
overridable through YAML and therefore not exported.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from researchclaw.prompts.manager import _load_bank
from researchclaw.prompts.shared import _DEFAULT_BLOCKS, _DEFAULT_SUB_PROMPTS

_HEADER = """\
# =============================================================================
# AutoResearchClaw — prompt override template ({domain} bank)
# =============================================================================
# GENERATED FILE — do not edit by hand if you want to stay in sync with the
# built-in prompts. Regenerate with:
#
#   python scripts/export_prompts.py --domain {domain}
#
# It mirrors the built-in bank exactly, so copying it and pointing config at the
# copy changes nothing:
#
#   prompts:
#     custom_file: "my_prompts.yaml"
#
# Edit only the stages/blocks you care about. Template variables use
# {{var_name}} syntax; unknown stage names are warned about and ignored.
# A stage entry replaces the whole `system` / `user` string of that stage.
# =============================================================================

"""


class _PromptDumper(yaml.SafeDumper):
    """SafeDumper that writes multi-line strings as readable block scalars."""


def _represent_str(dumper: yaml.SafeDumper, data: str) -> yaml.ScalarNode:
    style = "|" if "\n" in data else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", data, style=style)


_PromptDumper.add_representer(str, _represent_str)


def export_bank(domain: str = "ml") -> dict[str, Any]:
    """Return the YAML structure for *domain*'s prompt bank.

    Unknown domains fall back to the ML bank, matching ``PromptManager``.
    """
    stages, _debate_hyp, _debate_ana = _load_bank(domain)
    return {
        "blocks": {name: text for name, text in _DEFAULT_BLOCKS.items()},
        "stages": {name: dict(entry) for name, entry in stages.items()},
        "sub_prompts": {
            name: dict(entry) for name, entry in _DEFAULT_SUB_PROMPTS.items()
        },
        "version": "2.0",
    }


def write_default_prompts(
    path: str | Path = "prompts.default.yaml",
    domain: str = "ml",
) -> Path:
    """Write the export to *path* and return it."""
    target = Path(path)
    data = export_bank(domain)
    body = yaml.dump(
        data,
        Dumper=_PromptDumper,
        sort_keys=False,
        allow_unicode=True,
        width=100,
        default_flow_style=False,
    )
    target.write_text(
        _HEADER.format(domain=domain) + body, encoding="utf-8"
    )
    return target
