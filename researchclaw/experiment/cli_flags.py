"""Per-agent CLI flags for the domain experiment agents.

The HEP / biology / statistics sandboxes shell out to a coding CLI with a single
instruction (``<binary> -p <instruction>``). Most of that command line is
shared; the flag set is not. ``--max-turns`` exists only in Claude Code, and
passing it to pi, codex or opencode makes the process exit immediately with an
unknown-option error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence


def is_claude_binary(binary: str) -> bool:
    """True when *binary* looks like the Claude Code CLI."""
    return Path(binary).name.lower().startswith("claude")


def agent_cli_flags(
    binary: str,
    *,
    max_turns: int = 0,
    extra_args: Sequence[str] = (),
) -> list[str]:
    """Flags to append to ``<binary> -p <instruction>``.

    ``--max-turns`` is Claude Code specific and is emitted only for it; every
    other CLI (pi, codex, opencode ...) would abort on the unknown flag.
    ``extra_args`` is passed through unchanged, minus empty entries.
    """
    flags: list[str] = []
    if max_turns > 0 and is_claude_binary(binary):
        flags += ["--max-turns", str(max_turns)]
    flags += [str(arg) for arg in extra_args if arg]
    return flags
