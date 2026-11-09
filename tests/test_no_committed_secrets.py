"""No real API key may reach the repository.

``.gitignore`` covers the known config paths and a local pre-commit hook blocks
a pasted key, but neither survives a machine change or a ``git add -f``. This
test is the version-controlled half of the guard: it scans every tracked file
for the shapes of keys this project can hold, and fails with the file and line.

Placeholders are allowed: every match whose secret part is a single repeated
character (``sk-xxxxxxxxxxxxxxxx``) is a documentation stub.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# Shapes of the keys this project can hold: OpenCode Go (``oc_sk_``), the
# paperclip/gxl gateway, OpenAI/Anthropic, GitHub, Slack, Google.
_PATTERNS = (
    ("opencode-go", re.compile(r"oc_sk_[A-Za-z0-9_-]{16,}")),
    ("gxl/paperclip", re.compile(r"gxl_[A-Za-z0-9_-]{16,}")),
    ("openai/anthropic", re.compile(r"sk-[A-Za-z0-9_-]{24,}")),
    ("github", re.compile(r"ghp_[A-Za-z0-9]{20,}")),
    ("google", re.compile(r"AIza[0-9A-Za-z_-]{30,}")),
    ("slack", re.compile(r"xox[baprs]-[A-Za-z0-9-]{10,}")),
)

_LOCAL_ONLY_FILES = ("config.arc.yaml", "config.yaml", "config.researchclaw.yaml")

_MAX_BYTES = 2_000_000


def _tracked_files() -> list[Path]:
    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    listed = subprocess.run(
        ["git", "ls-files", "-z"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return [ROOT / name for name in listed.split("\0") if name]


def _is_placeholder(secret: str) -> bool:
    """``sk-xxxxxxxxxxxxxxxx`` and friends: a run of one repeated character.

    Documentation stubs spell a key as a single repeated letter, so any
    hyphen/underscore-separated segment of eight or more characters with one
    distinct character marks the whole match as a placeholder.
    """
    segments = [part for part in re.split(r"[-_]", secret) if part]
    if any(len(part) >= 8 and len(set(part)) == 1 for part in segments):
        return True
    return set(secret) <= set("xX0*-.")


def test_no_api_keys_are_tracked() -> None:
    offenders: list[str] = []
    for path in _tracked_files():
        if not path.is_file() or path.stat().st_size > _MAX_BYTES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue  # binaries are not where keys are pasted
        for lineno, line in enumerate(text.splitlines(), start=1):
            for label, pattern in _PATTERNS:
                for match in pattern.finditer(line):
                    if _is_placeholder(match.group(0)):
                        continue
                    rel = path.relative_to(ROOT)
                    offenders.append(f"{rel}:{lineno} ({label})")
    assert offenders == [], (
        "possible API key in tracked files — move it to "
        "~/.config/researchclaw/ and reference it via api_key_env:\n"
        + "\n".join(offenders)
    )


def test_local_config_and_env_files_are_ignored() -> None:
    """The files a user puts a key into must not be trackable."""
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")
    for name in (*_LOCAL_ONLY_FILES, ".env"):
        assert name in gitignore, f"{name} missing from .gitignore"

    if shutil.which("git") is None or not (ROOT / ".git").exists():
        pytest.skip("not a git checkout")
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=ROOT, capture_output=True, text=True, check=True
    ).stdout.split("\n")
    leaked = [name for name in _LOCAL_ONLY_FILES if name in tracked]
    assert leaked == [], f"local config is tracked: {leaked}"
