"""Tests for the coding-CLI codegen backends and their per-agent flags."""

from __future__ import annotations

from pathlib import Path

import pytest

from researchclaw.config import RCConfig
from researchclaw.experiment.cli_flags import agent_cli_flags, is_claude_binary
from researchclaw.experiment.code_agent import PiAgent, create_code_agent


def _config(tmp_path: Path, *, provider: str = "pi", binary_path: str = "pi") -> RCConfig:
    return RCConfig.from_dict(
        {
            "project": {"name": "codegen-test", "mode": "docs-first"},
            "research": {"topic": "codegen testing"},
            "runtime": {"timezone": "UTC"},
            "notifications": {"channel": "local"},
            "knowledge_base": {"root": str(tmp_path / "kb")},
            "llm": {
                "provider": "openai-compatible",
                "base_url": "http://localhost:1234/v1",
                "api_key_env": "RC_TEST_KEY",
                "api_key": "inline",
            },
            "experiment": {
                "cli_agent": {
                    "provider": provider,
                    "binary_path": binary_path,
                    "model": "deepseek-v4.1-flash",
                    "timeout_sec": 120,
                    "extra_args": ["--approve"],
                }
            },
        },
        project_root=tmp_path,
        check_paths=False,
    )


class TestAgentCliFlags:
    def test_claude_gets_max_turns(self) -> None:
        flags = agent_cli_flags("claude", max_turns=5, extra_args=["--foo"])
        assert flags == ["--max-turns", "5", "--foo"]

    def test_windows_claude_wrapper_recognised(self) -> None:
        assert is_claude_binary(r"C:\bin\claude.CMD") is True
        assert agent_cli_flags(
            r"C:\bin\claude.CMD", max_turns=2
        ) == ["--max-turns", "2"]

    @pytest.mark.parametrize("binary", ["pi", "pi.CMD", "codex", "opencode", "/usr/bin/pi"])
    def test_non_claude_agents_never_get_max_turns(self, binary: str) -> None:
        assert is_claude_binary(binary) is False
        assert agent_cli_flags(binary, max_turns=5) == []

    def test_zero_max_turns_emits_nothing(self) -> None:
        assert agent_cli_flags("claude", max_turns=0) == []

    def test_extra_args_skip_empty_entries(self) -> None:
        assert agent_cli_flags("pi", extra_args=["", "--x", ""]) == ["--x"]


class TestPiAgent:
    def test_command_uses_print_mode_and_tools(self) -> None:
        agent = PiAgent(
            binary_path="pi",
            model="deepseek-v4.1-flash",
            extra_args=["--approve"],
        )
        cmd = agent._build_cmd("PROMPT", Path("."))

        assert cmd[:3] == ["pi", "-p", "PROMPT"]
        assert "--no-session" in cmd
        assert cmd[cmd.index("--mode") + 1] == "text"
        assert cmd[cmd.index("--tools") + 1] == "read,bash,edit,write"
        assert cmd[cmd.index("--model") + 1] == "deepseek-v4.1-flash"
        assert cmd[-1] == "--approve"
        assert "--max-budget-usd" not in cmd

    def test_name(self) -> None:
        assert PiAgent(binary_path="pi").name == "pi"

    def test_isolated_from_user_config(self) -> None:
        cmd = PiAgent(binary_path="pi")._build_cmd("PROMPT", Path("."))
        assert "--no-mcp" in cmd
        assert "--no-skills" in cmd
        assert "--no-context-files" in cmd

    def test_collects_nested_files(self, tmp_path: Path) -> None:
        from researchclaw.experiment.code_agent import _collect_py_files

        (tmp_path / "pkg").mkdir()
        (tmp_path / "pkg" / "main.py").write_text("x = 1\n", encoding="utf-8")
        (tmp_path / "main.py").write_text("y = 2\n", encoding="utf-8")

        files = _collect_py_files(tmp_path)

        assert files["main.py"] == "y = 2\n"
        assert files["pkg/main.py"] == "x = 1\n"

    def test_nonzero_exit_with_files_is_an_error(self, tmp_path: Path) -> None:
        agent = PiAgent(binary_path="pi")
        (tmp_path / "main.py").write_text("x = 1\n", encoding="utf-8")

        result = agent._build_result(tmp_path, 2, "", "boom", 1.0, False)

        assert result.files
        assert result.error is not None
        assert result.ok is False


class TestCreateCodeAgent:
    def test_pi_provider(self, tmp_path: Path) -> None:
        agent = create_code_agent(_config(tmp_path))
        assert isinstance(agent, PiAgent)
        assert agent.name == "pi"

    def test_pi_provider_missing_binary(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            "researchclaw.experiment.code_agent.shutil.which", lambda _name: None
        )
        with pytest.raises(RuntimeError, match="pi binary not found"):
            create_code_agent(_config(tmp_path, binary_path=""))

    def test_config_accepts_pi_provider(self, tmp_path: Path) -> None:
        config = _config(tmp_path)
        assert config.experiment.cli_agent.provider == "pi"

    def test_config_rejects_unknown_provider(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Invalid experiment.cli_agent.provider"):
            _config(tmp_path, provider="not-a-cli")
