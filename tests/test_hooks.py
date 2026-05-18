"""Tests for the Claude Code PreToolUse safety hook (§6.5)."""

import io
import json
import sys

from cli.hook_pretool import main as pretool_main


def _run_hook(event: dict, monkeypatch) -> int:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(event)))
    return pretool_main()


def test_hook_blocks_rm_rf_root(monkeypatch):
    event = {"tool_name": "Bash", "tool_input": {"command": "rm -rf /"}}
    assert _run_hook(event, monkeypatch) == 2


def test_hook_blocks_force_push(monkeypatch):
    event = {"tool_name": "Bash", "tool_input": {"command": "git push origin main --force"}}
    assert _run_hook(event, monkeypatch) == 2


def test_hook_blocks_git_reset_hard(monkeypatch):
    event = {"tool_name": "Bash", "tool_input": {"command": "git reset --hard HEAD~5"}}
    assert _run_hook(event, monkeypatch) == 2


def test_hook_allows_safe_bash(monkeypatch):
    event = {"tool_name": "Bash", "tool_input": {"command": "pytest tests/"}}
    assert _run_hook(event, monkeypatch) == 0


def test_hook_allows_non_bash_tools(monkeypatch):
    event = {"tool_name": "Edit", "tool_input": {"file_path": "foo.py", "command": "rm -rf /"}}
    assert _run_hook(event, monkeypatch) == 0


def test_hook_handles_malformed_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("not json{"))
    assert pretool_main() == 0
