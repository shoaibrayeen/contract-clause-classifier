"""Every AI assistant's rules file must match the core rules in AGENTS.md (regenerate with `make rules`)."""

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sync_module():
    spec = importlib.util.spec_from_file_location("sync_agent_rules", ROOT / "scripts" / "sync_agent_rules.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_agent_rules"] = mod  # dataclasses resolve their module through sys.modules
    spec.loader.exec_module(mod)
    return mod


def test_generated_rule_files_are_in_sync():
    mod = _sync_module()
    stale = [rel for rel, content in mod.expected_files().items()
             if not (ROOT / rel).exists() or (ROOT / rel).read_text() != content]
    assert not stale, f"Run `make rules`; out of sync with AGENTS.md: {stale}"


def test_core_rules_cover_the_essentials():
    rules = _sync_module().core_rules((ROOT / "AGENTS.md").read_text())
    for must in ("uv run", "make sanity", "immutable", "OpenAI-compatible", "make test", "secrets",
                 "CHANGELOG.md", "make changelog-html", "MEMORY.md"):
        assert must in rules, must


def test_claude_md_imports_agents_md():
    assert "@AGENTS.md" in (ROOT / "CLAUDE.md").read_text()
