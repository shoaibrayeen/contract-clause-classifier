"""changelog.html must be generated from CHANGELOG.md, and versions must agree everywhere."""

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _module():
    spec = importlib.util.spec_from_file_location("sync_changelog_html", ROOT / "scripts" / "sync_changelog_html.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sync_changelog_html"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_changelog_html_in_sync():
    mod = _module()
    page = (ROOT / "changelog.html").read_text()
    assert mod.render(page, (ROOT / "CHANGELOG.md").read_text()) == page, "run `make changelog-html`"


def test_versions_agree():
    import classification

    md = (ROOT / "CHANGELOG.md").read_text()
    released = re.findall(r"^## \[(\d+\.\d+\.\d+)\] - \d{4}-\d{2}-\d{2}$", md, re.M)
    pyproject = re.search(r'^version = "([^"]+)"', (ROOT / "pyproject.toml").read_text(), re.M).group(1)
    assert released and released[0] == pyproject == classification.__version__
    assert f"contract-clause-classifier:{pyproject}" in (ROOT / "Makefile").read_text()
    assert f"contract-clause-classifier:{pyproject}" in (ROOT / "docker-compose.yml").read_text()
    assert md.index("## [Unreleased]") < md.index(f"## [{pyproject}]")
