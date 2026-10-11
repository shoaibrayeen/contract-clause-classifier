"""Regenerate the release sections of changelog.html from CHANGELOG.md.

CHANGELOG.md is the source. Every `## [Unreleased]` / `## [X.Y.Z] - DATE` section is converted to an HTML
<section> (same entries, same order) and written between the `<!-- releases:start -->` and
`<!-- releases:end -->` markers in changelog.html; the version links in the page header are rebuilt too.

    uv run python scripts/sync_changelog_html.py          # write (make changelog-html)
    uv run python scripts/sync_changelog_html.py --check  # exit 1 if changelog.html is out of date (used by tests)
"""

from __future__ import annotations

import html
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MD, HTML = ROOT / "CHANGELOG.md", ROOT / "changelog.html"
START, END = "<!-- releases:start -->", "<!-- releases:end -->"
NAV_START, NAV_END = "<!-- versions:start -->", "<!-- versions:end -->"
TAGS = {"Added": "added", "Changed": "changed", "Fixed": "fixed", "Removed": "fixed", "Deprecated": "changed",
        "Security": "fixed", "Measured": "pinned", "Pinned": "pinned"}


def inline(text: str) -> str:
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    return re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<a href="\2">\1</a>', text)


def releases(md: str) -> list[tuple[str, str, str]]:
    """[(version, date, body)] for every release heading, in file order."""
    heads = list(re.finditer(r"^## \[([^\]]+)\](?: - (\S+))?\s*$", md, re.M))
    out = []
    for i, m in enumerate(heads):
        end = heads[i + 1].start() if i + 1 < len(heads) else md.find("\n[", m.end())
        body = md[m.end(): end if end != -1 else len(md)]
        body = re.split(r"^\[[^\]]+\]: http", body, flags=re.M)[0]  # drop the link-reference footer
        out.append((m.group(1), m.group(2) or "", body.strip()))
    return out


def anchor(version: str) -> str:
    return "unreleased" if version.lower() == "unreleased" else "v" + version.replace(".", "-")


def section(version: str, date: str, body: str) -> str:
    lines = [f'<section class="release" id="{anchor(version)}">',
             f'  <h2><span class="tag version">{html.escape(version)}</span>'
             + (f' <span class="meta">{date}</span>' if date else "") + "</h2>"]
    in_ul = False
    for ln in body.splitlines():
        if ln.startswith("### ") or ln.startswith("#### "):
            if in_ul:
                lines.append("  </ul>"); in_ul = False
            name = ln.lstrip("#").strip()
            if ln.startswith("### "):
                tag = TAGS.get(name.split()[0], "added")
                lines.append(f'  <h3><span class="tag {tag}">{inline(name)}</span></h3>')
            else:
                lines.append(f"  <h4>{inline(name)}</h4>")
        elif ln.startswith("- "):
            if not in_ul:
                lines.append("  <ul>"); in_ul = True
            lines.append(f"    <li>{inline(ln[2:])}</li>")
        elif ln.strip():
            if in_ul:
                lines.append("  </ul>"); in_ul = False
            lines.append(f"  <p>{inline(ln.strip())}</p>")
    if in_ul:
        lines.append("  </ul>")
    lines.append("</section>")
    return "\n".join(lines)


def render(page: str, md: str) -> str:
    rels = releases(md)
    body = "\n\n".join(section(v, d, b) for v, d, b in rels)
    nav = "\n".join(f'    <a href="#{anchor(v)}">{html.escape(v)}</a>' for v, _, _ in rels)
    if START not in page or NAV_START not in page:
        raise SystemExit(f"{HTML.name} is missing the {START} / {NAV_START} markers")
    page = re.sub(re.escape(START) + r".*?" + re.escape(END), lambda _: f"{START}\n{body}\n{END}", page, flags=re.S)
    page = re.sub(re.escape(NAV_START) + r".*?" + re.escape(NAV_END), lambda _: f"{NAV_START}\n{nav}\n    {NAV_END}",
                  page, flags=re.S)
    return page


def main(argv: list[str]) -> int:
    page, md = HTML.read_text(), MD.read_text()
    new = render(page, md)
    if "--check" in argv:
        if new != page:
            print("changelog.html is out of date with CHANGELOG.md (run `make changelog-html`)")
            return 1
        print("changelog.html is in sync with CHANGELOG.md")
        return 0
    HTML.write_text(new)
    print(f"wrote {HTML.name}: {len(releases(md))} releases")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
