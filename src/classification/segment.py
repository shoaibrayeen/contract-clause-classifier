"""Split a full contract into provision-sized chunks (what LEDGAR examples look like).

Heuristics, in order:
1. Split into paragraphs on blank lines (or on single newlines when the text has no blank lines).
2. Short heading-only paragraphs ("ARTICLE V", "GOVERNING LAW") are glued to the next paragraph.
3. A paragraph that starts with a section number / "Section x" / "Article x" starts a new provision;
   sub-items like "(a) ..." and unnumbered continuations join the current provision.
4. Very long chunks are split on inline section numbers, then on sentence boundaries.
5. Fragments shorter than `min_words` are merged into a neighbour.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

NUMBERED = re.compile(
    r"""^\s*(
        (ARTICLE|Article|SECTION|Section|§)\s+[\dIVXLCivxlc]+[\.:]?   # Article IV / Section 5.2
      | \d{1,3}(\.\d{1,3})*\.?\)?\s+[A-Z(]                              # 1. / 1.1 / 12.3.4 Title
      | [IVXL]{1,6}\.\s+[A-Z]                                           # IV. Title
    )""",
    re.VERBOSE,
)
SUBITEM = re.compile(r"^\s*\(?([a-z]{1,2}|[ivx]{1,5}|\d{1,2})\)\s+")
INLINE_SECTION = re.compile(r"(?<=[.;:])\s+(?=(?:Section\s+)?\d{1,2}(?:\.\d{1,2})*\.?\s+[A-Z][a-z])")
SENTENCE = re.compile(r"(?<=[.!?;])\s+(?=[A-Z(\"])")


@dataclass
class Provision:
    index: int
    text: str
    start: int  # char offset in the normalised text (approximate for merged chunks)


def _normalise(text: str) -> str:
    text = text.replace("\r\n", "\n").replace("\r", "\n").replace(" ", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"-\n(?=[a-z])", "", text)  # de-hyphenate PDF line breaks
    return text.strip()


def _paragraphs(text: str) -> list[str]:
    blocks = [b.strip() for b in re.split(r"\n\s*\n", text) if b.strip()]
    if len(blocks) <= 2 and text.count("\n") > 5:
        blocks = [b.strip() for b in text.split("\n") if b.strip()]
    # Inside a block, join hard-wrapped lines unless a line starts a numbered item.
    out = []
    for b in blocks:
        lines = b.split("\n")
        cur = lines[0]
        for ln in lines[1:]:
            if NUMBERED.match(ln) or SUBITEM.match(ln):
                out.append(cur.strip())
                cur = ln
            else:
                cur += " " + ln.strip()
        out.append(cur.strip())
    return [p for p in out if p]


def _is_heading(p: str) -> bool:
    words = p.split()
    if not words or len(words) > 12 or p.endswith((".", ";", ",")) and len(words) > 6:
        return False
    letters = [c for c in p if c.isalpha()]
    upper = letters and sum(c.isupper() for c in letters) / len(letters) > 0.8
    article = re.match(r"^\s*(ARTICLE|Article)\s+[\dIVXLC]+", p) is not None
    return bool(upper or article)


def _words(s: str) -> int:
    return len(s.split())


def _split_long(text: str, max_words: int) -> list[str]:
    if _words(text) <= max_words:
        return [text]
    parts = [p for p in INLINE_SECTION.split(text) if p.strip()]
    if len(parts) == 1:
        parts = [p for p in SENTENCE.split(text) if p.strip()]
    chunks, cur = [], ""
    for p in parts:
        if cur and _words(cur) + _words(p) > max_words:
            chunks.append(cur.strip())
            cur = p
        else:
            cur = f"{cur} {p}" if cur else p
    if cur.strip():
        chunks.append(cur.strip())
    # A single sentence longer than max_words: hard split by words.
    final = []
    for c in chunks:
        w = c.split()
        final.extend(" ".join(w[i : i + max_words]) for i in range(0, len(w), max_words)) if len(w) > max_words else final.append(c)
    return final


def segment_contract(text: str, min_words: int = 25, max_words: int = 350) -> list[Provision]:
    text = _normalise(text)
    if not text:
        return []
    paras = _paragraphs(text)

    # Glue headings to the following paragraph.
    glued: list[str] = []
    pending = ""
    for p in paras:
        if _is_heading(p):
            pending = f"{pending} {p}".strip()
            continue
        glued.append(f"{pending} {p}".strip() if pending else p)
        pending = ""
    if pending:
        glued.append(pending)

    # Group sub-items / continuations under the current numbered provision.
    groups: list[str] = []
    for p in glued:
        starts_new = bool(NUMBERED.match(p)) or not groups or not SUBITEM.match(p) and _words(groups[-1]) >= min_words
        if starts_new:
            groups.append(p)
        else:
            groups[-1] = f"{groups[-1]} {p}"

    chunks = [c for g in groups for c in _split_long(g, max_words)]

    # Merge tiny *unnumbered* fragments into the previous chunk (or the next one if first).
    # Short numbered sections ("12. Counterparts. ...") are real provisions and stay separate.
    merged: list[str] = []
    for c in chunks:
        tiny = _words(c) < min_words and not NUMBERED.match(c)
        if merged and tiny and _words(merged[-1]) + _words(c) <= max_words:
            merged[-1] = f"{merged[-1]} {c}"
        else:
            merged.append(c)
    if len(merged) > 1 and _words(merged[0]) < min_words and not NUMBERED.match(merged[0]):
        merged[1] = f"{merged[0]} {merged[1]}"
        merged.pop(0)

    out, cursor = [], 0
    for i, c in enumerate(merged):
        probe = c[:40]
        pos = text.find(probe, cursor)
        if pos >= 0:
            cursor = pos
        out.append(Provision(index=i, text=c, start=max(pos, 0)))
    return out
