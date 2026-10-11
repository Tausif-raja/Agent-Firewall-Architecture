from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HiddenSpan:
    kind: str       # e.g. "css_hidden", "html_comment", "pdf_white_text", "pdf_metadata"
    text: str
    where: str = "" # page number, element path, JSON path...


@dataclass
class Extracted:
    """What a human would see (visible) vs. what only a parser sees (hidden).

    Hidden text is scanned, then removed before content reaches the model.
    Hidden text that contains instructions is close to a confirmed attack.
    """
    visible: str
    hidden: list[HiddenSpan] = field(default_factory=list)
    meta: dict = field(default_factory=dict)
