"""HTML extraction that separates rendered text from text hidden from humans.

Static analysis: inline styles, hidden attributes, well-known hiding classes,
and simple .class / #id rules from <style> blocks earlier in the page. Pages
that hide text through external stylesheets or JavaScript need a headless
browser render diffed against the raw text.
"""

from __future__ import annotations

import re
from html.parser import HTMLParser

from prompt_injection.extractors.base import Extracted, HiddenSpan

_VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
         "param", "source", "track", "wbr"}
_SKIP = {"script", "style", "template"}                 # not text at all
_HIDDEN_TAGS = {"noscript", "title"} | {"head"}          # text exists but isn't page body
_BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section",
          "article", "header", "footer", "blockquote", "pre", "td", "th", "table", "ul", "ol"}

_HIDDEN_STYLE = re.compile(
    r"display\s*:\s*none|visibility\s*:\s*(hidden|collapse)|opacity\s*:\s*0(\.0+)?\s*(;|$)"
    r"|font-size\s*:\s*0(\.\d+)?(px|pt|em|rem)?\s*(;|$)|font-size\s*:\s*(0?\.\d+|1)px"
    r"|(left|top|right|text-indent)\s*:\s*-\d{3,}"
    r"|clip\s*:\s*rect\(\s*0|clip-path\s*:\s*inset\(\s*(50|100)%"
    r"|(width|height)\s*:\s*0(px)?\s*;.*overflow\s*:\s*hidden"
    r"|color\s*:\s*(#fff(fff)?|white|rgba?\(\s*255\s*,\s*255\s*,\s*255)\b[^;]*;?.*background(-color)?\s*:\s*(#fff(fff)?|white)"
    r"|color\s*:\s*transparent",
    re.I,
)
# Utility classes from Bootstrap/Tailwind/WordPress etc. that hide content visually.
_HIDING_CLASSES = {
    "hidden", "d-none", "hide", "invisible", "sr-only", "visually-hidden", "visuallyhidden",
    "screen-reader-text", "screen-reader-only", "offscreen", "a11y-hidden", "u-hidden", "is-hidden",
}
_CSS_RULE = re.compile(r"([^{}]+)\{([^{}]*)\}")
_SIMPLE_SELECTOR = re.compile(r"^([.#])([\w-]+)$")
_TEXT_ATTRS = ("alt", "title", "aria-label", "aria-description", "placeholder", "data-prompt", "data-instructions")


class _Parser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, bool, bool]] = []  # (tag, hidden, skip)
        self.visible: list[str] = []
        self.hidden: list[HiddenSpan] = []
        self.hidden_classes: set[str] = set(_HIDING_CLASSES)
        self.hidden_ids: set[str] = set()

    def _learn_css(self, css: str) -> None:
        for selectors, body in _CSS_RULE.findall(css):
            if not _HIDDEN_STYLE.search(body):
                continue
            for sel in selectors.split(","):
                m = _SIMPLE_SELECTOR.match(sel.strip())
                if m:
                    (self.hidden_classes if m.group(1) == "." else self.hidden_ids).add(m.group(2))

    @property
    def _hidden(self) -> bool:
        return bool(self.stack) and self.stack[-1][1]

    @property
    def _skip(self) -> bool:
        return bool(self.stack) and self.stack[-1][2]

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        for name in _TEXT_ATTRS:
            if a.get(name, "").strip():
                self.hidden.append(HiddenSpan("html_attr", a[name].strip(), f"{tag}[{name}]"))
        if tag == "meta" and a.get("content", "").strip():
            self.hidden.append(HiddenSpan("html_meta", a["content"].strip(), a.get("name") or a.get("property", "")))
        if tag in _BLOCK:
            self.visible.append("\n")
        if tag in _VOID:
            return
        hidden = (
            self._hidden
            or tag in _HIDDEN_TAGS
            or "hidden" in a
            or a.get("aria-hidden", "").lower() == "true"
            or a.get("type", "").lower() == "hidden"
            or bool(_HIDDEN_STYLE.search(a.get("style", "")))
            or bool(self.hidden_classes.intersection(a.get("class", "").split()))
            or a.get("id", "") in self.hidden_ids
        )
        self.stack.append((tag, hidden, self._skip or tag in _SKIP))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID and self.stack and self.stack[-1][0] == tag:
            self.stack.pop()

    def handle_endtag(self, tag: str) -> None:
        # Tolerate malformed HTML: pop up to the matching open tag, if any.
        for i in range(len(self.stack) - 1, -1, -1):
            if self.stack[i][0] == tag:
                del self.stack[i:]
                break
        if tag in _BLOCK:
            self.visible.append("\n")

    def handle_data(self, data: str) -> None:
        if self.stack and self.stack[-1][0] == "style":
            self._learn_css(data)
        if self._skip or not data.strip():
            if not self._skip:
                self.visible.append(data)
            return
        if self._hidden:
            tag = self.stack[-1][0] if self.stack else ""
            self.hidden.append(HiddenSpan("css_hidden", data.strip(), tag))
        else:
            self.visible.append(data)

    def handle_comment(self, data: str) -> None:
        if data.strip():
            self.hidden.append(HiddenSpan("html_comment", data.strip()))


def extract_html(raw: str) -> Extracted:
    p = _Parser()
    p.feed(raw)
    p.close()
    text = re.sub(r"[ \t\r\f\v]+", " ", "".join(p.visible))
    text = re.sub(r"\s*\n\s*", "\n", text).strip()
    # Merge consecutive hidden fragments from the same element so detectors see whole sentences.
    merged: list[HiddenSpan] = []
    for span in p.hidden:
        if merged and merged[-1].kind == span.kind == "css_hidden" and merged[-1].where == span.where:
            merged[-1].text += " " + span.text
        else:
            merged.append(span)
    return Extracted(visible=text, hidden=merged)
