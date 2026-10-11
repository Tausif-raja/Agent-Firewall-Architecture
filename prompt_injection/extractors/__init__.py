"""Format-aware extraction: visible text plus text hidden from human readers."""

from __future__ import annotations

from prompt_injection.extractors.base import Extracted, HiddenSpan
from prompt_injection.extractors.html import extract_html
from prompt_injection.extractors.json_tool import extract_json
from prompt_injection.extractors.pdf import extract_pdf


def extract(raw: str | bytes, mime: str) -> Extracted:
    mime = (mime or "").split(";")[0].strip().lower()
    if mime == "application/pdf":
        if isinstance(raw, str):
            raise TypeError("PDF content must be bytes")
        return extract_pdf(raw)
    text = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else raw
    if mime in ("text/html", "application/xhtml+xml"):
        return extract_html(text)
    if mime == "application/json" or mime.endswith("+json"):
        return extract_json(text)
    return Extracted(visible=text)


__all__ = ["Extracted", "HiddenSpan", "extract", "extract_html", "extract_json", "extract_pdf"]
