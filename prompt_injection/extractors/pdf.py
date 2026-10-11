"""PDF extraction with hidden-text detection (requires PyMuPDF: pip install pymupdf).

Flags text a human reader would not see: white / near-background colour,
microscopic font size, text positioned outside the page, invisible render
mode, plus metadata, annotations and form fields.

Not covered: text hidden *under* images or vector shapes. Detecting that needs
rendering and comparing against OCR output.
"""

from __future__ import annotations

from prompt_injection.extractors.base import Extracted, HiddenSpan

MIN_VISIBLE_FONT_PT = 2.0
NEAR_WHITE = 0.92           # luminance above this on a (presumed) white page counts as invisible


def _luminance(srgb: int) -> float:
    r, g, b = (srgb >> 16) & 255, (srgb >> 8) & 255, srgb & 255
    return (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0


def extract_pdf(data: bytes) -> Extracted:
    try:
        import fitz  # PyMuPDF
    except ImportError as e:
        raise RuntimeError("PDF extraction requires PyMuPDF: pip install pymupdf") from e

    visible_pages: list[str] = []
    hidden: list[HiddenSpan] = []
    with fitz.open(stream=data, filetype="pdf") as doc:
        for key, val in (doc.metadata or {}).items():
            if isinstance(val, str) and len(val.strip()) > 3 and key in ("title", "subject", "keywords", "author", "creator"):
                hidden.append(HiddenSpan("pdf_metadata", val.strip(), key))
        if doc.embfile_count():
            hidden.append(HiddenSpan("pdf_embedded_files", f"{doc.embfile_count()} embedded file(s)"))

        for pno, page in enumerate(doc, start=1):
            page_rect = page.rect
            lines_out: list[str] = []
            # flags=0 keeps text in invisible render mode (3) visible to us; we classify it below.
            info = page.get_text("rawdict", flags=fitz.TEXT_PRESERVE_WHITESPACE)
            for block in info.get("blocks", []):
                for line in block.get("lines", []):
                    vis_parts: list[str] = []
                    for span in line.get("spans", []):
                        text = span.get("text")
                        if text is None:  # rawdict stores characters individually
                            text = "".join(ch.get("c", "") for ch in span.get("chars", []))
                        if not text.strip():
                            vis_parts.append(text)
                            continue
                        reason = None
                        bbox = fitz.Rect(span["bbox"])
                        if span.get("size", 12) < MIN_VISIBLE_FONT_PT:
                            reason = "pdf_tiny_text"
                        elif _luminance(span.get("color", 0)) > NEAR_WHITE:
                            reason = "pdf_white_text"
                        elif not bbox.intersects(page_rect):
                            reason = "pdf_offpage_text"
                        elif span.get("alpha", 255) == 0:
                            reason = "pdf_transparent_text"
                        if reason:
                            hidden.append(HiddenSpan(reason, text.strip(), f"page {pno}"))
                        else:
                            vis_parts.append(text)
                    line_text = "".join(vis_parts).strip()
                    if line_text:
                        lines_out.append(line_text)

            # Render mode 3 ("invisible") text: present in the text layer, never drawn.
            try:
                for tr in page.get_texttrace():
                    if tr.get("type") == 3 or tr.get("opacity", 1) == 0:
                        chars = "".join(chr(c[0]) for c in tr.get("chars", []))
                        if chars.strip():
                            hidden.append(HiddenSpan("pdf_invisible_render", chars.strip(), f"page {pno}"))
                            lines_out = [l for l in lines_out if chars.strip() not in l]
            except AttributeError:
                pass  # older PyMuPDF

            for annot in page.annots() or []:
                content = (annot.info or {}).get("content", "").strip()
                if content:
                    hidden.append(HiddenSpan("pdf_annotation", content, f"page {pno}"))
            for widget in page.widgets() or []:
                value = str(widget.field_value or "").strip()
                if value:
                    hidden.append(HiddenSpan("pdf_form_field", value, f"page {pno}:{widget.field_name}"))

            visible_pages.append("\n".join(lines_out))

        has_js = False
        try:
            has_js = any("JavaScript" in (doc.xref_object(x) or "") for x in range(1, doc.xref_length()))
        except Exception:
            pass

    return Extracted(visible="\n\n".join(visible_pages).strip(), hidden=hidden, meta={"pdf_has_javascript": has_js})
