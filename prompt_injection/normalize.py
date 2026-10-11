"""L0: canonicalize text and surface payloads hidden by encoding tricks.

Detectors run on the normalized text *and* on every decoded variant, so an
attacker can't slip a payload past them with base64, zero-width characters,
homoglyphs or Unicode tag characters.
"""

from __future__ import annotations

import base64
import binascii
import codecs
import re
import unicodedata
from dataclasses import dataclass, field
from urllib.parse import unquote

# Zero-width, bidi-control and other invisible formatting characters.
_INVISIBLE = re.compile(
    "[­͏؜ᅟᅠ឴឵᠋-᠏​-‏"
    "‪-‮⁠-⁤⁦-⁯ㅤ︀-️﻿ﾠ]"
)
# Unicode tag block (U+E0000-U+E007F): invisible, but maps 1:1 onto ASCII
# ("ASCII smuggling"). We decode it rather than just strip it.
_TAG_CHARS = re.compile("[\U000e0000-\U000e007f]+")

# Common confusables that NFKC does not fold (Cyrillic / Greek look-alikes).
_CONFUSABLES = str.maketrans({
    "а": "a", "е": "e", "о": "o", "р": "p", "с": "c", "у": "y", "х": "x",
    "і": "i", "ј": "j", "ѕ": "s", "ԁ": "d", "ɡ": "g", "һ": "h", "ӏ": "l",
    "А": "A", "В": "B", "Е": "E", "К": "K", "М": "M", "Н": "H", "О": "O",
    "Р": "P", "С": "C", "Т": "T", "Х": "X", "І": "I",
    "α": "a", "ο": "o", "ε": "e", "ι": "i", "κ": "k", "ν": "v", "ρ": "p",
    "τ": "t", "υ": "u", "Α": "A", "Β": "B", "Ε": "E", "Ζ": "Z", "Η": "H",
    "Ι": "I", "Κ": "K", "Μ": "M", "Ν": "N", "Ο": "O", "Ρ": "P", "Τ": "T",
    "Υ": "Y", "Χ": "X",
})

_B64_TOKEN = re.compile(r"(?<![A-Za-z0-9+/=_-])[A-Za-z0-9+/_-]{16,}={0,2}(?![A-Za-z0-9+/=_-])")
_HEX_TOKEN = re.compile(r"(?<![0-9A-Fa-f])(?:[0-9A-Fa-f]{2}){12,}(?![0-9A-Fa-f])")
_URLENC = re.compile(r"(?:%[0-9A-Fa-f]{2}){4,}")
_ROT13_HINT = re.compile(r"\brot[\s-]?13\b", re.IGNORECASE)

MAX_DECODED_VARIANTS = 8


@dataclass
class Normalized:
    text: str                                          # canonical form fed to detectors
    variants: list[str] = field(default_factory=list)  # decoded hidden payloads, scanned separately
    signals: dict[str, bool] = field(default_factory=dict)


def _printable_ratio(s: str) -> float:
    if not s:
        return 0.0
    ok = sum(1 for ch in s if ch.isprintable() or ch in "\n\t\r")
    return ok / len(s)


def _looks_like_text(s: str) -> bool:
    # Decoded binary (images, hashes, keys) is not interesting; decoded prose is.
    return len(s) >= 8 and _printable_ratio(s) > 0.95 and sum(ch.isalpha() for ch in s) / len(s) > 0.5 and " " in s


def _decode_tags(match: re.Match) -> str:
    out = []
    for ch in match.group(0):
        cp = ord(ch) - 0xE0000
        if 0x20 <= cp <= 0x7E:
            out.append(chr(cp))
    return "".join(out)


def _try_b64(token: str) -> str | None:
    t = token.replace("-", "+").replace("_", "/")
    t += "=" * (-len(t) % 4)
    try:
        raw = base64.b64decode(t, validate=True)
        s = raw.decode("utf-8")
    except (binascii.Error, ValueError, UnicodeDecodeError):
        return None
    return s if _looks_like_text(s) else None


def _try_hex(token: str) -> str | None:
    try:
        s = bytes.fromhex(token).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return None
    return s if _looks_like_text(s) else None


def canonicalize(text: str) -> str:
    """Unicode-normalize and fold confusables. Does not decode encodings."""
    text = unicodedata.normalize("NFKC", text)
    text = _INVISIBLE.sub("", text)
    text = text.translate(_CONFUSABLES)
    return text


def normalize(text: str) -> Normalized:
    signals: dict[str, bool] = {}
    variants: list[str] = []

    tag_payloads = [_decode_tags(m) for m in _TAG_CHARS.finditer(text)]
    tag_payloads = [p for p in tag_payloads if p.strip()]
    if tag_payloads:
        signals["unicode_tag_smuggling"] = True
        variants.extend(tag_payloads)
    text = _TAG_CHARS.sub("", text)

    if _INVISIBLE.search(text):
        signals["invisible_chars"] = True
    folded = unicodedata.normalize("NFKC", text).translate(_CONFUSABLES)
    if folded != unicodedata.normalize("NFKC", text):
        # Mixed-script words are a homoglyph signal; whole-script Cyrillic/Greek text is not.
        if re.search(r"[A-Za-z][Ͱ-ϿЀ-ӿ]|[Ͱ-ϿЀ-ӿ][A-Za-z]", text):
            signals["homoglyphs"] = True

    canon = canonicalize(text)

    for m in _B64_TOKEN.finditer(canon):
        decoded = _try_b64(m.group(0))
        if decoded:
            signals["base64_payload"] = True
            variants.append(decoded)
    for m in _HEX_TOKEN.finditer(canon):
        decoded = _try_hex(m.group(0))
        if decoded:
            signals["hex_payload"] = True
            variants.append(decoded)
    if _URLENC.search(canon):
        decoded = unquote(canon)
        if decoded != canon:
            signals["url_encoded"] = True
            variants.append(decoded)
    if _ROT13_HINT.search(canon):
        # ROT13 of ordinary text is gibberish, so only decode when the text asks for it.
        variants.append(codecs.decode(canon, "rot13"))

    # Decoded payloads may themselves be obfuscated.
    variants = [canonicalize(v) for v in variants][:MAX_DECODED_VARIANTS]
    return Normalized(text=canon, variants=variants, signals=signals)
