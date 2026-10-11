"""
Sensitive data & GDPR engine: detects secrets, PII and GDPR Art. 9 special
categories, then masks, pseudonymizes (reversible per-session token vault)
or blocks them.

Detection is native regex + checksum validation. If Microsoft Presidio is
installed (pip install presidio-analyzer && python -m spacy download en_core_web_lg)
its PERSON / LOCATION / etc. findings are merged in.

Modes for sanitize_or_pseudonymize():
  "tokenize"  replace with <TYPE_n> tokens; unmask_egress() restores them in the reply
  "mask"      replace with <TYPE> (irreversible)
  "block"     tell the caller to reject the request if anything is found
Secrets are always masked irreversibly: they are never stored in the vault.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("agent_firewall.sensitive_data")

SECRET, PII, ART9 = "SECRET", "PII", "GDPR_ART9"
_PRIORITY = {SECRET: 3, PII: 2, ART9: 1}


@dataclass
class SensitiveFinding:
    entity_type: str
    category: str          # SECRET | PII | GDPR_ART9
    start: int
    end: int
    confidence: float
    description: str
    raw: str               # never serialized; kept only in memory for replacement

    @property
    def snippet(self) -> str:
        """Partially masked preview that is safe to show in a UI or log."""
        r = self.raw
        if self.category == SECRET:
            return r[:4] + "…" if len(r) > 8 else "***"
        if len(r) <= 6:
            return "***"
        return f"{r[:2]}***{r[-2:]}"

    def to_dict(self) -> dict:
        return {
            "entity_type": self.entity_type,
            "category": self.category,
            "snippet": self.snippet,
            "start": self.start,
            "end": self.end,
            "confidence": self.confidence,
            "description": self.description,
        }


# ---------------------------------------------------------------- validators

def _luhn_ok(s: str) -> bool:
    digits = [int(c) for c in s if c.isdigit()]
    if not 13 <= len(digits) <= 19:
        return False
    total = 0
    for i, d in enumerate(reversed(digits)):
        if i % 2:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    return total % 10 == 0


def _iban_ok(s: str) -> bool:
    s = s.replace(" ", "").upper()
    if not 15 <= len(s) <= 34:
        return False
    rearranged = s[4:] + s[:4]
    try:
        return int("".join(str(int(c, 36)) for c in rearranged)) % 97 == 1
    except ValueError:
        return False


def _ipv4_ok(s: str) -> bool:
    return all(0 <= int(p) <= 255 for p in s.split("."))


def _phone_ok(s: str) -> bool:
    return 9 <= sum(c.isdigit() for c in s) <= 15


@dataclass(frozen=True)
class _Pattern:
    entity_type: str
    category: str
    regex: re.Pattern
    confidence: float
    description: str
    validator: Optional[Callable[[str], bool]] = None
    group: int = 0  # capture group holding the sensitive value


def _p(entity, category, pattern, conf, desc, validator=None, group=0, flags=0):
    return _Pattern(entity, category, re.compile(pattern, flags), conf, desc, validator, group)


PATTERNS: List[_Pattern] = [
    # --- Secrets & credentials
    _p("PRIVATE_KEY", SECRET, r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?(?:-----END [A-Z ]*PRIVATE KEY-----|$)", 0.99, "PEM private key"),
    _p("OPENROUTER_API_KEY", SECRET, r"\bsk-or-v1-[A-Za-z0-9]{32,}", 0.99, "OpenRouter API key"),
    _p("ANTHROPIC_API_KEY", SECRET, r"\bsk-ant-[A-Za-z0-9_-]{20,}", 0.99, "Anthropic API key"),
    _p("OPENAI_API_KEY", SECRET, r"\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}", 0.95, "OpenAI-style API key"),
    _p("GROQ_API_KEY", SECRET, r"\bgsk_[A-Za-z0-9]{20,}", 0.99, "Groq API key"),
    _p("FIREWALL_API_KEY", SECRET, r"\bagf-live-[a-f0-9]{16,}", 0.99, "Agent Firewall virtual key"),
    _p("AWS_ACCESS_KEY", SECRET, r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b", 0.99, "AWS access key id"),
    _p("GITHUB_TOKEN", SECRET, r"\bgh[pousr]_[A-Za-z0-9]{36,}\b", 0.99, "GitHub token"),
    _p("GOOGLE_API_KEY", SECRET, r"\bAIza[0-9A-Za-z_-]{35}\b", 0.99, "Google API key"),
    _p("SLACK_TOKEN", SECRET, r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b", 0.99, "Slack token"),
    _p("JWT", SECRET, r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b", 0.95, "JSON Web Token"),
    _p("BEARER_TOKEN", SECRET, r"\bBearer\s+([A-Za-z0-9._~+/-]{20,}=*)", 0.9, "HTTP bearer token", group=1),
    _p("PASSWORD", SECRET, r"\b(?:password|passwd|pwd|passcode)\s*(?:is|=|:)\s*[\"']?([^\s\"']{4,})", 0.85, "Password in plain text", group=1, flags=re.I),
    _p("CONNECTION_STRING", SECRET, r"\b[a-z][a-z0-9+.-]*://[^\s:/@]+:([^\s@/]+)@[^\s]+", 0.95, "Credential inside connection URL", group=1, flags=re.I),
    _p("GENERIC_SECRET", SECRET, r"\b(?:api[_-]?key|secret|access[_-]?token|auth[_-]?token|client[_-]?secret)\s*(?:is|=|:)\s*[\"']?([A-Za-z0-9_\-./+=]{12,})", 0.8, "Secret value assignment", group=1, flags=re.I),

    # --- PII
    _p("EMAIL_ADDRESS", PII, r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", 0.95, "Email address"),
    _p("CREDIT_CARD", PII, r"\b(?:\d[ -]?){12,18}\d\b", 0.95, "Payment card number (Luhn-valid)", _luhn_ok),
    _p("IBAN", PII, r"\b[A-Z]{2}\d{2}(?: ?[A-Z0-9]{4}){2,7}(?: ?[A-Z0-9]{1,3})?\b", 0.9, "International bank account number", _iban_ok),
    _p("US_SSN", PII, r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", 0.85, "US social security number"),
    _p("PHONE_NUMBER", PII, r"(?<![\w.])(?<!\d[\s.-])(?:\+\d{1,3}[\s.-]?)?(?:\(\d{2,4}\)[\s.-]?)?\d{2,4}[\s.-]\d{3,4}[\s.-]?\d{3,4}(?![\w.])(?![\s.-]\d)", 0.7, "Phone number", _phone_ok),
    _p("IP_ADDRESS", PII, r"\b(?:\d{1,3}\.){3}\d{1,3}\b", 0.6, "IPv4 address", _ipv4_ok),
    _p("DATE_OF_BIRTH", PII, r"\b(?:born\s+on|date\s+of\s+birth|dob)\s*[:\-]?\s*([0-9]{1,4}[-/.][0-9]{1,2}[-/.][0-9]{1,4})", 0.8, "Date of birth", group=1, flags=re.I),

    # --- GDPR Art. 9 special categories (keyword heuristics, lower confidence)
    _p("HEALTH_DATA", ART9, r"\b(?:diagnosed\s+with\s+[\w\s-]{3,30}?(?=[.,;]|$)|(?:i|he|she|they)\s+(?:have|has)\s+(?:hiv|aids|cancer|diabetes|epilepsy|depression|schizophrenia|bipolar\s+disorder)|hiv[- ]positive|pregnan(?:t|cy))", 0.7, "Health data (Art. 9)", flags=re.I | re.M),
    _p("SEXUAL_ORIENTATION", ART9, r"\b(?:i\s*'?\s*a?m|he\s+is|she\s+is|they\s+are)\s+(?:gay|lesbian|bisexual|transgender|homosexual|queer)\b", 0.7, "Sexual orientation (Art. 9)", flags=re.I),
    _p("RELIGIOUS_BELIEF", ART9, r"\b(?:i\s*'?\s*a?m|he\s+is|she\s+is|they\s+are)\s+(?:a\s+)?(?:muslim|christian|jewish|hindu|buddhist|sikh|atheist|catholic)\b", 0.65, "Religious belief (Art. 9)", flags=re.I),
    _p("POLITICAL_OPINION", ART9, r"\b(?:i|he|she|they)\s+(?:vote[sd]?\s+for|(?:am|is|are)\s+a\s+member\s+of)\s+(?:the\s+)?[\w\s]{2,30}?\s+party\b", 0.6, "Political opinion (Art. 9)", flags=re.I),
    _p("TRADE_UNION", ART9, r"\b(?:member\s+of|joined)\s+(?:the\s+)?[\w\s]{0,30}?(?:trade\s+)?union\b", 0.6, "Trade-union membership (Art. 9)", flags=re.I),
    _p("BIOMETRIC_GENETIC", ART9, r"\b(?:fingerprint\s+(?:scan|data|template)|face\s*id\s+template|retina\s+scan|dna\s+(?:test|sequence|profile)|genetic\s+(?:test|marker|profile))", 0.7, "Biometric / genetic data (Art. 9)", flags=re.I),
]

_TOKEN_RE = re.compile(r"<([A-Z_]+)_(\d+)>")


class _Vault:
    """Per-session reversible token store, bounded so it can't grow forever."""

    def __init__(self, max_sessions: int = 2000):
        self._data: "OrderedDict[str, Dict[str, Dict[str, str]]]" = OrderedDict()
        self._max = max_sessions
        self._lock = threading.Lock()

    def tokenize(self, session_id: str, entity_type: str, value: str) -> str:
        with self._lock:
            s = self._data.setdefault(session_id, {"by_value": {}, "by_token": {}})
            self._data.move_to_end(session_id)
            if len(self._data) > self._max:
                self._data.popitem(last=False)
            key = f"{entity_type}\x00{value}"
            if key in s["by_value"]:
                return s["by_value"][key]
            n = sum(1 for t in s["by_token"] if t.startswith(f"<{entity_type}_")) + 1
            token = f"<{entity_type}_{n}>"
            s["by_value"][key] = token
            s["by_token"][token] = value
            return token

    def restore(self, session_id: str, text: str) -> str:
        with self._lock:
            s = self._data.get(session_id)
            if not s:
                return text
            table = dict(s["by_token"])
        return _TOKEN_RE.sub(lambda m: table.get(m.group(0), m.group(0)), text)

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._data.pop(session_id, None)


class SensitiveDataEngine:
    def __init__(self, use_presidio: bool = True):
        self.vault = _Vault()
        self._presidio = None
        if use_presidio:
            try:
                from presidio_analyzer import AnalyzerEngine
                self._presidio = AnalyzerEngine()
                logger.info("Presidio analyzer enabled")
            except Exception as e:  # not installed, or spaCy model missing
                logger.info("Presidio unavailable, using native detectors only (%s)", e.__class__.__name__)

    # ------------------------------------------------------------ detection

    def detect(self, text: str) -> List[SensitiveFinding]:
        if not text:
            return []
        found: List[SensitiveFinding] = []
        for p in PATTERNS:
            for m in p.regex.finditer(text):
                start, end = m.span(p.group)
                value = m.group(p.group)
                if not value or (p.validator and not p.validator(value)):
                    continue
                found.append(SensitiveFinding(p.entity_type, p.category, start, end, p.confidence, p.description, value))
        if self._presidio is not None:
            try:
                for r in self._presidio.analyze(text=text, language="en", score_threshold=0.6):
                    found.append(SensitiveFinding(r.entity_type, PII, r.start, r.end, round(r.score, 2),
                                                  f"Presidio: {r.entity_type}", text[r.start:r.end]))
            except Exception as e:
                logger.warning("Presidio analysis failed: %s", e)
        return self._resolve_overlaps(found)

    @staticmethod
    def _resolve_overlaps(found: List[SensitiveFinding]) -> List[SensitiveFinding]:
        # Prefer higher category priority, then longer span, then confidence.
        found.sort(key=lambda f: (-_PRIORITY[f.category], -(f.end - f.start), -f.confidence))
        kept: List[SensitiveFinding] = []
        for f in found:
            if all(f.end <= k.start or f.start >= k.end for k in kept):
                kept.append(f)
        return sorted(kept, key=lambda f: f.start)

    # ------------------------------------------------------- transformation

    def sanitize_or_pseudonymize(
        self, text: str, session_id: str, mode: str = "tokenize"
    ) -> Tuple[str, List[SensitiveFinding], bool]:
        """Return (sanitized_text, findings, should_block)."""
        mode = (mode or "tokenize").lower()
        if mode not in ("tokenize", "mask", "block"):
            mode = "tokenize"
        findings = self.detect(text)
        if not findings:
            return text, [], False
        if mode == "block":
            return self._mask(text, findings), findings, True

        out, last = [], 0
        for f in findings:
            out.append(text[last:f.start])
            if mode == "tokenize" and f.category != SECRET:
                out.append(self.vault.tokenize(session_id, f.entity_type, f.raw))
            else:
                out.append(f"<{f.entity_type}>")
            last = f.end
        out.append(text[last:])
        return "".join(out), findings, False

    @staticmethod
    def _mask(text: str, findings: List[SensitiveFinding]) -> str:
        out, last = [], 0
        for f in findings:
            out += [text[last:f.start], f"<{f.entity_type}>"]
            last = f.end
        out.append(text[last:])
        return "".join(out)

    def redact_secrets(self, text: str) -> Tuple[str, int]:
        """Mask only credentials (used on tool output). Returns (text, count)."""
        secrets = [f for f in self.detect(text) if f.category == SECRET]
        return (self._mask(text, secrets), len(secrets)) if secrets else (text, 0)

    def unmask_egress(self, text: str, session_id: str) -> str:
        """Restore tokenized values in a response, for the authorized sender only."""
        return self.vault.restore(session_id, text or "")

    def clear_session(self, session_id: str) -> None:
        self.vault.clear(session_id)

    # ------------------------------------------------------------ reporting

    def generate_ropa_record(self, findings: List[SensitiveFinding], session_id: str = "") -> dict:
        """Draft GDPR Art. 30 record-of-processing entry for this request.

        Generated automatically from detections; it is documentation support,
        not a legal determination. Review with your DPO before relying on it.
        """
        categories = sorted({f.entity_type for f in findings})
        special = sorted({f.entity_type for f in findings if f.category == ART9})
        return {
            "record_id": str(uuid.uuid4()),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "processing_activity": "AI agent request inspection by Agent Firewall",
            "purpose": "Prevent disclosure of personal data and credentials to upstream AI model providers",
            "data_categories": categories,
            "special_categories_art9": special,
            "data_subjects": "End users of the AI application and persons mentioned in their prompts",
            "recipients": "Upstream LLM provider receives pseudonymized/masked text only",
            "technical_measures": [m for m in (
                "Pseudonymization with per-session in-memory token vault" if any(f.category != SECRET for f in findings) else None,
                "Irreversible masking of credentials" if any(f.category == SECRET for f in findings) else None,
                "Audit log stores redacted text only",
            ) if m],
            "retention": "Token vault held in memory for the session lifetime only",
            "legal_basis_note": (
                "Art. 9 data present: an Art. 9(2) condition (e.g. explicit consent) is required"
                if special else "Determine the Art. 6(1) basis for the processing"
            ),
            "status": "DRAFT - requires DPO review",
        }


sensitive_engine = SensitiveDataEngine()
