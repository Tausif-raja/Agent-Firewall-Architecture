"""Egress checks: catch exfiltration that got past ingress detection.

Runs on model output before it is shown to the user, and on tool-call
arguments before they execute.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import urlparse

from prompt_injection.segment import SessionContext, ToolCall

_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_LINK = re.compile(r"(?<!!)\[([^\]]*)\]\(\s*<?([^)\s>]+)>?(?:\s+\"[^\"]*\")?\s*\)")
_MD_REF_DEF = re.compile(r"(?m)^\s*\[([^\]]+)\]:\s*(\S+)")
_HTML_SRC = re.compile(r"<(img|iframe|script|link|source|video|audio)\b[^>]*\b(src|href|srcset)\s*=\s*[\"']?([^\"'\s>]+)", re.I)

# Credential shapes worth stopping on the way out. PII is handled by the Presidio stage.
_SECRET_PATTERNS = {
    "openai_key": re.compile(r"\bsk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}"),
    "aws_access_key": re.compile(r"\b(AKIA|ASIA)[0-9A-Z]{16}\b"),
    "github_token": re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b"),
    "google_api_key": re.compile(r"\bAIza[0-9A-Za-z_-]{35}\b"),
    "slack_token": re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b"),
    "private_key": re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    "jwt": re.compile(r"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\b"),
}


@dataclass
class EgressResult:
    text: str                                       # sanitized output
    blocked: bool = False                           # hard stop: do not release at all
    findings: list[str] = field(default_factory=list)


def _host_allowed(url: str, allowlist: Iterable[str]) -> bool:
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    if not host:
        return True  # relative link, stays on our origin
    return any(host == d or host.endswith("." + d) for d in allowlist)


class EgressFilter:
    def __init__(self, allowed_domains: Iterable[str] = (), block_on_secrets: bool = True):
        self.allowed_domains = [d.lower().lstrip(".") for d in allowed_domains]
        self.block_on_secrets = block_on_secrets

    @staticmethod
    def _seen_in_context(url: str, ctx: SessionContext) -> bool:
        # A URL copied verbatim from the inputs can't carry data the attacker didn't already
        # have. A URL the model composed can: in the query, the path, or the subdomain.
        return any(url in t for t in (ctx.user_task, *ctx.user_texts, *ctx.untrusted_texts))

    def _common(self, text: str, ctx: SessionContext, findings: list[str]) -> bool:
        blocked = False
        if ctx.canary and ctx.canary in text:
            findings.append("canary_leak: system prompt content in output")
            blocked = True
        for name, rx in _SECRET_PATTERNS.items():
            if rx.search(text):
                findings.append(f"secret:{name}")
                blocked = blocked or self.block_on_secrets
        return blocked

    def check_output(self, text: str, ctx: SessionContext) -> EgressResult:
        findings: list[str] = []
        blocked = self._common(text, ctx, findings)

        def strip_image(m: re.Match) -> str:
            url = m.group(2)
            if _host_allowed(url, self.allowed_domains):
                return m.group(0)
            findings.append(f"external_image_removed:{urlparse(url).hostname}")
            return f"[image removed: {m.group(1) or 'external image'}]"

        def defang_link(m: re.Match) -> str:
            url = m.group(2)
            if _host_allowed(url, self.allowed_domains) or self._seen_in_context(url, ctx):
                return m.group(0)
            host = urlparse(url).hostname
            findings.append(f"novel_external_link_removed:{host}")
            return f"{m.group(1)} ({host})"

        def strip_html(m: re.Match) -> str:
            url = m.group(3)
            if _host_allowed(url, self.allowed_domains):
                return m.group(0)
            findings.append(f"external_html_{m.group(1).lower()}_removed:{urlparse(url).hostname}")
            return "[embedded content removed]"

        def strip_refdef(m: re.Match) -> str:
            url = m.group(2)
            if _host_allowed(url, self.allowed_domains) or self._seen_in_context(url, ctx):
                return m.group(0)
            findings.append(f"external_reference_removed:{urlparse(url).hostname}")
            return ""

        out = _MD_IMAGE.sub(strip_image, text)
        out = _MD_LINK.sub(defang_link, out)
        out = _MD_REF_DEF.sub(strip_refdef, out)
        out = _HTML_SRC.sub(strip_html, out)
        return EgressResult(text=out, blocked=blocked, findings=findings)

    def check_tool_call(self, call: ToolCall, ctx: SessionContext) -> EgressResult:
        findings: list[str] = []
        flat = json.dumps(call.args, default=str, ensure_ascii=False)
        blocked = self._common(flat, ctx, findings)
        return EgressResult(text=flat, blocked=blocked, findings=findings)
