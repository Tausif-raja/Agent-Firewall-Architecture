"""Taint tracking and tool-call authorization.

This is the safety boundary that still holds when detection misses. Once
untrusted content has entered a session, privileged tool calls are checked
against (a) the tool's risk class, (b) where their destination arguments came
from, and (c) optionally, whether the call fits what the user asked for.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional
from urllib.parse import urlparse

from prompt_injection.segment import Decision, SessionContext, ToolCall, ToolDecision

READ, WRITE, EXTERNAL, CRITICAL = "read", "write", "external", "critical"
_RISKS = (READ, WRITE, EXTERNAL, CRITICAL)

# Argument names whose values decide *where* data or effects go.
DEFAULT_DESTINATION_KEYS = frozenset({
    "to", "cc", "bcc", "recipient", "recipients", "email", "address", "phone", "number",
    "url", "uri", "endpoint", "webhook", "host", "domain", "path", "file", "filename",
    "destination", "dest", "target", "account", "account_id", "iban", "wallet", "channel",
    "repo", "repository", "branch",
})

_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_URL = re.compile(r"https?://[^\s\"'<>)\]]+", re.I)

# fn(user_task, call) -> True if the call plausibly serves the task. It must NOT
# be given untrusted content, otherwise the injection can argue for itself.
IntentChecker = Callable[[str, ToolCall], bool]


def _leaves(value: Any, key: str = "") -> Iterable[tuple[str, str]]:
    if isinstance(value, dict):
        for k, v in value.items():
            yield from _leaves(v, str(k).lower())
    elif isinstance(value, (list, tuple, set)):
        for v in value:
            yield from _leaves(v, key)
    elif value is not None:
        yield key, str(value)


def _mentions(haystack: str, atom: str) -> bool:
    # Whole-token match, so "bob" doesn't match "bobsled" and "evil.com" doesn't match "notevil.com".
    return re.search(r"(?<![\w.@-])" + re.escape(atom) + r"(?![\w@-])", haystack) is not None


def destination_atoms(call: ToolCall, destination_keys: frozenset[str] = DEFAULT_DESTINATION_KEYS) -> set[str]:
    """Values in a tool call that determine where data/effects go."""
    atoms: set[str] = set()
    for key, val in _leaves(call.args):
        atoms.update(m.lower() for m in _EMAIL.findall(val))
        for url in _URL.findall(val):
            host = (urlparse(url).hostname or "").lower()
            if host:
                atoms.add(host)
        if key in destination_keys and 3 <= len(val.strip()) <= 300:
            atoms.add(val.strip().lower())
    return atoms


@dataclass
class ToolPolicy:
    risks: dict[str, str] = field(default_factory=dict)  # tool name -> read|write|external|critical
    default_risk: str = EXTERNAL                          # unknown tools are treated as dangerous
    destination_keys: frozenset[str] = DEFAULT_DESTINATION_KEYS
    intent_checker: Optional[IntentChecker] = None
    # Destinations that are always acceptable (your own domains, internal addresses).
    trusted_destinations: frozenset[str] = frozenset()
    # Break the "lethal trifecta": with a tainted session, also require approval for
    # every external call, even when provenance and intent checks pass.
    approve_all_external_when_tainted: bool = False

    def __post_init__(self) -> None:
        for name, risk in self.risks.items():
            if risk not in _RISKS:
                raise ValueError(f"tool {name}: unknown risk {risk!r}")
        self.trusted_destinations = frozenset(d.lower() for d in self.trusted_destinations)

    def risk_of(self, tool: str) -> str:
        return self.risks.get(tool, self.default_risk)

    def _trusted(self, atom: str) -> bool:
        return any(atom == d or atom.endswith("." + d) or atom.endswith("@" + d) for d in self.trusted_destinations)

    def unvouched_destinations(self, call: ToolCall, ctx: SessionContext) -> tuple[list[str], list[str]]:
        """Split destinations the user never supplied into (seen in untrusted content, unknown origin).

        Allowlist, not denylist: a destination is fine only if the user (or config) supplied it.
        Checking "did it appear in untrusted content?" alone misses obfuscated payloads
        such as "attacker (at) evil (dot) example", which the model can reassemble.
        """
        user = "\n".join([ctx.user_task, *ctx.user_texts]).lower()
        untrusted = "\n".join(ctx.untrusted_texts).lower()
        from_untrusted, unknown = [], []
        for a in sorted(destination_atoms(call, self.destination_keys)):
            if self._trusted(a) or _mentions(user, a):
                continue
            (from_untrusted if _mentions(untrusted, a) else unknown).append(a)
        return from_untrusted, unknown

    def authorize(self, call: ToolCall, ctx: SessionContext) -> ToolDecision:
        risk = self.risk_of(call.name)
        if risk == READ:
            return ToolDecision(Decision.ALLOW, "read-only tool")
        if not ctx.tainted:
            return ToolDecision(Decision.ALLOW, "session not tainted")
        if risk == CRITICAL:
            return ToolDecision(Decision.REQUIRE_APPROVAL, f"critical tool in tainted session (sources: {ctx.taint_sources[:3]})")

        from_untrusted, unknown = self.unvouched_destinations(call, ctx)
        if from_untrusted:
            decision = Decision.BLOCK if risk == EXTERNAL else Decision.REQUIRE_APPROVAL
            return ToolDecision(decision, f"destination taken from untrusted content, not from the user: {from_untrusted[:5]}")

        if self.intent_checker is not None:
            try:
                aligned = self.intent_checker(ctx.user_task, call)
            except Exception as e:
                return ToolDecision(Decision.REQUIRE_APPROVAL, f"intent check failed: {e}")
            if not aligned:
                return ToolDecision(Decision.BLOCK, "tool call does not match the user's task")

        if unknown:
            return ToolDecision(Decision.REQUIRE_APPROVAL, f"destination not supplied by the user: {unknown[:5]}")

        if risk == EXTERNAL and self.approve_all_external_when_tainted:
            return ToolDecision(Decision.REQUIRE_APPROVAL, "external call in tainted session")
        return ToolDecision(Decision.ALLOW, "provenance and intent checks passed")
