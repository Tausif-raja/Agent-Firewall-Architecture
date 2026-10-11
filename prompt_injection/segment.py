"""Core data types shared by every layer of the injection guard."""

from __future__ import annotations

import secrets
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Optional


class Trust(IntEnum):
    """How much the firewall trusts a piece of content.

    Assigned by integration code based on *how the content arrived*,
    never based on what the content claims about itself.
    """
    UNTRUSTED = 0   # web pages, uploaded files, emails, third-party tool output
    SEMI = 1        # internal RAG corpus, first-party APIs
    USER = 2        # authenticated end-user message
    SYSTEM = 3      # developer/system prompt, firewall-authored text


class Channel(str, Enum):
    USER_MSG = "user_msg"
    RETRIEVED = "retrieved"
    TOOL_RESULT = "tool_result"
    AGENT_MSG = "agent_msg"
    SYSTEM = "system"


class Action(str, Enum):
    ALLOW = "allow"
    SANITIZE = "sanitize"      # drop/redact the offending span, keep the rest
    QUARANTINE = "quarantine"  # drop the segment and quarantine its source
    BLOCK = "block"            # drop the segment / refuse the request


class Label(str, Enum):
    BENIGN = "benign"
    SUSPICIOUS = "suspicious"
    INJECTION = "injection"


@dataclass
class Segment:
    text: str
    source: str               # "user", "rag:<doc_id>", "tool:<name>", "web:<url>", "file:<sha256>"
    trust: Trust
    channel: Channel
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass
class Verdict:
    score: float                      # 0.0 (benign) .. 1.0 (injection)
    label: Label
    layer: str                        # layer that decided: "rules" | "classifier" | "judge" | "policy"
    action: Action
    reasons: list[str] = field(default_factory=list)
    scores: dict[str, float] = field(default_factory=dict)  # per-layer scores, for logging/tuning
    sanitized_text: Optional[str] = None                     # set when action == SANITIZE

    @property
    def allowed(self) -> bool:
        return self.action in (Action.ALLOW, Action.SANITIZE)


@dataclass
class ToolCall:
    name: str
    args: dict[str, Any]


class Decision(str, Enum):
    ALLOW = "allow"
    REQUIRE_APPROVAL = "require_approval"
    BLOCK = "block"


@dataclass
class ToolDecision:
    decision: Decision
    reason: str


@dataclass
class SessionContext:
    """Per-session state the guard needs across turns."""
    session_id: str
    user_task: str = ""
    tainted: bool = False
    taint_sources: list[str] = field(default_factory=list)
    # Text of every user / untrusted segment that entered context, used for argument provenance.
    user_texts: list[str] = field(default_factory=list)
    untrusted_texts: list[str] = field(default_factory=list)
    # Cumulative suspicion across turns, to catch attacks split over several messages.
    suspicion: float = 0.0
    canary: str = field(default_factory=lambda: f"cnry-{secrets.token_hex(8)}")
    nonce: str = field(default_factory=lambda: secrets.token_hex(4))
