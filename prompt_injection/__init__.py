"""Prompt injection defense for the Agent Firewall. See docs/PROMPT_INJECTION.md."""

from prompt_injection.segment import (
    Action, Channel, Decision, Label, Segment, SessionContext, ToolCall, ToolDecision, Trust, Verdict,
)

__all__ = [
    "Action", "Channel", "Decision", "Label", "Segment", "SessionContext",
    "ToolCall", "ToolDecision", "Trust", "Verdict", "InjectionGuard",
]


def __getattr__(name: str):
    # Lazy so that importing the legacy modules in this package stays cheap.
    if name == "InjectionGuard":
        from prompt_injection.guard import InjectionGuard
        return InjectionGuard
    raise AttributeError(name)
