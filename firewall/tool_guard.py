"""
Dual-gate tool guard.

Gate 1 (pre-execution): allowlist, role check, argument validation,
destructive-operation detection, human-in-the-loop (HITL) requirement.
Execution: run with a timeout.
Gate 2 (post-execution): strip leaked credentials from the output, truncate.

Unknown tools are denied by default; register them with ToolSecurityGuard.register().
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

from firewall.sensitive_data_engine import sensitive_engine

logger = logging.getLogger("agent_firewall.tool_guard")

APPROVED, BLOCKED, REQUIRES_HITL = "APPROVED", "BLOCKED", "REQUIRES_HITL"
SUCCESS, TIMEOUT, FAILED = "SUCCESS", "TIMEOUT", "FAILED"

MAX_ARGS_BYTES = 16_000

DESTRUCTIVE_PATTERNS = [
    (re.compile(r"\b(drop|truncate)\s+(table|database|schema)\b", re.I), "destructive SQL (DROP/TRUNCATE)"),
    (re.compile(r"\bdelete\s+from\s+\w+\s*(;|$)(?![\s\S]*\bwhere\b)", re.I), "DELETE without WHERE"),
    (re.compile(r"\balter\s+table\b[\s\S]*\bdrop\b", re.I), "destructive ALTER TABLE"),
    (re.compile(r"\brm\s+-[a-z]*[rf][a-z]*\b", re.I), "recursive/forced file deletion"),
    (re.compile(r"\b(mkfs|dd\s+if=|shutdown|reboot|halt)\b", re.I), "system-destructive command"),
    (re.compile(r":\(\)\s*\{\s*:\|:&\s*\};:"), "fork bomb"),
    (re.compile(r"\b(curl|wget)\b[^|]*\|\s*(ba|z)?sh\b", re.I), "download-and-execute"),
    (re.compile(r"\bchmod\s+(-R\s+)?777\b", re.I), "world-writable permissions"),
]

_SAFE_ARITHMETIC = re.compile(r"^[\d\s+\-*/().%]+$")


def _validate_calculator(args: Dict[str, Any]) -> Optional[str]:
    expr = str(args.get("expression", ""))
    if len(expr) > 200:
        return "expression too long"
    if not _SAFE_ARITHMETIC.match(expr):
        # Letters/underscores would allow attribute access (e.g. ().__class__...) inside eval().
        return "only numbers and + - * / % ( ) are allowed"
    if "**" in expr:
        return "exponentiation is not allowed (resource exhaustion)"
    return None


@dataclass
class ToolPolicy:
    name: str
    risk_level: str = "low"                 # low | medium | high | critical
    requires_hitl: bool = False
    blocked: bool = False                   # never executable through the proxy
    allowed_roles: tuple = ("standard", "admin")
    timeout_s: float = 10.0
    max_output_chars: int = 8_000
    validator: Optional[Callable[[Dict[str, Any]], Optional[str]]] = None
    description: str = ""


@dataclass
class ToolAuthorizationResult:
    authorized: bool
    status: str                             # APPROVED | BLOCKED | REQUIRES_HITL
    reason: Optional[str] = None
    tool_name: str = ""
    risk_level: str = "unknown"


@dataclass
class ToolExecutionResult:
    status: str                             # SUCCESS | TIMEOUT | FAILED
    output: Any = None
    latency_ms: float = 0.0
    estimated_tokens: int = 0
    error_message: Optional[str] = None
    secrets_redacted: int = 0
    truncated: bool = False


DEFAULT_POLICIES = [
    ToolPolicy("fetch_weather", "low", description="Read-only weather lookup"),
    ToolPolicy("search_knowledge_base", "low", description="Read-only internal search"),
    ToolPolicy("calculator", "low", validator=_validate_calculator, description="Arithmetic only"),
    ToolPolicy("execute_sql", "high", requires_hitl=True, description="Database access; needs human approval"),
    ToolPolicy("bash_command", "critical", blocked=True, description="Shell access is never proxied"),
]


class ToolSecurityGuard:
    def __init__(self, policies=DEFAULT_POLICIES, max_workers: int = 8):
        self.policies: Dict[str, ToolPolicy] = {p.name: p for p in policies}
        self._pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix="tool-guard")

    def register(self, policy: ToolPolicy) -> None:
        self.policies[policy.name] = policy

    # ----------------------------------------------------------------- gate 1

    def pre_execution_check(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        user_role: str = "standard",
        session_id: str = "",
        hitl_approved: bool = False,
    ) -> ToolAuthorizationResult:
        def deny(reason: str, status: str = BLOCKED, risk: str = "unknown") -> ToolAuthorizationResult:
            return ToolAuthorizationResult(False, status, reason, tool_name, risk)

        policy = self.policies.get(tool_name)
        if policy is None:
            return deny(f"Tool '{tool_name}' is not on the allowlist")
        if policy.blocked:
            return deny(f"Tool '{tool_name}' is blocked by policy: {policy.description}", risk=policy.risk_level)
        if user_role not in policy.allowed_roles:
            return deny(f"Role '{user_role}' may not call '{tool_name}'", risk=policy.risk_level)
        if not isinstance(tool_args, dict):
            return deny("Tool arguments must be a JSON object", risk=policy.risk_level)
        try:
            serialized = json.dumps(tool_args, default=str)
        except (TypeError, ValueError):
            return deny("Tool arguments are not serializable", risk=policy.risk_level)
        if len(serialized) > MAX_ARGS_BYTES:
            return deny("Tool arguments too large", risk=policy.risk_level)

        for rx, label in DESTRUCTIVE_PATTERNS:
            if rx.search(serialized):
                return deny(f"Destructive operation detected: {label}", risk="critical")

        if policy.validator is not None:
            problem = policy.validator(tool_args)
            if problem:
                return deny(f"Invalid arguments for '{tool_name}': {problem}", risk=policy.risk_level)

        if policy.requires_hitl and not hitl_approved:
            return deny(f"'{tool_name}' is {policy.risk_level}-risk and requires human approval",
                        status=REQUIRES_HITL, risk=policy.risk_level)

        return ToolAuthorizationResult(True, APPROVED, None, tool_name, policy.risk_level)

    def screen_proposed_call(self, tool_name: str, tool_args: Dict[str, Any], user_role: str = "standard") -> Optional[str]:
        """Check a tool call the *model* proposed, which the client agent will execute.

        Registered tools get the full Gate 1 check. Unregistered tools belong to the
        client, so they are not denied for being unknown, but destructive arguments
        are still blocked. Returns a block reason, or None if the call may proceed.
        """
        if tool_name in self.policies:
            r = self.pre_execution_check(tool_name, tool_args, user_role=user_role)
            return None if r.authorized else f"{r.status}: {r.reason}"
        try:
            serialized = json.dumps(tool_args, default=str)
        except (TypeError, ValueError):
            return "tool arguments are not serializable"
        for rx, label in DESTRUCTIVE_PATTERNS:
            if rx.search(serialized):
                return f"Destructive operation detected: {label}"
        return None

    # ---------------------------------------------------- execution + gate 2

    def execute_with_guard(
        self, tool_callable: Callable[..., Any], tool_name: str, tool_args: Dict[str, Any]
    ) -> ToolExecutionResult:
        policy = self.policies.get(tool_name) or ToolPolicy(tool_name)
        start = time.perf_counter()
        future = self._pool.submit(tool_callable, **tool_args)
        try:
            output = future.result(timeout=policy.timeout_s)
        except FuturesTimeout:
            future.cancel()  # best effort: a running thread can't be killed
            return ToolExecutionResult(TIMEOUT, None, (time.perf_counter() - start) * 1000,
                                       error_message=f"Tool exceeded {policy.timeout_s}s timeout")
        except Exception as e:
            return ToolExecutionResult(FAILED, None, (time.perf_counter() - start) * 1000,
                                       error_message=f"{e.__class__.__name__}: {str(e)[:200]}")
        latency_ms = (time.perf_counter() - start) * 1000

        output, redacted = self._redact(output)
        output, truncated = self._truncate(output, policy.max_output_chars)
        if redacted:
            logger.warning("Gate 2: redacted %d credential(s) from '%s' output", redacted, tool_name)
        size = len(json.dumps(output, default=str))
        return ToolExecutionResult(SUCCESS, output, latency_ms, estimated_tokens=max(1, size // 4),
                                   secrets_redacted=redacted, truncated=truncated)

    def _redact(self, value: Any) -> tuple[Any, int]:
        if isinstance(value, str):
            return sensitive_engine.redact_secrets(value)
        if isinstance(value, dict):
            total, out = 0, {}
            for k, v in value.items():
                out[k], n = self._redact(v)
                total += n
            return out, total
        if isinstance(value, (list, tuple)):
            total, out = 0, []
            for v in value:
                r, n = self._redact(v)
                out.append(r)
                total += n
            return out, total
        return value, 0

    @staticmethod
    def _truncate(value: Any, limit: int) -> tuple[Any, bool]:
        text = value if isinstance(value, str) else json.dumps(value, default=str)
        if len(text) <= limit:
            return value, False
        return text[:limit] + "… [truncated by firewall]", True
