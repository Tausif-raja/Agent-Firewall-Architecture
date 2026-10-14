import os
import re
import time
import uuid
import json
import logging
import threading
import importlib.util
from collections import OrderedDict
from datetime import datetime, date
from typing import Dict, Any, List, Optional, Tuple

from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), ".env"))
load_dotenv(os.path.join(os.path.dirname(__file__), "..", ".env"))

if os.getenv("OPENROUTER_KEY") and not os.getenv("OPENROUTER_API_KEY"):
    os.environ["OPENROUTER_API_KEY"] = os.getenv("OPENROUTER_KEY")

from litellm import completion, cost_calculator
from sqlalchemy.orm import Session

from Agent_Firewall.database.database import SessionLocal
from Agent_Firewall.database.models import (
    RequestLog,
    ContentLog,
    MetricsLog,
    DailyStats,
    ToolExecutionLog,
    ApiKeyConfig,
)
from firewall.tool_guard import (
    ToolSecurityGuard,
    ToolPolicy,
    ToolAuthorizationResult,
    ToolExecutionResult,
)
from firewall.sensitive_data_engine import sensitive_engine, SensitiveFinding
from Agent_Firewall.database.models import utcnow
from Agent_Firewall.security import hash_api_key
from prompt_injection import spotlight
from prompt_injection.capabilities import READ, ToolPolicy as CapabilityPolicy
from prompt_injection.classifier import HFInjectionClassifier, NullClassifier
from prompt_injection.guard import InjectionGuard
from prompt_injection.policy import PolicyConfig
from prompt_injection.segment import Action, Channel, Decision, Segment, SessionContext, Trust
from prompt_injection.segment import ToolCall as GuardToolCall

logger = logging.getLogger("agent_firewall.engine")

MAX_TRACKED_SESSIONS = 10_000   # kill-switch memory bound
LOOP_HISTORY_WINDOW = 20
UPSTREAM_AUTH_ERRORS = ("api key", "expired", "401", "authentication", "unauthorized", "forbidden", "403")


def _message_text(content: Any) -> str:
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if isinstance(p, dict))
    return "" if content is None else str(content)


def _tool_call_to_dict(tc: Any) -> Dict[str, Any]:
    if hasattr(tc, "model_dump"):
        d = tc.model_dump()
    elif isinstance(tc, dict):
        d = dict(tc)
    else:
        d = dict(vars(tc))
    fn = d.get("function") or {}
    if not isinstance(fn, dict):
        fn = fn.model_dump() if hasattr(fn, "model_dump") else dict(vars(fn))
    return {"id": d.get("id"), "type": d.get("type") or "function",
            "function": {"name": fn.get("name") or "", "arguments": fn.get("arguments") or "{}"}}


def _parse_args(raw: Any) -> Dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(raw or "{}")
    except (TypeError, ValueError):
        return {"_raw": str(raw)}
    return parsed if isinstance(parsed, dict) else {"value": parsed}


def _build_injection_guard(tool_guard: ToolSecurityGuard) -> InjectionGuard:
    """Injection guard for tool results and proposed tool calls.

    INJECTION_CLASSIFIER=auto (default) uses the Hugging Face classifier when
    transformers+torch are installed; otherwise it runs on rules only.
    """
    mode = os.getenv("INJECTION_CLASSIFIER", "auto").strip().lower()
    hf_available = all(importlib.util.find_spec(m) for m in ("transformers", "torch"))
    cfg = PolicyConfig()
    if mode == "none" or (mode == "auto" and not hf_available):
        classifier = NullClassifier()
        cfg.classifier_fail_mode = {t: "rules_only" for t in cfg.classifier_fail_mode}
        logger.warning("Injection classifier disabled (INJECTION_CLASSIFIER=%s, transformers/torch installed=%s): "
                       "tool results are screened by rules only", mode, hf_available)
    else:
        classifier = HFInjectionClassifier()
    read_only = {name: READ for name, pol in tool_guard.policies.items() if pol.risk_level == "low" and not pol.blocked}
    return InjectionGuard(classifier=classifier, config=cfg, tool_policy=CapabilityPolicy(risks=read_only))

# -------------------------------------------------------------
# Direct Patterns for Prompt Injection & Harmful Instructions
# -------------------------------------------------------------
PROMPT_INJECTION_PATTERNS = [
    r"(?i)\b(ignore\s+previous\s+(instructions|prompts|rules)|disregard\s+earlier|forget\s+all\s+rules)\b",
    r"(?i)\b(reveal\s+(system\s+prompt|hidden\s+prompt|secret|api[_-]?key|password|credential))\b",
    r"(?i)\b(show\s+system\s+instructions|print\s+initial\s+prompt|developer\s+mode\s+enabled)\b",
    r"(?i)\b(jailbreak|do\s+anything\s+now|DAN\s+mode|unfiltered\s+mode)\b",
    r"(?i)\b(act\s+as\s+admin|bypass\s+safety\s+filter|override\s+guidelines)\b",
    r"(?i)\b(how\s+to\s+make\s+(a\s+bomb|explosive|nuclear\s+bomb|chemical\s+weapon|malware|ransomware))\b",
    r"(?i)\b(steal\s+passwords|dump\s+database|extract\s+browser\s+credentials|dump\s+memory)\b"
]


class IngressSecurityResult:
    def __init__(
        self,
        is_safe: bool,
        threat_type: Optional[str] = None,
        matched_pattern: Optional[str] = None,
        score: float = 0.0,
        verdict: Optional[str] = None,
        sensitive_findings: Optional[List[Dict[str, Any]]] = None,
        redacted_prompt: Optional[str] = None
    ):
        self.is_safe = is_safe
        self.threat_type = threat_type
        self.matched_pattern = matched_pattern
        self.score = score
        self.verdict = verdict or ("SAFE" if is_safe else "UNSAFE")
        self.sensitive_findings = sensitive_findings or []
        self.redacted_prompt = redacted_prompt or ""

    def to_dict(self):
        return {
            "is_safe": self.is_safe,
            "verdict": self.verdict,
            "threat_type": self.threat_type,
            "matched_pattern": self.matched_pattern,
            "score": self.score,
            "sensitive_findings": self.sensitive_findings,
            "redacted_prompt": self.redacted_prompt
        }


class FirewallCoreEngine:
    """
    Unified Systematic Agent Firewall Middleware:
    1. Ingress Prompt Inspection (Injection, Jailbreak, Safety, Presidio PII/Secret Guard)
    2. Tool Execution Guard (Dual-Gate Proxy: Pre-Exec & Post-Exec)
    3. Loop Detector & Kill Switch
    4. Cost, Token & Latency Telemetry Audit Logging
    """

    def __init__(self):
        self.tool_guard = ToolSecurityGuard()
        self.sensitive_engine = sensitive_engine
        self.injection_guard = _build_injection_guard(self.tool_guard)
        # Keyed by "<tenant>:<session>" so clients can never collide; bounded LRU.
        self.session_histories: "OrderedDict[str, List[str]]" = OrderedDict()
        self._history_lock = threading.Lock()

    def inspect_prompt(self, prompt: str, check_sensitive: bool = True) -> IngressSecurityResult:
        """
        Ingress Firewall: Scans prompt against direct injection, harmful commands, jailbreaks,
        and sensitive data/PII (Microsoft Presidio & native secret engines).
        When sensitive data is found, it redacts it, logs the details, and returns verdict UNSAFE.
        """
        if not prompt or not isinstance(prompt, str):
            return IngressSecurityResult(is_safe=True, verdict="SAFE", score=0.0)

        cleaned_text = prompt.strip()

        # Step 1: Prompt Injection / Jailbreak / Harmful Command Scan
        for pattern in PROMPT_INJECTION_PATTERNS:
            match = re.search(pattern, cleaned_text)
            if match:
                threat = "Prompt Injection / Jailbreak Attempt"
                if any(w in match.group(0).lower() for w in ["bomb", "explosive", "weapon", "malware", "ransomware"]):
                    threat = "Harmful / Prohibited Content"
                elif any(w in match.group(0).lower() for w in ["api", "key", "password", "credential", "system"]):
                    threat = "Prompt / Secret Exfiltration Attempt"

                return IngressSecurityResult(
                    is_safe=False,
                    verdict="UNSAFE",
                    threat_type=threat,
                    matched_pattern=match.group(0),
                    score=0.98
                )

        # Step 2: Microsoft Presidio & Sensitive Data Guard (Redaction + Unsafe Verdict).
        # Callers that already ran the sensitive-data stage pass check_sensitive=False;
        # otherwise PII that was just tokenized would be re-detected and blocked here.
        if not check_sensitive:
            return IngressSecurityResult(is_safe=True, verdict="SAFE", score=0.05)
        redacted_text, findings, _ = self.sensitive_engine.sanitize_or_pseudonymize(
            text=cleaned_text,
            session_id="ingress_inspect",
            mode="mask"
        )
        if findings:
            types = ", ".join(sorted(set(f.entity_type for f in findings)))
            findings_list = [f.to_dict() for f in findings]
            logger.warning(
                f"🚨 [Agent Firewall Ingress]: Sensitive information detected! Redacting. "
                f"Entities: {types} | Findings: {findings_list}"
            )
            return IngressSecurityResult(
                is_safe=False,
                verdict="UNSAFE",
                threat_type=f"Sensitive Data & PII Detected ({types})",
                matched_pattern=types,
                score=0.95,
                sensitive_findings=findings_list,
                redacted_prompt=redacted_text
            )

        return IngressSecurityResult(is_safe=True, verdict="SAFE", score=0.05)

    def check_kill_switch(self, session_id: str, new_response: str, fuzzy: bool = True) -> Dict[str, Any]:
        """
        Monitors agent responses to detect and terminate infinite runaway loops.
        fuzzy=False disables the lexical-prefix layer: for tool calls only exact
        repetition (same tool, same arguments) counts, since calling the same tool
        with different arguments is usually progress (e.g. iterating over a list).
        """
        if not session_id:
            session_id = "default"

        with self._history_lock:
            history = self.session_histories.setdefault(session_id, [])
            self.session_histories.move_to_end(session_id)
            while len(self.session_histories) > MAX_TRACKED_SESSIONS:
                self.session_histories.popitem(last=False)
            history.append(new_response)
            del history[:-LOOP_HISTORY_WINDOW]
            history = list(history)

        # Layer A: Exact match loop detection over window of 3
        if len(history) >= 3:
            last_3 = history[-3:]
            if len(set(last_3)) == 1:
                return {
                    "loop_detected": True,
                    "layer": "A",
                    "reason": "Exact repeat loop detected in consecutive responses."
                }

        # Layer B: Near-identical repetition (substring containment or simple lexical similarity)
        if fuzzy and len(history) >= 4:
            recent = history[-4:]
            if len(recent[0]) > 20 and all(r == recent[0] or (len(r) > 15 and r[:30] == recent[0][:30]) for r in recent):
                return {
                    "loop_detected": True,
                    "layer": "B",
                    "reason": "High repetitive lexical similarity across multiple turns."
                }

        return {"loop_detected": False, "layer": None, "reason": "OK"}

    def reset_session(self, session_id: str):
        with self._history_lock:
            self.session_histories.pop(session_id, None)

    def _sanitize_content(self, content: Any, scope: str, mode: str):
        """Sanitize a message's content, string or multimodal list (each text part separately).

        Returns (joined_sanitized_text, findings, should_block, new_content).
        """
        if isinstance(content, list):
            new_parts, texts, findings, block = [], [], [], False
            for part in content:
                if isinstance(part, dict) and isinstance(part.get("text"), str):
                    clean, f, b = self.sensitive_engine.sanitize_or_pseudonymize(text=part["text"], session_id=scope, mode=mode)
                    part = {**part, "text": clean}
                    findings += f
                    block = block or b
                    texts.append(clean)
                new_parts.append(part)
            return " ".join(texts), findings, block, new_parts
        clean, findings, block = self.sensitive_engine.sanitize_or_pseudonymize(text=_message_text(content), session_id=scope, mode=mode)
        return clean, findings, block, clean

    # ------------------------------------------------------------------
    # Indirect injection: tool results in, tool calls out
    # ------------------------------------------------------------------
    def _screen_tool_messages(self, messages: List[Dict[str, Any]], ctx: SessionContext):
        """Scan role=tool messages as untrusted content; wrap or remove them before the LLM sees them."""
        idx = [i for i, m in enumerate(messages) if m.get("role") in ("tool", "function")]
        if not idx:
            return messages, []
        segments = [
            Segment(_message_text(messages[i].get("content")),
                    f"tool:{messages[i].get('name') or messages[i].get('tool_call_id') or 'unknown'}",
                    Trust.UNTRUSTED, Channel.TOOL_RESULT)
            for i in idx
        ]
        verdicts = self.injection_guard.scan_many(segments, ctx)
        out = [dict(m) for m in messages]
        findings = []
        for i, seg, v in zip(idx, segments, verdicts):
            out[i]["content"] = self.injection_guard.wrap(seg, ctx, v)
            if v.action is not Action.ALLOW:
                findings.append({"source": seg.source, "action": v.action.value, "label": v.label.value,
                                 "score": round(v.score, 3), "reasons": v.reasons[:5]})

        # Tell the model that wrapped content is data, not instructions.
        rule = spotlight.system_rule(ctx)
        if out and out[0].get("role") == "system" and isinstance(out[0].get("content"), str):
            out[0]["content"] = f"{out[0]['content']}\n\n{rule}"
        else:
            out.insert(0, {"role": "system", "content": rule})
        return out, findings

    def _screen_tool_calls(self, raw_calls: List[Any], ctx: SessionContext):
        """Gate the model's proposed tool calls before the client agent executes them."""
        kept, blocked = [], []
        for tc in raw_calls:
            call = _tool_call_to_dict(tc)
            name = call["function"]["name"]
            args = _parse_args(call["function"]["arguments"])
            reason = self.tool_guard.screen_proposed_call(name, args)
            if reason is None:
                decision = self.injection_guard.authorize_tool_call(GuardToolCall(name, args), ctx)
                if decision.decision is not Decision.ALLOW:
                    reason = f"{decision.decision.value}: {decision.reason}"
            if reason:
                blocked.append({"tool": name, "reason": reason})
            else:
                kept.append(call)
        return kept, blocked

    def execute_tool_proxy(
        self,
        tool_name: str,
        tool_args: Dict[str, Any],
        tool_callable=None,
        user_role: str = "standard",
        session_id: str = "default_session",
        db: Optional[Session] = None,
        request_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Dual-Gate Tool Execution:
        Gate 1: Authorization, Parameter Validation & HITL Check
        Execution: Sandboxed with Timeout
        Gate 2: Leaked Secret Inspection & Output Truncation
        """
        req_id = request_id or str(uuid.uuid4())
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        try:
            # Gate 1
            auth_result = self.tool_guard.pre_execution_check(
                tool_name=tool_name,
                tool_args=tool_args,
                user_role=user_role,
                session_id=session_id
            )

            if not auth_result.authorized:
                self._record_tool_log(db, req_id, tool_name, tool_args, auth_result)
                return {
                    "success": False,
                    "status": auth_result.status,
                    "error": auth_result.reason,
                    "gate": "Gate 1 (Pre-Execution Authorization)"
                }

            # Safe Execution
            if tool_callable is None:
                # Built-in mock handler for playground tests
                def mock_handler(**kwargs):
                    if tool_name == "fetch_weather":
                        return {"location": kwargs.get("location", "New York"), "temp": "72F", "condition": "Sunny"}
                    elif tool_name == "calculator":
                        expr = kwargs.get("expression", "2+2")
                        return {"expression": expr, "result": eval(expr, {"__builtins__": None}, {})}
                    elif tool_name == "search_knowledge_base":
                        return {"query": kwargs.get("query"), "results": ["Firewall policy doc #12", "Security checklist v2.4"]}
                    else:
                        return {"status": "ok", "args": kwargs}
                tool_callable = mock_handler

            exec_result = self.tool_guard.execute_with_guard(
                tool_callable=tool_callable,
                tool_name=tool_name,
                tool_args=tool_args
            )

            self._record_tool_log(db, req_id, tool_name, tool_args, auth_result, exec_result)

            return {
                "success": exec_result.status == "SUCCESS",
                "status": exec_result.status,
                "output": exec_result.output,
                "latency_ms": exec_result.latency_ms,
                "estimated_tokens": exec_result.estimated_tokens,
                "error": exec_result.error_message,
                "gate": "Gate 2 (Output Sanitized)"
            }
        finally:
            if close_session:
                db.close()

    def _record_tool_log(
        self,
        db: Session,
        request_id: str,
        tool_name: str,
        tool_args: dict,
        auth_result: ToolAuthorizationResult,
        exec_result: Optional[ToolExecutionResult] = None
    ):
        try:
            status = exec_result.status if exec_result else auth_result.status
            latency = exec_result.latency_ms if exec_result else 0.0
            tokens = exec_result.estimated_tokens if exec_result else 0
            snippet = str(exec_result.output)[:1000] if exec_result and exec_result.output else None

            log_entry = ToolExecutionLog(
                request_id=request_id,
                tool_name=tool_name,
                tool_args=json.dumps(tool_args),
                authorized=auth_result.status,
                block_reason=auth_result.reason if not auth_result.authorized else None,
                execution_status=status,
                latency_ms=latency,
                estimated_tokens=tokens,
                output_snippet=snippet,
                timestamp=utcnow()
            )
            db.add(log_entry)
            db.commit()
        except Exception as e:
            logger.error(f"Tool log record failed: {e}")
            db.rollback()

    def process_chat(
        self,
        user_id: str,
        messages: List[Dict[str, Any]],
        model: str = "openrouter/openai/gpt-4o-mini",
        api_key: Optional[str] = None,
        firewall_key: Optional[str] = None,
        session_id: Optional[str] = None,
        db: Optional[Session] = None,
        tenant_id: Optional[str] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Executes request through complete firewall pipeline:
        1. Ingress Prompt Inspection
        2. LLM Call via LiteLLM
        3. Loop Detector / Kill Switch Inspection
        4. Telemetry, Token & Cost Logging
        """
        start_time = time.time()
        close_session = False
        if db is None:
            db = SessionLocal()
            close_session = True

        req_id = str(uuid.uuid4())
        session_id = session_id or "default_session"
        # Per-tenant scope: the same session_id from two API keys never shares state.
        scope = f"{tenant_id or user_id or 'anonymous'}:{session_id}"

        # Extract last user message
        last_user_prompt = ""
        for m in reversed(messages):
            if m.get("role") == "user":
                content = m.get("content", "")
                if isinstance(content, list):
                    last_user_prompt = " ".join([item.get("text", "") for item in content if isinstance(item, dict)])
                else:
                    last_user_prompt = str(content)
                break

        # Check API key config policies if firewall key provided
        prompt_injection_active = True
        kill_switch_active = True
        tool_guard_active = True
        sensitive_data_active = True
        gdpr_mode = "tokenize"

        if firewall_key:
            key_cfg = db.query(ApiKeyConfig).filter(ApiKeyConfig.api_key == hash_api_key(firewall_key)).first()
            if key_cfg:
                prompt_injection_active = bool(key_cfg.prompt_injection_enabled)
                kill_switch_active = bool(key_cfg.kill_switch_enabled)
                tool_guard_active = bool(key_cfg.tool_guard_enabled)
                sensitive_data_active = bool(getattr(key_cfg, "sensitive_data_enabled", 1))
                gdpr_mode = getattr(key_cfg, "gdpr_mode", "tokenize") or "tokenize"
                model = key_cfg.model or model
                key_cfg.total_calls = (key_cfg.total_calls or 0) + 1
                db.commit()

        # Step 0: Ingress Sensitive Data & GDPR Protection (Secrets, PII, Article 9 Special Categories)
        sensitive_findings = []
        sanitized_prompt = last_user_prompt
        effective_messages = messages
        last_user_idx = max((i for i, m in enumerate(messages) if m.get("role") == "user"), default=None)
        if sensitive_data_active and last_user_prompt:
            sanitized_prompt, sensitive_findings, should_block, sanitized_content = self._sanitize_content(
                messages[last_user_idx].get("content"), scope, gdpr_mode
            )

            if should_block:
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000.0
                blocked_types = ", ".join(set(f.entity_type for f in sensitive_findings))
                blocked_response = f"🔒 [Agent Firewall Sensitive Data Guard]: Request blocked due to unauthorized sensitive payload / GDPR violation. Detected: {blocked_types}"

                self._log_transaction(
                    db=db,
                    req_id=req_id,
                    user_id=user_id,
                    model=model,
                    user_query=sanitized_prompt,
                    bot_response=blocked_response,
                    status="blocked_sensitive_data",
                    latency_ms=latency_ms,
                    input_tokens=len(last_user_prompt.split()),
                    output_tokens=10,
                    cost=0.0
                )

                return {
                    "request_id": req_id,
                    "status": "BLOCKED",
                    "reason": f"Sensitive Data / GDPR Violation ({blocked_types})",
                    "pipeline_stage": "Ingress Sensitive Data & GDPR Shield",
                    "sensitive_findings": [f.to_dict() for f in sensitive_findings],
                    "response": {
                        "id": req_id,
                        "choices": [{
                            "message": {"role": "assistant", "content": blocked_response},
                            "finish_reason": "security_block"
                        }],
                        "usage": {"prompt_tokens": len(last_user_prompt.split()), "completion_tokens": 10, "total_tokens": len(last_user_prompt.split()) + 10}
                    }
                }

            # Update messages payload with pseudonymized text if modified
            if sensitive_findings:
                effective_messages = list(messages)
                effective_messages[last_user_idx] = {**messages[last_user_idx], "content": sanitized_content}

        # Step 1: Prompt Injection Check
        ingress_result = (self.inspect_prompt(last_user_prompt, check_sensitive=False)
                          if prompt_injection_active else IngressSecurityResult(is_safe=True))

        if not ingress_result.is_safe:
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000.0
            blocked_response = f"🚫 [Agent Firewall Blocked Request]: {ingress_result.threat_type}. Matched trigger: '{ingress_result.matched_pattern}'"

            self._log_transaction(
                db=db,
                req_id=req_id,
                user_id=user_id,
                model=model,
                user_query=last_user_prompt,
                bot_response=blocked_response,
                status="blocked_injection",
                latency_ms=latency_ms,
                input_tokens=len(last_user_prompt.split()),
                output_tokens=10,
                cost=0.0
            )

            return {
                "request_id": req_id,
                "status": "BLOCKED",
                "reason": ingress_result.threat_type,
                "matched_pattern": ingress_result.matched_pattern,
                "pipeline_stage": "Ingress Prompt Injection Filter",
                "response": {
                    "id": req_id,
                    "choices": [{
                        "message": {"role": "assistant", "content": blocked_response},
                        "finish_reason": "security_block"
                    }],
                    "usage": {"prompt_tokens": len(last_user_prompt.split()), "completion_tokens": 10, "total_tokens": len(last_user_prompt.split()) + 10}
                }
            }

        # Step 1b: Indirect injection. Tool results are untrusted content.
        guard_ctx = SessionContext(session_id=scope)
        for m in effective_messages:
            if m.get("role") == "user":
                text = _message_text(m.get("content"))
                guard_ctx.user_texts.append(text)
                guard_ctx.user_task = guard_ctx.user_task or text
        tool_message_findings: List[Dict[str, Any]] = []
        if prompt_injection_active:
            effective_messages, tool_message_findings = self._screen_tool_messages(effective_messages, guard_ctx)

        # Step 2: Route & Execute Completion
        target_model = model or "openrouter/openai/gpt-4o-mini"
        if not target_model.startswith(("openrouter/", "groq/", "openai/", "anthropic/")):
            target_model = f"openrouter/{target_model}"

        if target_model.startswith("groq/"):
            provider_env_key = os.getenv("GROQ_API_KEY")
        elif target_model.startswith("openrouter/"):
            provider_env_key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY")
        else:
            provider_env_key = os.getenv("OPENROUTER_KEY") or os.getenv("OPENROUTER_API_KEY") or os.getenv("GROQ_API_KEY")

        call_api_key = api_key or provider_env_key

        try:
            try:
                completion_res = completion(
                    model=target_model,
                    messages=effective_messages,
                    api_key=call_api_key,
                    **kwargs
                )
            except Exception as e:
                return self._upstream_error(db, req_id, user_id, target_model, last_user_prompt,
                                            sensitive_findings, sanitized_prompt if sensitive_findings else None,
                                            e, start_time)

            latency_ms = (time.time() - start_time) * 1000.0
            choice = completion_res.choices[0] if getattr(completion_res, "choices", None) else None
            message = choice.message if choice is not None else None
            bot_text = (getattr(message, "content", None) or "") if message is not None else ""
            raw_calls = list(getattr(message, "tool_calls", None) or []) if message is not None else []
            finish_reason = getattr(choice, "finish_reason", None) or "stop"

            # Step 2b: Gate the model's proposed tool calls
            if tool_guard_active and raw_calls:
                kept_calls, blocked_calls = self._screen_tool_calls(raw_calls, guard_ctx)
            else:
                kept_calls, blocked_calls = [_tool_call_to_dict(tc) for tc in raw_calls], []
            if blocked_calls:
                logger.warning(f"🚫 [Agent Firewall Tool Gate]: blocked {blocked_calls}")
                if not kept_calls:
                    finish_reason = "stop"
                    note = "; ".join(f"{b['tool']} ({b['reason']})" for b in blocked_calls)
                    bot_text = (bot_text + "\n\n" if bot_text else "") + f"🛡️ [Agent Firewall]: Tool call(s) blocked: {note}"

            # Step 3: Loop Detector / Kill Switch (text + tool calls, per tenant session)
            status = "success"
            if kill_switch_active:
                fingerprint = bot_text + "|" + json.dumps(
                    [(c["function"]["name"], c["function"]["arguments"]) for c in kept_calls], sort_keys=True)
                loop_check = self.check_kill_switch(scope, fingerprint, fuzzy=not kept_calls)
                if loop_check.get("loop_detected"):
                    bot_text = f"🛑 [Agent Firewall Kill-Switch]: Terminated response loop. Layer {loop_check['layer']} triggered: {loop_check['reason']}"
                    kept_calls, finish_reason = [], "stop"
                    status = "blocked_loop"

            # Step 3b: Egress. Never echo credentials. When untrusted tool content was in
            # context, also strip exfiltration channels (external images, data-carrying links).
            bot_text, leaked = self.sensitive_engine.redact_secrets(bot_text)
            egress_findings: List[str] = [f"credentials_redacted:{leaked}"] if leaked else []
            if guard_ctx.tainted and bot_text:
                egress = self.injection_guard.check_output(bot_text, guard_ctx)
                bot_text = egress.text
                egress_findings += egress.findings

            # Step 4: Token and Cost Calculations
            input_tokens = 0
            output_tokens = 0
            if hasattr(completion_res, "usage") and completion_res.usage:
                input_tokens = getattr(completion_res.usage, "prompt_tokens", 0) or 0
                output_tokens = getattr(completion_res.usage, "completion_tokens", 0) or 0

            cost = 0.0
            try:
                clean_pricing_model = target_model
                for pfx in ["openrouter/openai/", "openrouter/anthropic/", "openrouter/google/", "openrouter/meta-llama/", "openrouter/"]:
                    if clean_pricing_model.startswith(pfx):
                        clean_pricing_model = clean_pricing_model[len(pfx):]
                        break
                cost = cost_calculator.completion_cost(completion_response=completion_res, model=clean_pricing_model) or 0.0
            except Exception:
                # Fallback for OpenRouter models not in LiteLLM's local price table
                cost = (input_tokens * 0.00000015) + (output_tokens * 0.0000006)

            final_status = "UNSAFE" if sensitive_findings else ("SUCCESS" if status == "success" else "LOOP_TERMINATED")
            final_verdict = "UNSAFE" if sensitive_findings else ("SAFE" if status == "success" else "SUSPICIOUS")
            log_status = "unsafe_sensitive_data" if sensitive_findings else status

            if sensitive_findings:
                types = ", ".join(sorted(set(f.entity_type for f in sensitive_findings)))
                logger.warning(
                    f"🚨 [Agent Firewall]: Sensitive information detected & redacted ({types})! "
                    f"Verdict: UNSAFE | Redacted Items: {[f.to_dict() for f in sensitive_findings]}"
                )

            logged_calls = "".join(f"\n[tool_call] {c['function']['name']}({c['function']['arguments']})" for c in kept_calls)
            self._log_transaction(
                db=db,
                req_id=req_id,
                user_id=user_id,
                model=target_model,
                user_query=sanitized_prompt if (sensitive_findings and sanitized_prompt) else last_user_prompt,
                bot_response=bot_text + logged_calls,
                status=log_status,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost=cost
            )

            out_message: Dict[str, Any] = {"role": "assistant", "content": bot_text if (bot_text or not kept_calls) else None}
            if kept_calls:
                out_message["tool_calls"] = kept_calls
            response_body: Dict[str, Any] = {
                "id": getattr(completion_res, "id", None) or req_id,
                "object": "chat.completion",
                "created": getattr(completion_res, "created", None) or int(time.time()),
                "model": target_model,
                "choices": [{"index": 0, "message": out_message, "finish_reason": finish_reason}],
                "usage": {
                    "prompt_tokens": input_tokens,
                    "completion_tokens": output_tokens,
                    "total_tokens": input_tokens + output_tokens
                }
            }
            if blocked_calls or tool_message_findings or egress_findings:
                response_body["firewall"] = {
                    "blocked_tool_calls": blocked_calls,
                    "tool_message_findings": tool_message_findings,
                    "egress_findings": egress_findings,
                }

            return {
                "request_id": req_id,
                "status": final_status,
                "verdict": final_verdict,
                "pipeline_stage": "Completed",
                "model_used": target_model,
                "latency_ms": round(latency_ms, 2),
                "redacted_prompt": sanitized_prompt if sensitive_findings else None,
                "sensitive_findings": [f.to_dict() for f in sensitive_findings],
                "ropa_record": self.sensitive_engine.generate_ropa_record(sensitive_findings, session_id=session_id) if sensitive_findings else None,
                "blocked_tool_calls": blocked_calls,
                "tool_message_findings": tool_message_findings,
                "tokens": {
                    "input": input_tokens,
                    "output": output_tokens,
                    "total": input_tokens + output_tokens
                },
                "cost_usd": round(cost, 6),
                "response": response_body,
            }

        finally:
            if close_session:
                db.close()

    def _upstream_error(self, db, req_id, user_id, model, user_prompt, sensitive_findings, redacted_prompt, error, start_time):
        """Report upstream failures as failures. Never as a simulated success."""
        latency_ms = (time.time() - start_time) * 1000.0
        first_line = (str(error).splitlines() or [error.__class__.__name__])[0][:300]
        safe_message, _ = self.sensitive_engine.redact_secrets(first_line)  # provider errors can echo key fragments
        is_auth = any(t in first_line.lower() for t in UPSTREAM_AUTH_ERRORS)
        error_type = "upstream_auth_error" if is_auth else "upstream_error"
        logger.warning(f"Upstream provider call failed ({error_type}): {safe_message}")
        self._log_transaction(
            db=db,
            req_id=req_id,
            user_id=user_id,
            model=model,
            user_query=redacted_prompt if sensitive_findings else user_prompt,
            bot_response=f"Error ({error_type}): {safe_message}",
            status=f"failed_{error_type}",
            latency_ms=latency_ms,
            input_tokens=0,
            output_tokens=0,
            cost=0.0
        )
        return {
            "request_id": req_id,
            "status": "ERROR",
            "error_type": error_type,
            "error": safe_message,
            "pipeline_stage": "Model Execution",
            "latency_ms": round(latency_ms, 2)
        }

    def _log_transaction(
        self,
        db: Session,
        req_id: str,
        user_id: str,
        model: str,
        user_query: str,
        bot_response: str,
        status: str,
        latency_ms: float,
        input_tokens: int,
        output_tokens: int,
        cost: float
    ):
        try:
            # 1. RequestLog
            req_log = RequestLog(
                request_id=req_id,
                user_id=user_id,
                model=model,
                timestamp=utcnow(),
                latency_ms=latency_ms,
                status=status
            )
            db.add(req_log)

            # 2. ContentLog
            content_log = ContentLog(
                request_id=req_id,
                user_query=user_query,
                bot_response=bot_response
            )
            db.add(content_log)

            # 3. MetricsLog
            total_tokens = input_tokens + output_tokens
            metrics_log = MetricsLog(
                request_id=req_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                input_cost=cost * 0.4,
                output_cost=cost * 0.6,
                total_cost=cost,
                cost_per_token=cost / max(total_tokens, 1),
                input_output_ratio=float(input_tokens) / float(max(output_tokens, 1)),
                model_used=model
            )
            db.add(metrics_log)

            # 4. DailyStats
            today = date.today()
            stats = db.query(DailyStats).filter(DailyStats.date == today).first()
            if not stats:
                stats = DailyStats(
                    date=today,
                    total_requests=0,
                    total_tokens=0,
                    total_cost=0.0,
                    success_count=0,
                    blocked_count=0,
                    failed_count=0
                )
                db.add(stats)

            stats.total_requests += 1
            stats.total_tokens += total_tokens
            stats.total_cost += cost
            if status == "success":
                stats.success_count += 1
            elif "blocked" in status:
                stats.blocked_count += 1
            else:
                stats.failed_count += 1

            db.commit()
        except Exception as e:
            logger.error(f"Transaction log failed: {e}")
            db.rollback()


# Singleton Instance
firewall_engine = FirewallCoreEngine()
