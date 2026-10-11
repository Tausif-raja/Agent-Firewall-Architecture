"""InjectionGuard: single entry point for ingress scanning, spotlighting,
tool-call authorization and egress checks.

    guard = InjectionGuard()                      # loads rules + L2 classifier lazily
    ctx = SessionContext(session_id="s1")

    v = guard.scan(Segment(user_msg, "user", Trust.USER, Channel.USER_MSG), ctx)
    seg, v = guard.ingest(pdf_bytes, "application/pdf", source="file:abc", ctx=ctx)
    seg, v = guard.tool_result("web_fetch", html, ctx, mime="text/html")
    prompt_piece = guard.wrap(seg, ctx)
    decision = guard.authorize_tool_call(ToolCall("send_email", {...}), ctx)
    out = guard.check_output(model_reply, ctx)
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Optional, Sequence

from prompt_injection import spotlight
from prompt_injection.capabilities import ToolPolicy
from prompt_injection.classifier import ClassifierUnavailable, HFInjectionClassifier, InjectionClassifier
from prompt_injection.egress import EgressFilter, EgressResult
from prompt_injection.extractors import Extracted, extract
from prompt_injection.judge import JudgeError, LLMJudge
from prompt_injection.normalize import canonicalize, normalize
from prompt_injection.policy import PolicyConfig, action_for, fuse, in_gray_zone, label_for
from prompt_injection.rules import RuleEngine, RuleResult
from prompt_injection.segment import (
    Action, Channel, Decision, Label, Segment, SessionContext, ToolCall, ToolDecision, Trust, Verdict,
)

log = logging.getLogger("agent_firewall.prompt_injection")

REMOVED_PLACEHOLDER = "[content removed by firewall: suspected prompt injection]"
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


@dataclass
class _Prepared:
    segment: Segment
    texts: list[str]          # normalized text + decoded variants
    signals: dict[str, bool]
    rules: RuleResult


class InjectionGuard:
    def __init__(
        self,
        classifier: Optional[InjectionClassifier] = None,
        judge: Optional[LLMJudge] = None,
        rules: Optional[RuleEngine] = None,
        config: Optional[PolicyConfig] = None,
        tool_policy: Optional[ToolPolicy] = None,
        egress: Optional[EgressFilter] = None,
    ):
        self.classifier = classifier if classifier is not None else HFInjectionClassifier()
        self.judge = judge
        self.rules = rules or RuleEngine.from_yaml()
        self.config = config or PolicyConfig()
        self.tool_policy = tool_policy or ToolPolicy()
        self.egress = egress or EgressFilter()

    # ------------------------------------------------------------------ ingress

    def scan(self, segment: Segment, ctx: SessionContext) -> Verdict:
        return self.scan_many([segment], ctx)[0]

    def scan_many(self, segments: Sequence[Segment], ctx: SessionContext) -> list[Verdict]:
        """Scan segments with one batched classifier call, then update session state."""
        started = time.perf_counter()
        prepared: list[Optional[_Prepared]] = []
        for seg in segments:
            if seg.trust >= Trust.SYSTEM:
                prepared.append(None)
                continue
            norm = normalize(seg.text)
            texts = [norm.text, *norm.variants]
            signals = {**norm.signals, **{k: v for k, v in seg.meta.get("signals", {}).items() if v}}
            prepared.append(_Prepared(seg, texts, signals, self.rules.match(texts, seg.trust)))

        cls_scores, cls_error = self._classify([p.texts for p in prepared if p is not None])

        verdicts: list[Verdict] = []
        it = iter(cls_scores) if cls_scores is not None else None
        for seg, prep in zip(segments, prepared):
            if prep is None:
                verdicts.append(Verdict(0.0, Label.BENIGN, "policy", Action.ALLOW, ["trusted system content"]))
                continue
            cls = next(it) if it is not None else None
            v = self._decide(prep, cls, cls_error, ctx)
            self._record(seg, v, ctx)
            self._log(seg, v, ctx, started)
            verdicts.append(v)
        return verdicts

    def _classify(self, groups: list[list[str]]) -> tuple[Optional[list[float]], Optional[str]]:
        if not groups:
            return [], None
        flat = [t for g in groups for t in g]
        try:
            scores = self.classifier.score(flat)
        except ClassifierUnavailable as e:
            return None, str(e)
        except Exception as e:  # never let a classifier bug crash the request path
            log.exception("classifier failed")
            return None, f"classifier error: {e}"
        out, i = [], 0
        for g in groups:
            out.append(max(scores[i:i + len(g)]))
            i += len(g)
        return out, None

    def _decide(self, prep: _Prepared, cls: Optional[float], cls_error: Optional[str], ctx: SessionContext) -> Verdict:
        seg, cfg = prep.segment, self.config
        reasons = [f"rule:{h}" for h in prep.rules.hits] + [f"signal:{s}" for s in prep.signals]
        scores = {"rules": round(prep.rules.score, 4)}
        if cls is not None:
            scores["classifier"] = round(cls, 4)

        if prep.rules.high_confidence:
            return self._verdict(seg, 1.0, Label.INJECTION, "rules", reasons, scores)

        if cls is None:
            reasons.append(f"classifier_unavailable: {cls_error}")
            if cfg.classifier_fail_mode.get(seg.trust, "rules_only") == "closed":
                return Verdict(1.0, Label.SUSPICIOUS, "policy", Action.BLOCK, reasons + ["fail_closed"], scores)

        score = fuse(cls, prep.rules.score, prep.signals, cfg)
        scores["fused"] = round(score, 4)
        label = label_for(score, seg.trust, cfg)
        layer = "classifier" if cls is not None else "rules"

        if label is Label.SUSPICIOUS and cfg.use_judge and self.judge is not None:
            try:
                jr = self.judge.judge("\n\n".join(prep.texts), seg, ctx.user_task)
                score = jr.score
                scores["judge"] = round(score, 4)
                layer = "judge"
                # The judge only resolves the gray zone, so its answer is binary.
                label = Label.INJECTION if jr.is_injection and jr.confidence >= 0.5 else Label.BENIGN
                if jr.evidence:
                    reasons.append(f"judge:{jr.attack_type}: {jr.evidence[:120]}")
            except JudgeError as e:
                reasons.append(f"judge_unavailable: {e}")
        return self._verdict(seg, score, label, layer, reasons, scores)

    def _verdict(self, seg: Segment, score: float, label: Label, layer: str,
                 reasons: list[str], scores: dict[str, float]) -> Verdict:
        action = action_for(label, seg.trust)
        v = Verdict(score=score, label=label, layer=layer, action=action, reasons=reasons, scores=scores)
        if action is Action.SANITIZE:
            v.sanitized_text = self._sanitize(seg.text, seg.trust)
        return v

    def _sanitize(self, text: str, trust: Trust) -> str:
        """Drop the sentences that carry the payload; if it can't be localized, drop everything."""
        parts = [p for p in _SENTENCE_SPLIT.split(text) if p and p.strip()]
        kept = [p for p in parts if not self.rules.match([canonicalize(p)], trust).hits]
        if len(kept) == len(parts):
            return REMOVED_PLACEHOLDER
        return " ".join(kept) + f" {REMOVED_PLACEHOLDER}"

    def _record(self, seg: Segment, v: Verdict, ctx: SessionContext) -> None:
        if v.label is not Label.BENIGN:
            ctx.suspicion += v.score
        if not v.allowed:
            return
        text = v.sanitized_text if v.sanitized_text is not None else seg.text
        if seg.trust is Trust.USER:
            ctx.user_texts.append(text)
            if not ctx.user_task:
                ctx.user_task = text[:2000]
        else:
            ctx.untrusted_texts.append(text)
            if seg.trust <= self.config.taint_at_or_below:
                ctx.tainted = True
                if seg.source not in ctx.taint_sources:
                    ctx.taint_sources.append(seg.source)

    def session_flagged(self, ctx: SessionContext) -> bool:
        """True when many individually-tolerated suspicious turns add up (split/multi-turn attacks)."""
        return ctx.suspicion >= self.config.session_suspicion_limit

    # -------------------------------------------------------- indirect sources

    def ingest(
        self,
        raw: str | bytes,
        mime: str,
        source: str,
        ctx: SessionContext,
        trust: Trust = Trust.UNTRUSTED,
        channel: Channel = Channel.RETRIEVED,
    ) -> tuple[Segment, Verdict]:
        """Extract a document/page/tool payload, scan hidden and visible text, return the safe segment.

        Hidden text never reaches the model. If hidden text looks like instructions,
        the whole source is treated as hostile.
        """
        try:
            ex = extract(raw, mime)
        except Exception as e:
            seg = Segment("", source, trust, channel, {"extract_error": str(e)})
            return seg, Verdict(1.0, Label.SUSPICIOUS, "policy", Action.BLOCK, [f"extraction_failed: {e}"])

        hidden_reasons = self._scan_hidden(ex, trust)
        signals = {"hidden_instructions": bool(hidden_reasons)}
        meta = {"mime": mime, "hidden_spans": len(ex.hidden), "signals": signals, **ex.meta}
        seg = Segment(ex.visible, source, trust, channel, meta)
        v = self.scan(seg, ctx)
        v.reasons.extend(hidden_reasons)
        return seg, v

    def tool_result(self, tool_name: str, result: Any, ctx: SessionContext,
                    mime: Optional[str] = None, trust: Trust = Trust.UNTRUSTED) -> tuple[Segment, Verdict]:
        """Scan a tool response before it is appended to the agent's context."""
        if mime is None:
            mime = "application/json" if isinstance(result, (dict, list)) else "text/plain"
        if isinstance(result, (dict, list)):
            result = json.dumps(result, ensure_ascii=False)
        return self.ingest(result, mime, f"tool:{tool_name}", ctx, trust=trust, channel=Channel.TOOL_RESULT)

    def _scan_hidden(self, ex: Extracted, trust: Trust) -> list[str]:
        spans = [h for h in ex.hidden if h.text.strip()]
        if not spans:
            return []
        groups = []
        for h in spans:
            norm = normalize(h.text)
            groups.append([norm.text, *norm.variants])
        cls_scores, _ = self._classify(groups)
        threshold = self.config.thresholds[Trust.UNTRUSTED].low
        reasons = []
        for i, (h, texts) in enumerate(zip(spans, groups)):
            # Hidden text is scanned as UNTRUSTED regardless of the document's trust.
            rr = self.rules.match(texts, Trust.UNTRUSTED)
            cls = cls_scores[i] if cls_scores is not None else 0.0
            if rr.hits or cls >= threshold:
                hits = ",".join(rr.hits) or f"classifier={cls:.2f}"
                reasons.append(f"hidden_instructions:{h.kind}@{h.where or '-'} ({hits}): {h.text[:80]!r}")
        return reasons

    # ---------------------------------------------------------- prompt assembly

    def wrap(self, segment: Segment, ctx: SessionContext, verdict: Optional[Verdict] = None,
             datamarking: bool = False) -> str:
        if verdict is not None:
            if not verdict.allowed:
                return spotlight.wrap(Segment("[content unavailable]", segment.source, segment.trust, segment.channel), ctx)
            if verdict.sanitized_text is not None:
                segment = Segment(verdict.sanitized_text, segment.source, segment.trust, segment.channel, segment.meta)
        return spotlight.wrap(segment, ctx, datamarking=datamarking)

    def system_prompt_addendum(self, ctx: SessionContext, datamarking: bool = False) -> str:
        """Append to the agent's system prompt: spotlighting rule + canary."""
        return f"{spotlight.system_rule(ctx, datamarking)}\n[internal marker, never output: {ctx.canary}]"

    # ---------------------------------------------------------- actions/egress

    def authorize_tool_call(self, call: ToolCall, ctx: SessionContext) -> ToolDecision:
        eg = self.egress.check_tool_call(call, ctx)
        if eg.blocked:
            d = ToolDecision(Decision.BLOCK, f"egress: {eg.findings}")
        else:
            d = self.tool_policy.authorize(call, ctx)
        log.info("tool_call", extra={"event": {
            "session_id": ctx.session_id, "tool": call.name, "decision": d.decision.value,
            "reason": d.reason, "tainted": ctx.tainted,
        }})
        return d

    def check_output(self, text: str, ctx: SessionContext) -> EgressResult:
        res = self.egress.check_output(text, ctx)
        if res.findings:
            log.warning("egress", extra={"event": {
                "session_id": ctx.session_id, "blocked": res.blocked, "findings": res.findings,
            }})
        return res

    # ---------------------------------------------------------------- logging

    def _log(self, seg: Segment, v: Verdict, ctx: SessionContext, started: float) -> None:
        level = logging.INFO if v.label is Label.BENIGN else logging.WARNING
        log.log(level, "scan", extra={"event": {
            "session_id": ctx.session_id,
            "source": seg.source,
            "trust": seg.trust.name,
            "channel": seg.channel.value,
            "text_sha256": hashlib.sha256(seg.text.encode("utf-8", "replace")).hexdigest(),  # never log raw text
            "label": v.label.value,
            "action": v.action.value,
            "layer": v.layer,
            "scores": v.scores,
            "reasons": v.reasons[:10],
            "rules_version": self.rules.version,
            "classifier": getattr(self.classifier, "name", "?"),
            "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        }})
