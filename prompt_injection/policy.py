"""Score fusion, thresholds and the trust-dependent action table."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from prompt_injection.segment import Action, Label, Trust


@dataclass(frozen=True)
class Thresholds:
    low: float    # below: benign
    high: float   # at/above: injection. [low, high) is the gray zone sent to the judge.


def _default_thresholds() -> dict[Trust, Thresholds]:
    # Starting points only: recalibrate per channel on your own eval set
    # (target recall at a fixed FPR, see docs/PROMPT_INJECTION.md section 7).
    # Users legitimately *discuss* injection, so USER gets the most headroom.
    return {
        Trust.USER: Thresholds(low=0.50, high=0.90),
        Trust.SEMI: Thresholds(low=0.40, high=0.85),
        Trust.UNTRUSTED: Thresholds(low=0.30, high=0.80),
    }


@dataclass
class PolicyConfig:
    thresholds: dict[Trust, Thresholds] = field(default_factory=_default_thresholds)
    rule_weight: float = 0.6          # how much L1 alone can move the fused score
    signal_weights: dict[str, float] = field(default_factory=lambda: {
        "unicode_tag_smuggling": 0.5,
        "invisible_chars": 0.1,
        "homoglyphs": 0.15,
        "base64_payload": 0.1,
        "hex_payload": 0.1,
        "url_encoded": 0.05,
        "hidden_instructions": 0.85,  # set by ingest(): hidden text that looks like instructions
    })
    use_judge: bool = True
    # What to do when L2 can't run. "closed" drops the segment, "rules_only" continues on L1.
    classifier_fail_mode: dict[Trust, str] = field(default_factory=lambda: {
        Trust.USER: "rules_only",
        Trust.SEMI: "rules_only",
        Trust.UNTRUSTED: "closed",
    })
    # Trust levels that taint the session when they enter the agent's context.
    taint_at_or_below: Trust = Trust.UNTRUSTED
    # Cumulative suspicion (sum of gray-zone/injection scores) that flags a session.
    session_suspicion_limit: float = 2.0


def noisy_or(*ps: float) -> float:
    benign = 1.0
    for p in ps:
        benign *= 1.0 - min(max(p, 0.0), 1.0)
    return 1.0 - benign


def fuse(
    classifier_score: Optional[float],
    rule_score: float,
    signals: dict[str, bool],
    cfg: PolicyConfig,
) -> float:
    sig = [cfg.signal_weights.get(name, 0.0) for name, on in signals.items() if on]
    if classifier_score is None:
        # Rules-only mode: L1 is the only detector left, so it carries full weight.
        return noisy_or(rule_score, *sig)
    return noisy_or(classifier_score, cfg.rule_weight * rule_score, *sig)


def label_for(score: float, trust: Trust, cfg: PolicyConfig) -> Label:
    t = cfg.thresholds[trust]
    if score >= t.high:
        return Label.INJECTION
    if score >= t.low:
        return Label.SUSPICIOUS
    return Label.BENIGN


def in_gray_zone(score: float, trust: Trust, cfg: PolicyConfig) -> bool:
    return label_for(score, trust, cfg) is Label.SUSPICIOUS


# Action table: rows = trust, columns = label.
_ACTIONS: dict[Trust, dict[Label, Action]] = {
    Trust.USER: {Label.BENIGN: Action.ALLOW, Label.SUSPICIOUS: Action.ALLOW, Label.INJECTION: Action.BLOCK},
    Trust.SEMI: {Label.BENIGN: Action.ALLOW, Label.SUSPICIOUS: Action.SANITIZE, Label.INJECTION: Action.QUARANTINE},
    Trust.UNTRUSTED: {Label.BENIGN: Action.ALLOW, Label.SUSPICIOUS: Action.SANITIZE, Label.INJECTION: Action.BLOCK},
}


def action_for(label: Label, trust: Trust) -> Action:
    return _ACTIONS[trust][label]
