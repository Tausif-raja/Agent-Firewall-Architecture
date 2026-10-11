"""L1: rule-based signature matching, driven by rules.yaml."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import yaml

from prompt_injection.segment import Trust

DEFAULT_RULES_PATH = Path(__file__).with_name("rules.yaml")


@dataclass(frozen=True)
class Rule:
    id: str
    regex: re.Pattern
    weight: float
    scope: str                       # "all" | "untrusted"
    block_below: Optional[Trust]


@dataclass
class RuleResult:
    score: float = 0.0
    hits: list[str] = field(default_factory=list)
    high_confidence: bool = False    # a hit that should short-circuit to block


class RuleEngine:
    def __init__(self, rules: list[Rule], version: int | str = "?"):
        self.rules = rules
        self.version = version

    @classmethod
    def from_yaml(cls, path: Path | str = DEFAULT_RULES_PATH) -> "RuleEngine":
        data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
        rules = []
        for r in data["rules"]:
            scope = r.get("scope", "all")
            if scope not in ("all", "untrusted"):
                raise ValueError(f"rule {r['id']}: invalid scope {scope!r}")
            block_below = r.get("block_below")
            rules.append(Rule(
                id=r["id"],
                regex=re.compile(r["pattern"], re.IGNORECASE),
                weight=float(r["weight"]),
                scope=scope,
                block_below=Trust[block_below] if block_below else None,
            ))
        return cls(rules, version=data.get("version", "?"))

    def match(self, texts: Iterable[str], trust: Trust) -> RuleResult:
        """Match every rule against every text variant; each rule counts once."""
        texts = list(texts)
        result = RuleResult()
        benign = 1.0
        for rule in self.rules:
            if rule.scope == "untrusted" and trust > Trust.SEMI:
                continue
            if not any(rule.regex.search(t) for t in texts):
                continue
            result.hits.append(rule.id)
            benign *= 1.0 - rule.weight
            if rule.block_below is not None and trust < rule.block_below:
                result.high_confidence = True
        result.score = 1.0 - benign
        return result
