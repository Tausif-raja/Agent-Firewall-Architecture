import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from prompt_injection.classifier import CallableClassifier  # noqa: E402
from prompt_injection.guard import InjectionGuard  # noqa: E402
from prompt_injection.segment import SessionContext  # noqa: E402


def fake_score(text: str) -> float:
    """Stand-in for the L2 model so tests are fast and deterministic."""
    t = text.lower()
    if "exfiltrate" in t or "ignore all previous instructions" in t:
        return 0.97
    if "pretend" in t:
        return 0.6
    return 0.02


@pytest.fixture
def guard():
    return InjectionGuard(classifier=CallableClassifier(fake_score, "fake"))


@pytest.fixture
def ctx():
    return SessionContext(session_id="test")
