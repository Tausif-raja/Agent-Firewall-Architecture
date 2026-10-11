"""L2: fine-tuned transformer classifier (primary detector).

Default model is ProtectAI's DeBERTa-v3 prompt-injection classifier (ungated,
Apache-2.0). Llama Prompt Guard 2 is a drop-in alternative once you have
accepted its licence on Hugging Face:

    HFInjectionClassifier("meta-llama/Llama-Prompt-Guard-2-86M")

Long inputs are split into overlapping windows and the MAX window score is
returned, so a payload can't be diluted inside a long benign document.
"""

from __future__ import annotations

import os
import threading
from typing import Callable, Protocol, Sequence

DEFAULT_MODEL = os.environ.get("INJECTION_CLASSIFIER_MODEL", "protectai/deberta-v3-base-prompt-injection-v2")

# Label names that mean "attack" across the supported models.
_POSITIVE_LABELS = {"INJECTION", "MALICIOUS", "JAILBREAK", "LABEL_1", "UNSAFE"}


class ClassifierUnavailable(RuntimeError):
    """Raised when the classifier can't run; the policy applies its fail mode."""


class InjectionClassifier(Protocol):
    name: str

    def score(self, texts: Sequence[str]) -> list[float]:
        """Return P(injection) in [0, 1] for each text."""
        ...


class HFInjectionClassifier:
    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        max_length: int = 512,
        stride: int = 256,
        batch_size: int = 16,
        device: str | None = None,
    ):
        self.name = model_id
        self.model_id = model_id
        self.max_length = max_length
        self.stride = stride
        self.batch_size = batch_size
        self.device = device
        self._lock = threading.Lock()
        self._tokenizer = None
        self._model = None
        self._positive_ids: list[int] = []

    def _load(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            try:
                import torch
                from transformers import AutoModelForSequenceClassification, AutoTokenizer
            except ImportError as e:
                raise ClassifierUnavailable("install 'transformers' and 'torch' to enable the L2 classifier") from e
            try:
                tokenizer = AutoTokenizer.from_pretrained(self.model_id, use_fast=True)
                model = AutoModelForSequenceClassification.from_pretrained(self.model_id)
            except Exception as e:  # network, gated model, missing weights
                raise ClassifierUnavailable(f"could not load {self.model_id}: {e}") from e
            if not tokenizer.is_fast:
                raise ClassifierUnavailable(f"{self.model_id}: a fast tokenizer is required for windowing")
            device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device).eval()
            positive = [i for i, lbl in model.config.id2label.items() if str(lbl).upper() in _POSITIVE_LABELS]
            if not positive:
                raise ClassifierUnavailable(f"{self.model_id}: no injection label in {model.config.id2label}")
            self._positive_ids = positive
            self._device = device
            self._tokenizer = tokenizer
            self._model = model

    def score(self, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        self._load()
        import torch

        scores = [0.0] * len(texts)
        for b in range(0, len(texts), self.batch_size):
            batch = list(texts[b:b + self.batch_size])
            # Overflowing windows: max_length tokens each, overlapping by `stride` tokens.
            enc = self._tokenizer(
                batch, truncation=True, max_length=self.max_length, stride=self.stride,
                return_overflowing_tokens=True, padding=True, return_tensors="pt",
            )
            owners = enc.pop("overflow_to_sample_mapping").tolist()
            enc = {k: v.to(self._device) for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}
            with torch.inference_mode():
                probs = torch.softmax(self._model(**enc).logits.float(), dim=-1)
            p_inj = probs[:, self._positive_ids].sum(dim=-1).tolist()
            for owner, p in zip(owners, p_inj):
                scores[b + owner] = max(scores[b + owner], float(p))
        return scores


class CallableClassifier:
    """Adapter for tests or custom backends: wraps fn(text) -> score."""

    def __init__(self, fn: Callable[[str], float], name: str = "callable"):
        self.fn = fn
        self.name = name

    def score(self, texts: Sequence[str]) -> list[float]:
        return [float(self.fn(t)) for t in texts]


class NullClassifier:
    """Explicitly disables L2. Every call raises, so the fail policy applies."""

    name = "none"

    def score(self, texts: Sequence[str]) -> list[float]:
        raise ClassifierUnavailable("L2 classifier disabled")
