"""Flatten tool/API JSON so every string leaf is scanned with its path."""

from __future__ import annotations

import json
from typing import Any, Iterator

from prompt_injection.extractors.base import Extracted

MIN_LEAF_CHARS = 12  # ids, enums and timestamps aren't worth scanning


def iter_string_leaves(obj: Any, path: str = "$") -> Iterator[tuple[str, str]]:
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield from iter_string_leaves(v, f"{path}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from iter_string_leaves(v, f"{path}[{i}]")
    elif isinstance(obj, str):
        yield path, obj


def extract_json(raw: str | bytes | dict | list) -> Extracted:
    obj = json.loads(raw) if isinstance(raw, (str, bytes)) else raw
    leaves = [(p, s) for p, s in iter_string_leaves(obj) if len(s.strip()) >= MIN_LEAF_CHARS]
    visible = "\n".join(f"{p}: {s}" for p, s in leaves)
    return Extracted(visible=visible, meta={"json_paths": [p for p, _ in leaves]})
