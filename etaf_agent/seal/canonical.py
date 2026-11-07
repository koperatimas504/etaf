from __future__ import annotations

import json
from typing import Any

import numpy as np


def _quantize(value: Any, decimals: int) -> Any:
    if isinstance(value, bool):
        return value
    if isinstance(value, (float, np.floating)):
        f32 = float(np.float32(value))
        return format(f32, f".{decimals}e")
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (list, tuple)):
        return [_quantize(v, decimals) for v in value]
    if isinstance(value, np.ndarray):
        return [_quantize(v, decimals) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _quantize(value[k], decimals) for k in sorted(value)}
    return value


def canonicalize(payload: dict[str, Any], decimals: int) -> bytes:
    normalized = _quantize(payload, decimals)
    text = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return text.encode("utf-8")
