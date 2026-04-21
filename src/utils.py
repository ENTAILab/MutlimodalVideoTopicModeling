from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def save_json(path: str | Path, payload: Any) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def load_json(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_numpy(path: str | Path, arr: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, arr)


def l2_normalize(arr: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=-1, keepdims=True)
    return arr / np.maximum(norms, eps)
