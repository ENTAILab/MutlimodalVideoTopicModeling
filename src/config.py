from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass
class RuntimeConfig:
    raw: dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    def get(self, *keys: str, default: Any = None) -> Any:
        current: Any = self.raw
        for key in keys:
            if not isinstance(current, dict) or key not in current:
                return default
            current = current[key]
        return current


def load_config(path: str | Path) -> RuntimeConfig:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError("Config root must be a mapping")
    return RuntimeConfig(raw=data)


def ensure_dir(path: str | Path) -> Path:
    target = Path(path)
    target.mkdir(parents=True, exist_ok=True)
    return target
