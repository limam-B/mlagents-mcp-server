from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def list_configs(config_dir: Path) -> list[str]:
    """List all YAML config files under the config directory."""
    if not config_dir.is_dir():
        return []
    configs = []
    for f in sorted(config_dir.rglob("*.yaml")):
        configs.append(str(f.relative_to(config_dir)))
    for f in sorted(config_dir.rglob("*.yml")):
        configs.append(str(f.relative_to(config_dir)))
    return configs


def read_config(config_path: Path) -> dict[str, Any]:
    """Read and parse a YAML config file."""
    with open(config_path) as f:
        return yaml.safe_load(f) or {}


def write_config(config_path: Path, data: dict[str, Any]) -> None:
    """Write data to a YAML config file."""
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def update_config(config_path: Path, updates: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge updates into an existing YAML config. Returns the merged result."""
    existing = read_config(config_path)
    merged = _deep_merge(existing, updates)
    write_config(config_path, merged)
    return merged


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge override into base. Only overridden keys change."""
    result = dict(base)
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result
