from __future__ import annotations

import csv
import json
import os
from pathlib import Path

_ENV_FIELDS = ("base_url", "api_key", "project", "scope")


def parse_config(path: str | Path, names: list[str] | None = None) -> list[dict]:
    """Read a JSONL config, resolve env-var names in credential fields, optionally select by name."""
    configs = [json.loads(l) for l in Path(path).read_text(encoding="utf-8").split("\n") if l.strip()]
    for c in configs:
        for k in _ENV_FIELDS:
            v = c.get("provider", {}).get(k)
            if isinstance(v, str) and v in os.environ:
                c["provider"][k] = os.environ[v]
    if not names:
        return configs
    by_name = {c["name"]: c for c in configs}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise KeyError(f"Not found in {path}: {missing}")
    return [by_name[n] for n in names]


def load_dataset(path: str | Path) -> dict[str, dict]:
    """Read dataset TSV, return {id: row}."""
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in csv.DictReader(f, delimiter="\t")}
