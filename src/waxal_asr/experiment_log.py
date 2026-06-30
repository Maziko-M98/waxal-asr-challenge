"""Experiment logging utilities for reproducible competition work."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_COLUMNS = [
    "experiment_id",
    "created_at_utc",
    "stage",
    "hypothesis",
    "status",
    "dataset_version",
    "model",
    "config_path",
    "output_path",
    "local_wer",
    "local_cer",
    "local_error",
    "local_score",
    "public_score",
    "submitted",
    "notes",
]


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def safe_slug(value: str) -> str:
    slug = "".join(ch.lower() if ch.isalnum() else "_" for ch in value.strip())
    slug = "_".join(part for part in slug.split("_") if part)
    return slug or "experiment"


def make_experiment_id(name: str, *, timestamp: str | None = None) -> str:
    stamp = timestamp or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return f"{stamp}_{safe_slug(name)}"


def sha256_file(path: str | Path, *, prefix: int | None = None) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    value = digest.hexdigest()
    return value[:prefix] if prefix else value


def write_json(path: str | Path, payload: dict[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def append_experiment(log_path: str | Path, record: dict[str, Any]) -> None:
    target = Path(log_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    exists = target.exists()
    row = {column: record.get(column, "") for column in EXPERIMENT_COLUMNS}
    with target.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=EXPERIMENT_COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def dataset_fingerprint(paths: list[str | Path]) -> dict[str, dict[str, Any]]:
    fingerprint = {}
    for path in paths:
        current = Path(path)
        if not current.exists():
            fingerprint[str(current)] = {"exists": False}
            continue
        fingerprint[str(current)] = {
            "exists": True,
            "bytes": current.stat().st_size,
            "sha256": sha256_file(current, prefix=16),
        }
    return fingerprint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", type=Path, default=Path("experiments/experiment_log.csv"))
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--stage", required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--status", default="planned")
    parser.add_argument("--dataset-version", default="")
    parser.add_argument("--model", default="")
    parser.add_argument("--config-path", default="")
    parser.add_argument("--output-path", default="")
    parser.add_argument("--notes", default="")
    args = parser.parse_args()

    append_experiment(
        args.log,
        {
            "experiment_id": args.experiment_id,
            "created_at_utc": utc_now(),
            "stage": args.stage,
            "hypothesis": args.hypothesis,
            "status": args.status,
            "dataset_version": args.dataset_version,
            "model": args.model,
            "config_path": args.config_path,
            "output_path": args.output_path,
            "submitted": "false",
            "notes": args.notes,
        },
    )


if __name__ == "__main__":
    main()
