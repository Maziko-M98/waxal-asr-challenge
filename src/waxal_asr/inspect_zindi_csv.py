"""Inspect downloaded Zindi CSV files before training."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .columns import first_existing, id_column, language_column, text_column
from .config import AUDIO_COLUMN_CANDIDATES
from .zindi_csv import read_zindi_csv


def summarize_csv(path: Path) -> dict:
    table = read_zindi_csv(path)
    summary = {
        "path": str(path),
        "rows": int(len(table)),
        "columns": list(table.columns),
        "id_column": None,
        "text_column": None,
        "language_column": None,
        "audio_column": first_existing(table.columns, AUDIO_COLUMN_CANDIDATES),
        "nulls": {column: int(table[column].isna().sum()) for column in table.columns},
    }
    try:
        summary["id_column"] = id_column(table.columns)
        summary["unique_ids"] = int(table[summary["id_column"]].nunique())
    except ValueError:
        pass
    try:
        summary["text_column"] = text_column(table.columns)
    except ValueError:
        pass
    lang_col = language_column(table.columns)
    if lang_col:
        summary["language_column"] = lang_col
        summary["language_counts"] = {
            str(key): int(value)
            for key, value in table[lang_col].value_counts(dropna=False).items()
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/zindi"))
    parser.add_argument("--files", nargs="+", default=["Train.csv", "Test.csv", "SampleSubmission.csv"])
    args = parser.parse_args()

    summaries = []
    for name in args.files:
        path = args.data_dir / name
        if path.exists():
            summaries.append(summarize_csv(path))
        else:
            summaries.append({"path": str(path), "missing": True})
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
