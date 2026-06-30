"""Create validation metrics reports for prediction files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .columns import id_column, text_column
from .metrics import waxal_score
from .normalization import normalize_text
from .zindi_csv import read_zindi_csv


def word_bucket(word_count: int) -> str:
    if word_count <= 5:
        return "00_05"
    if word_count <= 10:
        return "06_10"
    if word_count <= 20:
        return "11_20"
    return "21_plus"


def score_frame(frame: pd.DataFrame) -> dict[str, float | int]:
    scores = waxal_score(frame["reference_text"], frame["prediction_text"])
    return {
        "rows": int(len(frame)),
        "wer": scores["wer"],
        "cer": scores["cer"],
        "error": scores["error"],
        "score": scores["score"],
    }


def metrics_by_group(frame: pd.DataFrame, group_column: str) -> pd.DataFrame:
    rows = []
    for group_value, group in frame.groupby(group_column, sort=True):
        row = {group_column: group_value}
        row.update(score_frame(group))
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame) -> str:
    if table.empty:
        return "_No rows._"
    columns = [str(column) for column in table.columns]
    rows = []
    for _, row in table.iterrows():
        rows.append([str(row[column]).replace("|", "\\|") for column in table.columns])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_validation_report(
    reference_csv: Path,
    prediction_csv: Path,
    out_dir: Path,
    *,
    split: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    reference = read_zindi_csv(reference_csv)
    prediction = read_zindi_csv(prediction_csv)

    if "original_split" in reference.columns:
        reference = reference[reference["original_split"].astype(str) == split].copy()
    if reference.empty:
        raise ValueError(f"No reference rows found for split={split!r}")

    ref_id = id_column(reference.columns)
    pred_id = id_column(prediction.columns)
    ref_text = text_column(reference.columns)
    pred_text = text_column(prediction.columns)

    merged = reference[[ref_id, ref_text, "language"]].rename(
        columns={ref_id: "id", ref_text: "reference_text"}
    ).merge(
        prediction[[pred_id, pred_text]].rename(columns={pred_id: "id", pred_text: "prediction_text"}),
        on="id",
        how="inner",
    )
    if len(merged) != len(reference):
        missing = len(reference) - len(merged)
        raise ValueError(f"Prediction file is missing {missing} validation IDs")

    merged["reference_word_len"] = merged["reference_text"].map(lambda text: len(normalize_text(text).split()))
    merged["length_bucket"] = merged["reference_word_len"].map(word_bucket)

    overall = pd.DataFrame([score_frame(merged)])
    by_language = metrics_by_group(merged, "language")
    by_length = metrics_by_group(merged, "length_bucket")

    overall.to_csv(out_dir / "overall_metrics.csv", index=False)
    by_language.to_csv(out_dir / "metrics_by_language.csv", index=False)
    by_length.to_csv(out_dir / "metrics_by_length_bucket.csv", index=False)

    report = f"""# Validation Report

Reference: `{reference_csv}`

Predictions: `{prediction_csv}`

Split: `{split}`

## Overall

{markdown_table(overall)}

## By Language

{markdown_table(by_language)}

## By Reference Length

{markdown_table(by_length)}

## Decision Use

Use this report to decide whether a model deserves a leaderboard submission.
Aggregate gains are not enough: regressions by language, especially Luganda,
must be understood before promoting an experiment.
"""
    report_path = out_dir / "validation_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", type=Path, default=Path("data/zindi/Train.csv"))
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--split", default="validation")
    args = parser.parse_args()
    report_path = build_validation_report(
        args.reference_csv,
        args.prediction_csv,
        args.out_dir,
        split=args.split,
    )
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
