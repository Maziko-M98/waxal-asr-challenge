"""Score transcript CSVs with the local Waxal metric."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .columns import id_column, text_column
from .metrics import waxal_score
from .zindi_csv import read_zindi_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-csv", type=Path, required=True)
    parser.add_argument("--prediction-csv", type=Path, required=True)
    parser.add_argument("--no-normalize", action="store_true")
    args = parser.parse_args()

    reference = read_zindi_csv(args.reference_csv)
    prediction = read_zindi_csv(args.prediction_csv)

    ref_id = id_column(reference.columns)
    pred_id = id_column(prediction.columns)
    ref_text = text_column(reference.columns)
    pred_text = text_column(prediction.columns)

    ref_table = reference[[ref_id, ref_text]].rename(
        columns={ref_id: "id", ref_text: "reference_text"}
    )
    pred_table = prediction[[pred_id, pred_text]].rename(
        columns={pred_id: "id", pred_text: "prediction_text"}
    )
    merged = ref_table.merge(pred_table, on="id", how="inner")
    if len(merged) != len(reference):
        missing = len(reference) - len(merged)
        raise ValueError(f"Prediction file is missing {missing} reference IDs")

    scores = waxal_score(
        merged["reference_text"],
        merged["prediction_text"],
        normalize=not args.no_normalize,
    )
    print(json.dumps(scores, indent=2))


if __name__ == "__main__":
    main()
