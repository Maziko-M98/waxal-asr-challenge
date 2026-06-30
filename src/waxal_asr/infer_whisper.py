"""Run Whisper inference and write a Zindi submission file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from .columns import id_column, language_column
from .config import TARGET_LANGUAGES
from .experiment_log import (
    append_experiment,
    dataset_fingerprint,
    make_experiment_id,
    utc_now,
    write_json,
)
from .normalization import submission_text
from .zindi_csv import read_zindi_csv


def batch_iter(rows: list[dict], batch_size: int):
    for start in range(0, len(rows), batch_size):
        yield rows[start:start + batch_size]


def load_submission_ids(path: Path | None) -> tuple[set[str] | None, dict[str, str]]:
    if path is None:
        return None, {}
    table = read_zindi_csv(path)
    id_col = id_column(table.columns)
    lang_col = language_column(table.columns)
    ids = set(table[id_col].astype(str))
    languages = {}
    if lang_col is not None:
        languages = dict(zip(table[id_col].astype(str), table[lang_col].astype(str), strict=True))
    return ids, languages


def build_prediction_rows(languages: list[str], split: str, test_csv: Path | None) -> list[dict]:
    from .hf_data import load_many_splits

    wanted_ids, id_to_language = load_submission_ids(test_csv)
    dataset = load_many_splits(languages, split, decode_audio=True, keep_text=False)
    rows = []
    for row in dataset:
        row_id = str(row["id"])
        if wanted_ids is not None and row_id not in wanted_ids:
            continue
        if id_to_language and row_id in id_to_language:
            row["language"] = id_to_language[row_id]
        rows.append(row)
    if wanted_ids is not None and len(rows) != len(wanted_ids):
        found = {str(row["id"]) for row in rows}
        missing = sorted(wanted_ids - found)[:10]
        raise ValueError(f"Found {len(rows)} of {len(wanted_ids)} test IDs. First missing IDs: {missing}")
    return rows


def write_submission(predictions: dict[str, str], sample_submission: Path | None, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if sample_submission is None:
        table = pd.DataFrame(
            [{"ID": row_id, "transcription": text} for row_id, text in predictions.items()]
        )
    else:
        table = read_zindi_csv(sample_submission)
        id_col = id_column(table.columns)
        target_cols = [col for col in table.columns if col != id_col]
        if not target_cols:
            raise ValueError("Sample submission does not contain a target column")
        target_col = target_cols[0]
        table[target_col] = table[id_col].astype(str).map(predictions).fillna("")
    table.to_csv(output, index=False)


def validate_local_submission_inputs(test_csv: Path | None, sample_submission: Path | None) -> dict:
    if test_csv is None or sample_submission is None:
        return {"validated": False, "reason": "test_csv or sample_submission not provided"}
    test = read_zindi_csv(test_csv)
    sample = read_zindi_csv(sample_submission)
    test_id = id_column(test.columns)
    sample_id = id_column(sample.columns)
    test_ids = set(test[test_id].astype(str))
    sample_ids = set(sample[sample_id].astype(str))
    prefixes = test[test_id].astype(str).map(lambda value: value.split("_", 1)[0]).value_counts()
    return {
        "validated": True,
        "test_rows": int(len(test)),
        "sample_rows": int(len(sample)),
        "test_unique_ids": int(test[test_id].nunique()),
        "sample_unique_ids": int(sample[sample_id].nunique()),
        "test_sample_symmetric_difference": len(test_ids ^ sample_ids),
        "test_id_prefix_counts": {str(key): int(value) for key, value in prefixes.items()},
    }


def build_parser(defaults: dict | None = None) -> argparse.ArgumentParser:
    defaults = defaults or {}
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--model", default=defaults.get("model", "openai/whisper-large-v3-turbo"))
    parser.add_argument("--adapter", type=Path, default=defaults.get("adapter"))
    parser.add_argument("--languages", nargs="+", default=defaults.get("languages", list(TARGET_LANGUAGES)))
    parser.add_argument("--split", default=defaults.get("split", "test"))
    parser.add_argument("--test-csv", type=Path, default=defaults.get("test_csv"))
    parser.add_argument("--sample-submission", type=Path, default=defaults.get("sample_submission"))
    parser.add_argument("--output", type=Path, default=defaults.get("output"))
    parser.add_argument("--batch-size", type=int, default=defaults.get("batch_size", 4))
    parser.add_argument("--num-beams", type=int, default=defaults.get("num_beams", 1))
    parser.add_argument("--max-new-tokens", type=int, default=defaults.get("max_new_tokens", 225))
    parser.add_argument("--device", default=defaults.get("device", "auto"))
    parser.add_argument("--max-samples", type=int, default=defaults.get("max_samples"))
    parser.add_argument("--raw-predictions", type=Path, default=defaults.get("raw_predictions"))
    parser.add_argument("--dry-run", action="store_true", default=defaults.get("dry_run", False))
    parser.add_argument("--log-dry-run", action="store_true", default=defaults.get("log_dry_run", False))
    parser.add_argument("--experiment-name", default=defaults.get("experiment_name", "whisper_inference"))
    parser.add_argument("--stage", default=defaults.get("stage", "baseline_submission"))
    parser.add_argument("--hypothesis", default=defaults.get("hypothesis", "Run Whisper inference to produce a baseline submission."))
    parser.add_argument("--notes", default=defaults.get("notes", ""))
    parser.add_argument("--dataset-version", default=defaults.get("dataset_version", "zindi_download_2026_06_30"))
    parser.add_argument("--experiment-log", type=Path, default=defaults.get("experiment_log", "experiments/experiment_log.csv"))
    parser.add_argument("--experiment-dir", type=Path, default=defaults.get("experiment_dir", "experiments/configs"))
    return parser


def parse_args() -> argparse.Namespace:
    config_parser = argparse.ArgumentParser(add_help=False)
    config_parser.add_argument("--config", type=Path, default=None)
    config_args, remaining = config_parser.parse_known_args()
    defaults = {}
    if config_args.config:
        defaults = json.loads(config_args.config.read_text(encoding="utf-8"))
    parser = build_parser(defaults)
    args = parser.parse_args(remaining)
    args.config = config_args.config
    for field in [
        "adapter",
        "test_csv",
        "sample_submission",
        "output",
        "raw_predictions",
        "experiment_log",
        "experiment_dir",
    ]:
        value = getattr(args, field)
        if value is not None and not isinstance(value, Path):
            setattr(args, field, Path(value))
    if args.output is None and not args.dry_run:
        parser.error("--output is required unless --dry-run is set")
    return args


def config_payload(args: argparse.Namespace, experiment_id: str, input_validation: dict) -> dict:
    watched_files = [
        args.test_csv,
        args.sample_submission,
        args.config,
    ]
    watched_files = [path for path in watched_files if path is not None]
    return {
        "experiment_id": experiment_id,
        "created_at_utc": utc_now(),
        "stage": args.stage,
        "hypothesis": args.hypothesis,
        "model": args.model,
        "adapter": str(args.adapter) if args.adapter else None,
        "languages": list(args.languages),
        "split": args.split,
        "test_csv": str(args.test_csv) if args.test_csv else None,
        "sample_submission": str(args.sample_submission) if args.sample_submission else None,
        "output": str(args.output) if args.output else None,
        "batch_size": args.batch_size,
        "num_beams": args.num_beams,
        "max_new_tokens": args.max_new_tokens,
        "device": args.device,
        "max_samples": args.max_samples,
        "raw_predictions": str(args.raw_predictions) if args.raw_predictions else None,
        "dataset_version": args.dataset_version,
        "input_validation": input_validation,
        "dataset_fingerprint": dataset_fingerprint(watched_files),
        "notes": args.notes,
    }


def main() -> None:
    args = parse_args()
    experiment_id = make_experiment_id(args.experiment_name)
    validation = validate_local_submission_inputs(args.test_csv, args.sample_submission)
    config_path = args.experiment_dir / f"{experiment_id}.json"
    payload = config_payload(args, experiment_id, validation)
    should_write_config = not args.dry_run or args.log_dry_run
    if should_write_config:
        write_json(config_path, payload)

    if args.dry_run:
        if args.log_dry_run:
            append_experiment(
                args.experiment_log,
                {
                    "experiment_id": experiment_id,
                    "created_at_utc": utc_now(),
                    "stage": args.stage,
                    "hypothesis": args.hypothesis,
                    "status": "dry_run_validated",
                    "dataset_version": args.dataset_version,
                    "model": args.model,
                    "config_path": str(config_path),
                    "output_path": str(args.output or ""),
                    "submitted": "false",
                    "notes": args.notes,
                },
            )
        print(json.dumps({
            "experiment_id": experiment_id,
            "config_snapshot_written": should_write_config,
            "config_path": str(config_path) if should_write_config else None,
            **validation,
        }, indent=2))
        return

    import torch
    from tqdm import tqdm
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    if args.device == "auto":
        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    processor_source = args.adapter if args.adapter is not None and args.adapter.exists() else args.model
    processor = WhisperProcessor.from_pretrained(processor_source, task="transcribe")
    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.to(args.device)
    model.eval()

    rows = build_prediction_rows(args.languages, args.split, args.test_csv)
    if args.max_samples:
        rows = rows[:args.max_samples]
    predictions: dict[str, str] = {}
    with torch.no_grad():
        for batch in tqdm(batch_iter(rows, args.batch_size), desc="predict", total=(len(rows) + args.batch_size - 1) // args.batch_size):
            audios = [item["audio"]["array"] for item in batch]
            sampling_rates = {item["audio"]["sampling_rate"] for item in batch}
            if len(sampling_rates) != 1:
                raise ValueError(f"Mixed sampling rates in batch: {sampling_rates}")
            inputs = processor.feature_extractor(
                audios,
                sampling_rate=next(iter(sampling_rates)),
                return_tensors="pt",
            ).input_features.to(args.device)
            generated = model.generate(
                inputs,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
            )
            texts = processor.batch_decode(generated, skip_special_tokens=True)
            for item, text in zip(batch, texts, strict=True):
                predictions[str(item["id"])] = submission_text(text)

    if args.raw_predictions:
        args.raw_predictions.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(
            [{"ID": row_id, "prediction": text} for row_id, text in predictions.items()]
        ).to_csv(args.raw_predictions, index=False)

    write_submission(predictions, args.sample_submission, args.output)
    append_experiment(
        args.experiment_log,
        {
            "experiment_id": experiment_id,
            "created_at_utc": utc_now(),
            "stage": args.stage,
            "hypothesis": args.hypothesis,
            "status": "completed_unsubmitted",
            "dataset_version": args.dataset_version,
            "model": args.model,
            "config_path": str(config_path),
            "output_path": str(args.output),
            "submitted": "false",
            "notes": args.notes,
        },
    )
    print(f"Wrote {len(predictions)} predictions to {args.output}")


if __name__ == "__main__":
    main()
