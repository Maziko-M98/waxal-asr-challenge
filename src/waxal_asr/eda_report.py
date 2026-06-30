"""Generate an EDA and validation report for the Waxal ASR challenge."""

from __future__ import annotations

import argparse
import re
import string
from collections import Counter
from pathlib import Path

import pandas as pd

from .experiment_log import dataset_fingerprint, utc_now
from .normalization import normalize_text
from .zindi_csv import read_zindi_csv

TOKEN_RE = re.compile(r"\b\w+\b", flags=re.UNICODE)


def id_prefix(value: object) -> str:
    text = str(value)
    return text.split("_", 1)[0] if "_" in text else ""


def tokenize(text: object) -> list[str]:
    return TOKEN_RE.findall(normalize_text(text))


def add_text_features(table: pd.DataFrame) -> pd.DataFrame:
    enriched = table.copy()
    normalized = enriched["transcription"].map(normalize_text)
    enriched["char_len"] = normalized.map(len)
    enriched["word_len"] = normalized.map(lambda text: len(text.split()))
    enriched["unique_word_len"] = normalized.map(lambda text: len(set(text.split())))
    enriched["comma_count"] = enriched["transcription"].astype(str).str.count(",")
    enriched["quote_count"] = enriched["transcription"].astype(str).str.count('"')
    enriched["apostrophe_count"] = enriched["transcription"].astype(str).str.count("'")
    enriched["digit_count"] = enriched["transcription"].astype(str).str.count(r"\d")
    enriched["punct_count"] = enriched["transcription"].astype(str).map(
        lambda text: sum(ch in string.punctuation for ch in text)
    )
    return enriched


def quantile_summary(table: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (language, split), frame in table.groupby(["language", "original_split"], sort=True):
        row = {"language": language, "original_split": split, "rows": int(len(frame))}
        for column in ["char_len", "word_len", "unique_word_len", "punct_count"]:
            stats = frame[column].quantile([0.0, 0.25, 0.5, 0.75, 0.95, 1.0])
            row.update({
                f"{column}_min": float(stats.loc[0.0]),
                f"{column}_p25": float(stats.loc[0.25]),
                f"{column}_median": float(stats.loc[0.5]),
                f"{column}_p75": float(stats.loc[0.75]),
                f"{column}_p95": float(stats.loc[0.95]),
                f"{column}_max": float(stats.loc[1.0]),
            })
        rows.append(row)
    return pd.DataFrame(rows)


def validation_oov(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for language, frame in train.groupby("language", sort=True):
        train_tokens = Counter()
        val_tokens = Counter()
        for text in frame.loc[frame["original_split"] == "train", "transcription"]:
            train_tokens.update(tokenize(text))
        for text in frame.loc[frame["original_split"] == "validation", "transcription"]:
            val_tokens.update(tokenize(text))
        unseen = {token: count for token, count in val_tokens.items() if token not in train_tokens}
        total_val_tokens = sum(val_tokens.values())
        rows.append({
            "language": language,
            "train_vocab": len(train_tokens),
            "validation_vocab": len(val_tokens),
            "validation_token_count": total_val_tokens,
            "validation_unseen_vocab": len(unseen),
            "validation_unseen_token_count": sum(unseen.values()),
            "validation_unseen_token_rate": sum(unseen.values()) / max(total_val_tokens, 1),
        })
    return pd.DataFrame(rows)


def top_tokens(train: pd.DataFrame, out_dir: Path, *, top_n: int) -> None:
    for language, frame in train.groupby("language", sort=True):
        counts = Counter()
        for text in frame["transcription"]:
            counts.update(tokenize(text))
        pd.DataFrame(
            [{"token": token, "count": count} for token, count in counts.most_common(top_n)]
        ).to_csv(out_dir / f"top_tokens_{language}.csv", index=False)


def quality_flags(train: pd.DataFrame, test: pd.DataFrame, sample: pd.DataFrame) -> pd.DataFrame:
    rows = []
    train_ids = set(train["id"].astype(str))
    test_ids = set(test["ID"].astype(str))
    sample_ids = set(sample["ID"].astype(str))
    rows.append({"check": "duplicate_train_ids", "value": int(train["id"].duplicated().sum()), "severity": "high"})
    rows.append({"check": "duplicate_test_ids", "value": int(test["ID"].duplicated().sum()), "severity": "high"})
    rows.append({"check": "train_test_id_overlap", "value": len(train_ids & test_ids), "severity": "high"})
    rows.append({"check": "test_sample_id_symmetric_difference", "value": len(test_ids ^ sample_ids), "severity": "high"})
    rows.append({"check": "empty_train_transcriptions", "value": int((train["transcription"].astype(str).str.len() == 0).sum()), "severity": "high"})
    rows.append({"check": "normalized_empty_train_transcriptions", "value": int((train["char_len"] == 0).sum()), "severity": "high"})
    rows.append({"check": "repeated_transcriptions", "value": int(train["transcription"].duplicated().sum()), "severity": "medium"})
    rows.append({"check": "transcriptions_with_digits", "value": int((train["digit_count"] > 0).sum()), "severity": "medium"})
    rows.append({"check": "transcriptions_with_quotes", "value": int((train["quote_count"] > 0).sum()), "severity": "medium"})
    return pd.DataFrame(rows)


def markdown_table(table: pd.DataFrame, *, max_rows: int = 20) -> str:
    if table.empty:
        return "_No rows._"
    shown = table.head(max_rows).copy()
    columns = [str(column) for column in shown.columns]
    rows = []
    for _, row in shown.iterrows():
        rows.append([str(row[column]).replace("|", "\\|") for column in shown.columns])
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(row) + " |" for row in rows]
    return "\n".join([header, separator, *body])


def build_report(data_dir: Path, out_dir: Path, *, top_n: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    train = read_zindi_csv(data_dir / "Train.csv")
    test = read_zindi_csv(data_dir / "Test.csv")
    sample = read_zindi_csv(data_dir / "SampleSubmission.csv")
    train = add_text_features(train)
    test = test.copy()
    test["id_prefix"] = test["ID"].map(id_prefix)

    split_counts = (
        train.groupby(["language", "original_split"], sort=True)
        .size()
        .reset_index(name="rows")
    )
    language_counts = train["language"].value_counts().rename_axis("language").reset_index(name="rows")
    test_prefix_counts = test["id_prefix"].value_counts().rename_axis("id_prefix").reset_index(name="rows")
    length_stats = quantile_summary(train)
    oov = validation_oov(train)
    flags = quality_flags(train, test, sample)

    split_counts.to_csv(out_dir / "language_split_counts.csv", index=False)
    language_counts.to_csv(out_dir / "language_counts.csv", index=False)
    test_prefix_counts.to_csv(out_dir / "test_id_prefix_counts.csv", index=False)
    length_stats.to_csv(out_dir / "transcript_length_by_language_split.csv", index=False)
    oov.to_csv(out_dir / "validation_oov_by_language.csv", index=False)
    flags.to_csv(out_dir / "quality_flags.csv", index=False)
    top_tokens(train, out_dir, top_n=top_n)

    fingerprint = dataset_fingerprint([
        data_dir / "Train.csv",
        data_dir / "Test.csv",
        data_dir / "SampleSubmission.csv",
        data_dir / "Waxal_Challenge_Starter_Code.ipynb",
    ])

    report = f"""# Waxal ASR EDA And Validation Report

Generated: {utc_now()}

## Executive Read

- Train rows: {len(train):,}
- Test rows: {len(test):,}
- Sample submission rows: {len(sample):,}
- Official validation rows inside `Train.csv`: {(train["original_split"] == "validation").sum():,}
- Target languages: {", ".join(sorted(train["language"].unique()))}

The first competitive priority is a trustworthy validation loop. Luganda is
materially underrepresented, so any fine-tuning plan should track language-level
metrics and consider balanced sampling before broader model changes.

## Dataset Integrity

{markdown_table(flags)}

## Language And Split Balance

{markdown_table(split_counts)}

## Overall Language Counts

{markdown_table(language_counts)}

## Test ID Prefix Counts

These prefixes are useful for Phase 1 diagnostics only. Phase 2 is audio-only
with no metadata, so final systems must not depend on ID-derived language.

{markdown_table(test_prefix_counts)}

## Transcript Length Statistics

{markdown_table(length_stats)}

## Validation Lexical Drift

This is a text-only proxy for how much validation vocabulary was unseen in the
training split. It should guide error analysis, not become a leaderboard hack.

{markdown_table(oov)}

## Immediate Hypotheses

1. Luganda balancing should improve `lug` validation WER/CER with limited cost to `lin` and `sna`.
2. Conservative transcript normalization should help scoring stability; aggressive language-specific normalization needs evidence.
3. A zero-shot Whisper submission is useful as a plumbing baseline, not as a model-quality claim.
4. Whisper LoRA and Gemma 3n LoRA are complementary enough to compare as separate model families before ensembling.

## Experiment Rules

- Use `original_split=validation` as the default validation benchmark.
- Report WER, CER, weighted error, and leaderboard-style score by language.
- Submit only when the local validation evidence supports the change.
- Keep Phase 2 constraints front and center: no metadata dependency.

## Dataset Fingerprints

```json
{pd.Series(fingerprint).to_json(indent=2)}
```
"""
    report_path = out_dir / "eda_report.md"
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data/zindi"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/eda"))
    parser.add_argument("--top-n", type=int, default=50)
    args = parser.parse_args()
    report_path = build_report(args.data_dir, args.out_dir, top_n=args.top_n)
    print(f"Wrote {report_path}")


if __name__ == "__main__":
    main()
