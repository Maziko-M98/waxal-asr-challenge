"""Hugging Face dataset loading helpers."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path

from datasets import Audio, Dataset, concatenate_datasets, load_dataset

from .columns import audio_column, id_column, text_column
from .config import HF_DATASET, TARGET_LANGUAGES
from .zindi_csv import read_zindi_csv


def parse_languages(languages: Iterable[str] | None) -> list[str]:
    if languages is None:
        return list(TARGET_LANGUAGES)
    parsed = [language.strip() for language in languages if language.strip()]
    return parsed or list(TARGET_LANGUAGES)


def load_language_split(
    language: str,
    split: str,
    *,
    sampling_rate: int = 16_000,
    decode_audio: bool = True,
    keep_text: bool = True,
) -> Dataset:
    dataset = load_dataset(HF_DATASET, language, split=split)
    columns = dataset.column_names
    audio_col = audio_column(columns)
    id_col = id_column(columns)
    text_col = text_column(columns) if keep_text else None

    def to_common(example: dict) -> dict:
        row = {
            "id": str(example[id_col]),
            "language": language,
            "audio": example[audio_col],
        }
        if text_col is not None:
            row["text"] = example[text_col]
        return row

    keep_columns = ["id", "language", "audio"] + (["text"] if keep_text else [])
    dataset = dataset.map(to_common, remove_columns=columns)
    dataset = dataset.select_columns(keep_columns)
    dataset = dataset.cast_column(
        "audio",
        Audio(sampling_rate=sampling_rate, decode=decode_audio),
    )
    return dataset


def load_many_splits(
    languages: Iterable[str],
    split: str,
    *,
    sampling_rate: int = 16_000,
    decode_audio: bool = True,
    keep_text: bool = True,
) -> Dataset:
    parts = [
        load_language_split(
            language,
            split,
            sampling_rate=sampling_rate,
            decode_audio=decode_audio,
            keep_text=keep_text,
        )
        for language in languages
    ]
    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)


def load_zindi_labelled_split(
    train_csv: str | Path,
    original_split: str,
    *,
    languages: Iterable[str],
    sampling_rate: int = 16_000,
    decode_audio: bool = True,
) -> Dataset:
    """Load HF audio for IDs listed in Zindi Train.csv."""
    zindi_train = read_zindi_csv(train_csv)
    required = {"id", "transcription", "language", "original_split"}
    missing = required - set(zindi_train.columns)
    if missing:
        raise ValueError(f"{train_csv} is missing required columns: {sorted(missing)}")

    languages = list(languages)
    wanted = zindi_train[
        (zindi_train["original_split"].astype(str) == original_split)
        & (zindi_train["language"].astype(str).isin(languages))
    ].copy()
    if wanted.empty:
        raise ValueError(f"No rows for split={original_split!r}, languages={languages!r}")

    by_language = {
        language: frame
        for language, frame in wanted.groupby("language", sort=False)
    }
    parts = []
    for language, frame in by_language.items():
        id_to_text = dict(zip(frame["id"].astype(str), frame["transcription"], strict=True))
        ids = set(id_to_text)
        split = load_language_split(
            language,
            original_split,
            sampling_rate=sampling_rate,
            decode_audio=decode_audio,
            keep_text=True,
        )
        split = split.filter(lambda row: str(row["id"]) in ids)
        split = split.map(lambda row: {"text": id_to_text[str(row["id"])]})
        found = set(split["id"])
        missing_ids = sorted(ids - found)
        if missing_ids:
            raise ValueError(
                f"Missing {len(missing_ids)} HF audio rows for {language}/{original_split}. "
                f"First missing IDs: {missing_ids[:5]}"
            )
        parts.append(split)

    if len(parts) == 1:
        return parts[0]
    return concatenate_datasets(parts)
