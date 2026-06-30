"""Column-detection helpers for Zindi and Hugging Face tables."""

from __future__ import annotations

from collections.abc import Iterable

from .config import (
    AUDIO_COLUMN_CANDIDATES,
    ID_COLUMN_CANDIDATES,
    LANGUAGE_COLUMN_CANDIDATES,
    TEXT_COLUMN_CANDIDATES,
)


def first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    column_set = {str(col): str(col) for col in columns}
    lower_map = {str(col).lower(): str(col) for col in columns}
    for candidate in candidates:
        if candidate in column_set:
            return column_set[candidate]
        lowered = candidate.lower()
        if lowered in lower_map:
            return lower_map[lowered]
    return None


def require_column(columns: Iterable[str], candidates: Iterable[str], role: str) -> str:
    found = first_existing(columns, candidates)
    if found is None:
        raise ValueError(f"Could not find {role} column. Available columns: {list(columns)}")
    return found


def id_column(columns: Iterable[str]) -> str:
    return require_column(columns, ID_COLUMN_CANDIDATES, "ID")


def text_column(columns: Iterable[str]) -> str:
    return require_column(columns, TEXT_COLUMN_CANDIDATES, "text")


def audio_column(columns: Iterable[str]) -> str:
    return require_column(columns, AUDIO_COLUMN_CANDIDATES, "audio")


def language_column(columns: Iterable[str]) -> str | None:
    return first_existing(columns, LANGUAGE_COLUMN_CANDIDATES)
