"""Shared challenge configuration."""

HF_DATASET = "google/WaxalNLP"

TARGET_LANGUAGES = ("lin", "sna", "lug")

TEXT_COLUMN_CANDIDATES = (
    "transcription",
    "sentence",
    "text",
    "normalized_text",
    "target",
    "prediction",
    "predicted",
    "transcript",
    "label",
)

ID_COLUMN_CANDIDATES = (
    "ID",
    "id",
    "uid",
    "audio_id",
    "file_id",
    "path",
)

AUDIO_COLUMN_CANDIDATES = (
    "audio",
    "speech",
    "path",
    "file",
    "audio_path",
)

LANGUAGE_COLUMN_CANDIDATES = (
    "language",
    "lang",
    "locale",
    "config",
)
