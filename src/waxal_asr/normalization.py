"""Text normalization used for local scoring and submission cleanup."""

from __future__ import annotations

import re
import string
import unicodedata

_PUNCT_TO_SPACE = str.maketrans({ch: " " for ch in string.punctuation})
_MULTISPACE = re.compile(r"\s+")


def normalize_text(text: object, *, strip_accents: bool = False) -> str:
    """Normalize transcript text without making language-specific assumptions."""
    value = "" if text is None else str(text)
    value = unicodedata.normalize("NFKC", value).lower()
    value = value.replace("'", " ").replace("`", " ").replace("’", " ")
    if strip_accents:
        value = "".join(
            ch for ch in unicodedata.normalize("NFD", value)
            if unicodedata.category(ch) != "Mn"
        )
    value = value.translate(_PUNCT_TO_SPACE)
    value = _MULTISPACE.sub(" ", value)
    return value.strip()


def submission_text(text: object) -> str:
    """Conservative cleanup for model outputs before writing a submission."""
    return normalize_text(text, strip_accents=False)
