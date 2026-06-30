"""Robust readers for the downloaded Zindi CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def read_zindi_csv(path: str | Path, **kwargs) -> pd.DataFrame:
    """Read challenge CSVs, including Train rows with escaped quotes.

    `Train.csv` contains transcript text that trips Pandas' default C parser.
    The Python parser with a backslash escape character preserves all observed
    rows in the authenticated download.
    """
    return pd.read_csv(
        path,
        engine="python",
        quotechar='"',
        escapechar="\\",
        **kwargs,
    )
