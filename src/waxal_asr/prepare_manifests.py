"""Create lightweight JSONL manifests from WaxalNLP splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tqdm import tqdm

from .config import TARGET_LANGUAGES
from .hf_data import load_language_split


def audio_pointer(audio: object) -> dict:
    if isinstance(audio, dict):
        return {
            "path": audio.get("path"),
            "bytes_present": audio.get("bytes") is not None,
        }
    return {"path": str(audio), "bytes_present": False}


def write_manifest(
    language: str,
    split: str,
    output_path: Path,
    *,
    sampling_rate: int,
) -> None:
    dataset = load_language_split(
        language,
        split,
        sampling_rate=sampling_rate,
        decode_audio=False,
        keep_text=split != "test",
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in tqdm(dataset, desc=f"{language}/{split}"):
            record = {
                "id": row["id"],
                "language": row["language"],
                "split": split,
                "audio": audio_pointer(row["audio"]),
            }
            if "text" in row:
                record["text"] = row["text"]
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--languages", nargs="+", default=list(TARGET_LANGUAGES))
    parser.add_argument("--splits", nargs="+", default=["train", "validation", "test"])
    parser.add_argument("--out", type=Path, default=Path("data/manifests"))
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    args = parser.parse_args()

    for language in args.languages:
        for split in args.splits:
            output_path = args.out / f"{language}_{split}.jsonl"
            write_manifest(language, split, output_path, sampling_rate=args.sampling_rate)


if __name__ == "__main__":
    main()
