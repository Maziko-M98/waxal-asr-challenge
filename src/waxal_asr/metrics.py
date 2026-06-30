"""Local implementation of the challenge metric."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

from .normalization import normalize_text


def edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> int:
    """Compute Levenshtein distance with O(min(n, m)) memory."""
    if len(reference) < len(hypothesis):
        reference, hypothesis = hypothesis, reference

    previous = list(range(len(hypothesis) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, hyp_token in enumerate(hypothesis, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            replace_cost = previous[j - 1] + (ref_token != hyp_token)
            current.append(min(insert_cost, delete_cost, replace_cost))
        previous = current
    return previous[-1]


def word_error_rate(
    references: Iterable[object],
    hypotheses: Iterable[object],
    *,
    normalize: bool = True,
) -> float:
    edits = 0
    total = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_text = normalize_text(reference) if normalize else str(reference)
        hyp_text = normalize_text(hypothesis) if normalize else str(hypothesis)
        ref_words = ref_text.split()
        hyp_words = hyp_text.split()
        edits += edit_distance(ref_words, hyp_words)
        total += len(ref_words)
    return edits / max(total, 1)


def char_error_rate(
    references: Iterable[object],
    hypotheses: Iterable[object],
    *,
    normalize: bool = True,
) -> float:
    edits = 0
    total = 0
    for reference, hypothesis in zip(references, hypotheses, strict=True):
        ref_text = normalize_text(reference) if normalize else str(reference)
        hyp_text = normalize_text(hypothesis) if normalize else str(hypothesis)
        edits += edit_distance(tuple(ref_text), tuple(hyp_text))
        total += len(ref_text)
    return edits / max(total, 1)


def waxal_score(
    references: Iterable[object],
    hypotheses: Iterable[object],
    *,
    normalize: bool = True,
) -> dict[str, float]:
    """Return component errors plus the likely Zindi leaderboard score.

    The competition text describes the metric as the weighted mean of WER and
    CER. The authenticated leaderboard ranks higher values above lower values,
    so we track both the raw weighted error and the display-style score.
    """
    refs = list(references)
    hyps = list(hypotheses)
    wer = word_error_rate(refs, hyps, normalize=normalize)
    cer = char_error_rate(refs, hyps, normalize=normalize)
    error = 0.5 * wer + 0.5 * cer
    return {"wer": wer, "cer": cer, "error": error, "score": 1.0 - error}
