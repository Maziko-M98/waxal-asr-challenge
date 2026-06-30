# Competition Strategy

## What Matters

The underlying metric gives equal weight to word error rate and character error rate. The authenticated public leaderboard appears to display `1 - (0.5 * WER + 0.5 * CER)`, so higher public scores are better while lower raw error is still the thing to optimize.

The target languages are Lingala, Shona, and Luganda in Phase 1. Phase 2 removes metadata, so the final system should either work without language labels or include an audio-only language identifier.

## Submission Ladder

1. **Baseline A: target-only Whisper LoRA**
   - Train on `lin`, `sna`, and `lug` train splits.
   - Validate on the official validation split.
   - Oversample or upweight `lug`; it has about 6.1k labelled rows versus 16.2k `lin` and 15.8k `sna`.
   - Submit early to establish the public leaderboard scale.

2. **Baseline B: multilingual warm start**
   - Fine-tune on all allowed labeled WaxalNLP ASR languages.
   - Continue fine-tuning on the three target languages with balanced sampling.
   - Compare against Baseline A by language, not only overall score.

3. **Model diversity**
   - Reproduce the official Gemma 3n audio LoRA starter as a second family.
   - Add a CTC model such as XLS-R or MMS.
   - Train a character/subword tokenizer from allowed transcripts.
   - Rescore CTC outputs with per-language language models trained only from allowed text.

4. **External data**
   - Add public open-source data only when it is license-safe and disclosed.
   - Start with target-language speech if available, then text-only corpora for LM rescoring.
   - Keep `data/external_sources.csv` with dataset, URL, license, language, and purpose.

5. **Phase 2 hardening**
   - Train a simple language classifier using audio embeddings from WaxalNLP validation clips.
   - Test inference with language metadata intentionally removed.
   - Pick final submissions from validation performance and robustness, not from the last public leaderboard nudge.

## Error Analysis Checklist

- Score WER/CER separately for each language.
- Bucket by audio duration, transcript length, and speaking rate.
- Inspect the top 100 worst validation examples manually.
- Track spelling variants and decide whether to normalize them in post-processing.
- Compare model outputs where Whisper and CTC disagree; those examples are high-value for validation review.

## Reproducibility Rules

- Every submission CSV gets a matching config file and model checkpoint path.
- Never train on ground-truth labels from the test set.
- Keep the final two submissions boring and reproducible.
- Respect the 5-submissions-per-day and 200-total-submissions limits.
- Select the final two private leaderboard submissions before the challenge closes.
