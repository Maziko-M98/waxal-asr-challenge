# Waxal ASR EDA And Validation Report

Generated: 2026-06-30T16:13:53Z

## Executive Read

- Train rows: 38,199
- Test rows: 4,253
- Sample submission rows: 4,253
- Official validation rows inside `Train.csv`: 4,235
- Target languages: lin, lug, sna

The first competitive priority is a trustworthy validation loop. Luganda is
materially underrepresented, so any fine-tuning plan should track language-level
metrics and consider balanced sampling before broader model changes.

## Dataset Integrity

| check | value | severity |
| --- | --- | --- |
| duplicate_train_ids | 0 | high |
| duplicate_test_ids | 0 | high |
| train_test_id_overlap | 0 | high |
| test_sample_id_symmetric_difference | 0 | high |
| empty_train_transcriptions | 0 | high |
| normalized_empty_train_transcriptions | 1 | high |
| repeated_transcriptions | 12 | medium |
| transcriptions_with_digits | 176 | medium |
| transcriptions_with_quotes | 151 | medium |

## Language And Split Balance

| language | original_split | rows |
| --- | --- | --- |
| lin | train | 14400 |
| lin | validation | 1844 |
| lug | train | 5455 |
| lug | validation | 664 |
| sna | train | 14109 |
| sna | validation | 1727 |

## Overall Language Counts

| language | rows |
| --- | --- |
| lin | 16244 |
| sna | 15836 |
| lug | 6119 |

## Test ID Prefix Counts

These prefixes are useful for Phase 1 diagnostics only. Phase 2 is audio-only
with no metadata, so final systems must not depend on ID-derived language.

| id_prefix | rows |
| --- | --- |
| lin | 1866 |
| sna | 1749 |
| lug | 638 |

## Transcript Length Statistics

| language | original_split | rows | char_len_min | char_len_p25 | char_len_median | char_len_p75 | char_len_p95 | char_len_max | word_len_min | word_len_p25 | word_len_median | word_len_p75 | word_len_p95 | word_len_max | unique_word_len_min | unique_word_len_p25 | unique_word_len_median | unique_word_len_p75 | unique_word_len_p95 | unique_word_len_max | punct_count_min | punct_count_p25 | punct_count_median | punct_count_p75 | punct_count_p95 | punct_count_max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lin | train | 14400 | 0.0 | 106.0 | 144.0 | 185.0 | 254.0 | 493.0 | 0.0 | 19.0 | 27.0 | 35.0 | 48.0 | 99.0 | 0.0 | 15.0 | 19.0 | 23.0 | 29.0 | 51.0 | 0.0 | 1.0 | 1.0 | 2.0 | 5.0 | 18.0 |
| lin | validation | 1844 | 17.0 | 105.0 | 144.0 | 184.0 | 257.0 | 437.0 | 3.0 | 19.0 | 27.0 | 34.0 | 48.0 | 86.0 | 2.0 | 15.0 | 19.0 | 23.0 | 30.0 | 49.0 | 0.0 | 0.0 | 1.0 | 2.0 | 6.0 | 16.0 |
| lug | train | 5455 | 41.0 | 144.0 | 189.0 | 249.0 | 364.3000000000002 | 626.0 | 6.0 | 22.0 | 29.0 | 39.0 | 57.0 | 97.0 | 6.0 | 19.0 | 25.0 | 32.0 | 44.0 | 74.0 | 0.0 | 4.0 | 5.0 | 8.0 | 12.0 | 26.0 |
| lug | validation | 664 | 47.0 | 148.0 | 192.5 | 251.25 | 375.85 | 638.0 | 6.0 | 22.0 | 30.0 | 39.0 | 57.0 | 90.0 | 6.0 | 20.0 | 25.0 | 32.0 | 44.0 | 71.0 | 0.0 | 4.0 | 5.0 | 8.0 | 12.0 | 22.0 |
| sna | train | 14109 | 17.0 | 144.0 | 182.0 | 225.0 | 302.0 | 484.0 | 3.0 | 18.0 | 23.0 | 28.0 | 38.0 | 58.0 | 3.0 | 17.0 | 21.0 | 25.0 | 33.0 | 52.0 | 0.0 | 2.0 | 3.0 | 4.0 | 6.0 | 14.0 |
| sna | validation | 1727 | 45.0 | 140.0 | 181.0 | 223.0 | 300.6999999999998 | 450.0 | 5.0 | 18.0 | 22.0 | 28.0 | 37.0 | 59.0 | 5.0 | 16.0 | 21.0 | 25.0 | 32.0 | 49.0 | 0.0 | 2.0 | 3.0 | 4.0 | 6.0 | 14.0 |

## Validation Lexical Drift

This is a text-only proxy for how much validation vocabulary was unseen in the
training split. It should guide error analysis, not become a leaderboard hack.

| language | train_vocab | validation_vocab | validation_token_count | validation_unseen_vocab | validation_unseen_token_count | validation_unseen_token_rate |
| --- | --- | --- | --- | --- | --- | --- |
| lin | 16880 | 4688 | 50808 | 1432 | 1542 | 0.030349551251771374 |
| lug | 16524 | 4425 | 21305 | 1128 | 1210 | 0.05679417977000704 |
| sna | 36868 | 9373 | 40114 | 2732 | 2852 | 0.07109737248840804 |

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
{
  "data\\zindi\\Train.csv":{
    "exists":true,
    "bytes":7878834,
    "sha256":"f619124231242198"
  },
  "data\\zindi\\Test.csv":{
    "exists":true,
    "bytes":42083,
    "sha256":"866605cc2150eb01"
  },
  "data\\zindi\\SampleSubmission.csv":{
    "exists":true,
    "bytes":59102,
    "sha256":"b912b2138ce43587"
  },
  "data\\zindi\\Waxal_Challenge_Starter_Code.ipynb":{
    "exists":true,
    "bytes":289242,
    "sha256":"56777803a4d00959"
  }
}
```
