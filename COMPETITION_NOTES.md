# Verified Competition Notes

Captured from the authenticated Zindi pages on 2026-06-30.

- Challenge: Google WAXAL ASR Challenge.
- Prize pool: 10,000 USD.
- Start: 2026-06-26.
- Close: 2026-08-03.
- Reveal: 2026-08-03.
- Joined participants shown: 361.
- Active participants shown: 36.
- Your submissions shown: 0 / 200.
- Daily submission limit: 5.
- Total submission limit: 200.
- Max team size: 4.
- Target Phase 1 focus: Lingala, Shona, and Luganda.
- Phase 1 uses WAXAL train, validation, and test splits from Hugging Face.
- Phase 2 releases a new unseen audio-only test set about one week before close.
- Phase 2 provides no language, speaker, gender, or auxiliary metadata.
- External public open-source speech or language datasets are allowed if legally licensed and disclosed in final documentation.
- Challenge data is listed as CC-BY 4.0 in the rules summary. The longer generic rules text also mentions CC-BY-SA 4.0, so treat attribution/share-alike conservatively in documentation.
- Top 10 private leaderboard solutions will be asked for code and have 48 hours to submit it.

## Authenticated Data Files

- `Waxal_Challenge_Starter_Code.ipynb`, 282.5 KB.
- `SampleSubmission.csv`, 57.7 KB.
- `Test.csv`, 41.1 KB.
- `Train.csv`, 7.5 MB.

## Downloaded CSV Shape

- `Train.csv`: 38,199 rows, columns `id`, `transcription`, `language`, `original_split`.
- `Train.csv` split counts: 33,964 train, 4,235 validation.
- `Train.csv` language counts: `lin=16,244`, `sna=15,836`, `lug=6,119`.
- `Test.csv`: 4,253 rows, column `ID`.
- `SampleSubmission.csv`: 4,253 rows, columns `ID`, `Target`.
- `Train.csv` requires a robust CSV reader with backslash escape handling; the starter code uses `waxal_asr.zindi_csv.read_zindi_csv`.

## Leaderboard Read

The public leaderboard ranks higher numeric scores above lower scores. The public best observed was about `0.830226493`, while the benchmark was `0.000000000`. This strongly suggests Zindi displays an accuracy-like transformation of the described error metric:

```text
display_score = 1 - (0.5 * WER + 0.5 * CER)
```

The code tracks both `error` and `score` so we can optimize correctly either way.
