# Google Waxal ASR Challenge Starter

This is a reproducible starter kit for the Zindi Google Waxal ASR Challenge.

Competition facts captured on 2026-06-30:
- Target Phase 1 languages: Lingala (`lin`), Shona (`sna`), Luganda (`lug`).
- Underlying error metric: `0.5 * WER + 0.5 * CER`.
- Authenticated leaderboard appears to display `1 - error`, so higher public scores are better.
- External public open-source speech or language data is allowed if disclosed.
- Phase 2 introduces a hidden test set with no metadata, so the solution should not depend on test-time language labels.
- Start: 2026-06-26. Close and reveal: 2026-08-03.
- Submission limits: 5 per day, 200 total.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e .
pip install -r requirements.txt
```

Download `Train.csv`, `Test.csv`, `SampleSubmission.csv`, and optionally
`Waxal_Challenge_Starter_Code.ipynb` from Zindi after joining the competition,
then place them in:

```text
data/zindi/
```

The audio is read from the Hugging Face dataset `google/WaxalNLP`.

Inspect the downloaded CSVs:

```powershell
python -m waxal_asr.inspect_zindi_csv --data-dir data/zindi
```

## First Baseline

Generate the EDA and validation-readiness report:

```powershell
python -m waxal_asr.eda_report --data-dir data/zindi --out-dir reports/eda
```

Create manifests for the target languages:

```powershell
python -m waxal_asr.prepare_manifests --languages lin sna lug --splits train validation test --out data/manifests
```

Train a Whisper LoRA model:

```powershell
accelerate launch -m waxal_asr.train_whisper_lora `
  --model openai/whisper-large-v3-turbo `
  --languages lin sna lug `
  --zindi-train-csv data/zindi/Train.csv `
  --train-split train `
  --validation-split validation `
  --output-dir models/whisper-large-v3-turbo-lora-target `
  --per-device-train-batch-size 4 `
  --gradient-accumulation-steps 8 `
  --learning-rate 1e-4 `
  --num-train-epochs 3
```

Generate a submission:

```powershell
python -m waxal_asr.infer_whisper `
  --config configs/baselines/zero_shot_whisper_large_v3_turbo.json
```

Validate a planned submission run without downloading models or audio:

```powershell
python -m waxal_asr.infer_whisper `
  --config configs/baselines/zero_shot_whisper_large_v3_turbo.json `
  --dry-run
```

Score a local validation prediction file:

```powershell
python -m waxal_asr.score_text `
  --reference-csv data/local_val_reference.csv `
  --prediction-csv submissions/local_val_predictions.csv
```

Create a sliced validation report:

```powershell
python -m waxal_asr.validation_report `
  --reference-csv data/zindi/Train.csv `
  --prediction-csv submissions/local_val_predictions.csv `
  --out-dir reports/validation/my_run
```

## Winning Game Plan

1. Ship a fast target-language Whisper LoRA baseline and submit it early.
2. Build trustworthy validation: score by language, clip duration, speaker, and transcript length. Do not chase public leaderboard noise.
3. Train in two stages: all allowed WaxalNLP ASR languages first, then target-only fine-tuning with oversampling for weaker languages.
4. Add a second architecture, usually a CTC model such as XLS-R or MMS, with language-model rescoring from allowed text. Ensembles often catch different errors than Whisper.
5. Prepare for Phase 2 by adding audio-only language identification or a model path that does not require language metadata.
6. Keep a reproducible disclosure log for every external dataset and text source used.

## Verified Data Files

- `Train.csv`: Zindi train metadata with target transcript column.
- `Test.csv`: Zindi test metadata without target transcript column.
- `SampleSubmission.csv`: required submission shape.
- `Waxal_Challenge_Starter_Code.ipynb`: official starter notebook.

The downloaded `Train.csv` has 38,199 labelled examples: 33,964 train and
4,235 validation. Language counts are `lin=16,244`, `sna=15,836`, and
`lug=6,119`.

## Professional Workflow

- Read [docs/EXPERIMENT_PROTOCOL.md](docs/EXPERIMENT_PROTOCOL.md) before running model experiments.
- Use [docs/FREE_GPU_WORKFLOW.md](docs/FREE_GPU_WORKFLOW.md) for Colab/Kaggle execution.
- Notebook launchers live in [notebooks/](notebooks/).
- Track every run in [experiments/experiment_log.csv](experiments/experiment_log.csv).
- Commit code, configs, and reports. Keep downloaded competition files and model artifacts local unless there is a deliberate release decision.
