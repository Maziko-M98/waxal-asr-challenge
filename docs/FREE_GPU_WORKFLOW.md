# Free GPU Workflow

Use free hosted GPUs deliberately. The goal is to run reproducible experiments,
not to create one-off notebook state.

## Recommendation

1. Use local Codex for code, reports, commits, and experiment design.
2. Use Colab free for quick smoke tests when a GPU is available.
3. Use Kaggle free GPU for the full zero-shot baseline if Colab free is unstable.
4. Save generated submissions and raw predictions from the notebook output area.
5. Record public submissions in `experiments/experiment_log.csv` after submitting.

## Colab Free

Best for:

- EDA
- dependency checks
- 10-20 sample smoke inference
- occasional full baseline if GPU availability is good

Risks:

- GPU availability is not guaranteed.
- Runtime can disconnect.
- Long full-test inference may need to be restarted.

## Kaggle Free GPU

Best for:

- full baseline inference
- repeatable notebook sessions with attached private datasets
- preserving outputs in `/kaggle/working`

Risks:

- GPU quota and accelerator availability are still limited.
- Internet access may need to be enabled in notebook settings for Hugging Face model/audio downloads.

## Data Handling

Do not commit Zindi files to GitHub. On Kaggle, create a private dataset with:

- `Train.csv`
- `Test.csv`
- `SampleSubmission.csv`

Current private Kaggle dataset:

- Dataset: `mazikomphepo/waxal-asr-zindi-csvs`
- Notebook input path: `/kaggle/input/waxal-asr-zindi-csvs`

On Colab, upload the CSVs manually or mount Google Drive.

## First Baseline Sequence

1. Run `inspect_zindi_csv`.
2. Run `eda_report`.
3. Run `infer_whisper --dry-run`.
4. Run `infer_whisper --max-samples 20`.
5. Inspect the 20-row output.
6. Run the full baseline only if the smoke test succeeds.
7. Submit once to Zindi and record the public score.
