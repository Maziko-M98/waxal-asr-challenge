# Notebook Runners

These notebooks are thin execution wrappers around the Python package in this
repository. They should not become separate sources of truth.

Use them for free hosted GPU environments:

- `colab_baseline_runner.ipynb`: quick smoke tests and small baseline runs on free Colab.
- `kaggle_baseline_runner.ipynb`: preferred free-GPU path for the full baseline run.

Keep the Zindi CSV files outside Git. In both environments, copy:

- `Train.csv`
- `Test.csv`
- `SampleSubmission.csv`

into `data/zindi/` before running the baseline.

The first run should use `--max-samples 20`. Only run the full 4,253-row test
after the smoke output looks valid.
