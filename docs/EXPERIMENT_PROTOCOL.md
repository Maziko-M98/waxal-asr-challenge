# Experiment Protocol

This project is run as a disciplined data science competition effort. The goal
is to separate signal from noise and build models that generalize to the Phase 2
audio-only evaluation set.

## Principles

1. Every experiment must have a written hypothesis.
2. Every leaderboard submission must be reproducible from a committed config.
3. The public leaderboard is a checkpoint, not the optimizer.
4. The official `original_split=validation` split is the primary local gate.
5. Report language-level metrics, not only aggregate score.
6. Phase 2 constraints dominate Phase 1 convenience: no final method may depend
   on test metadata or ID-derived language labels.

## Required Experiment Record

Each experiment records:

- experiment ID
- timestamp
- hypothesis
- dataset fingerprints
- model and decoding settings
- local WER, CER, weighted error, and leaderboard-style score
- output path
- public score, only if submitted
- notes explaining the decision

## Submission Policy

Early submissions are allowed for plumbing validation. After that, submit only
when the local validation report justifies the change. Avoid tiny unmotivated
changes such as random beam sizes, prompt wording, or post-processing rules
unless they test a specific failure mode found in validation error analysis.

## First Experiment Ladder

1. `eda_001`: text and split EDA, no leaderboard submission.
2. `baseline_001`: zero-shot Whisper submission to validate the full pipeline.
3. `train_001`: Whisper LoRA on official train split, evaluated on validation.
4. `train_002`: Whisper LoRA with Luganda balancing, compared against `train_001`.
5. `train_003`: official Gemma 3n LoRA baseline as an independent model family.

## Decision Standard

A change is worth keeping when it improves validation error overall or improves a
targeted weakness without unacceptable regressions elsewhere. A change is not
worth keeping merely because it improves a public score once.
