"""Fine-tune Whisper with LoRA for the Waxal ASR challenge."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

from .config import TARGET_LANGUAGES
from .hf_data import load_many_splits, load_zindi_labelled_split
from .metrics import waxal_score


@dataclass
class SpeechSeq2SeqCollator:
    processor: WhisperProcessor
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        input_features = [{"input_features": item["input_features"]} for item in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        label_features = [{"input_ids": item["labels"]} for item in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch


def prepare_dataset(batch: dict, feature_extractor: WhisperFeatureExtractor, tokenizer: WhisperTokenizer) -> dict:
    audio = batch["audio"]
    batch["input_features"] = feature_extractor(
        audio["array"],
        sampling_rate=audio["sampling_rate"],
    ).input_features[0]
    batch["labels"] = tokenizer(batch["text"]).input_ids
    return batch


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openai/whisper-large-v3-turbo")
    parser.add_argument("--languages", nargs="+", default=list(TARGET_LANGUAGES))
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--validation-split", default="validation")
    parser.add_argument("--zindi-train-csv", default=None)
    parser.add_argument("--output-dir", default="models/whisper-lora")
    parser.add_argument("--sampling-rate", type=int, default=16_000)
    parser.add_argument("--whisper-language", default=None)
    parser.add_argument("--max-train-samples", type=int, default=None)
    parser.add_argument("--max-eval-samples", type=int, default=None)
    parser.add_argument("--num-proc", type=int, default=1)
    parser.add_argument("--per-device-train-batch-size", type=int, default=4)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--num-train-epochs", type=float, default=3.0)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=25)
    parser.add_argument("--generation-max-length", type=int, default=225)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument("--lora-target-modules", default="q_proj,v_proj")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--bf16", action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    feature_extractor = WhisperFeatureExtractor.from_pretrained(args.model)
    tokenizer = WhisperTokenizer.from_pretrained(args.model, task="transcribe")
    if args.whisper_language:
        tokenizer.set_prefix_tokens(language=args.whisper_language, task="transcribe")
    processor = WhisperProcessor.from_pretrained(args.model, task="transcribe")
    processor.tokenizer = tokenizer

    if args.zindi_train_csv:
        train_dataset = load_zindi_labelled_split(
            args.zindi_train_csv,
            args.train_split,
            languages=args.languages,
            sampling_rate=args.sampling_rate,
            decode_audio=True,
        )
        eval_dataset = load_zindi_labelled_split(
            args.zindi_train_csv,
            args.validation_split,
            languages=args.languages,
            sampling_rate=args.sampling_rate,
            decode_audio=True,
        )
    else:
        train_dataset = load_many_splits(
            args.languages,
            args.train_split,
            sampling_rate=args.sampling_rate,
            decode_audio=True,
            keep_text=True,
        )
        eval_dataset = load_many_splits(
            args.languages,
            args.validation_split,
            sampling_rate=args.sampling_rate,
            decode_audio=True,
            keep_text=True,
        )

    if args.max_train_samples:
        train_dataset = train_dataset.shuffle(seed=13).select(range(args.max_train_samples))
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(args.max_eval_samples))

    map_kwargs = {}
    if args.num_proc and args.num_proc > 1:
        map_kwargs["num_proc"] = args.num_proc

    train_dataset = train_dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=train_dataset.column_names,
        **map_kwargs,
    )
    eval_dataset = eval_dataset.map(
        lambda batch: prepare_dataset(batch, feature_extractor, tokenizer),
        remove_columns=eval_dataset.column_names,
        **map_kwargs,
    )

    model = WhisperForConditionalGeneration.from_pretrained(args.model)
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.config.use_cache = False
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=[name.strip() for name in args.lora_target_modules.split(",") if name.strip()],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM,
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    collator = SpeechSeq2SeqCollator(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    def compute_metrics(prediction: Any) -> dict[str, float]:
        pred_ids = prediction.predictions
        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]
        label_ids = prediction.label_ids
        label_ids[label_ids == -100] = tokenizer.pad_token_id
        pred_text = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_text = tokenizer.batch_decode(label_ids, skip_special_tokens=True)
        scores = waxal_score(label_text, pred_text)
        return {
            "wer": scores["wer"],
            "cer": scores["cer"],
            "waxal_error": scores["error"],
            "waxal_score": scores["score"],
            "mean_pred_len": float(np.mean([len(text) for text in pred_text])),
        }

    training_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        num_train_epochs=args.num_train_epochs,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        predict_with_generate=True,
        generation_max_length=args.generation_max_length,
        fp16=args.fp16,
        bf16=args.bf16,
        gradient_checkpointing=args.gradient_checkpointing,
        remove_unused_columns=False,
        label_names=["labels"],
        load_best_model_at_end=True,
        metric_for_best_model="waxal_score",
        greater_is_better=True,
        save_total_limit=3,
        report_to=["tensorboard"],
        optim="adamw_torch",
    )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    processor.save_pretrained(args.output_dir)


if __name__ == "__main__":
    main()
