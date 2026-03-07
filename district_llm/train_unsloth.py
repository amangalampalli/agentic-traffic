from __future__ import annotations

import argparse
from pathlib import Path

from district_llm.data import load_jsonl_text_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Supervised fine-tune a district LLM on DQN-derived district traces with Unsloth/QLoRA."
    )
    parser.add_argument("--dataset", required=True, help="JSONL dataset with a 'text' field.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model-name", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--lora-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--warmup-steps", type=int, default=50)
    parser.add_argument("--max-steps", type=int, default=500)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--save-steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--dataset-num-proc", type=int, default=2)
    parser.add_argument("--eval-dataset", default=None)
    parser.add_argument("--resume-from-checkpoint", default=None)
    parser.add_argument(
        "--include-non-dqn-sources",
        action="store_true",
        help="By default the trainer keeps only DQN-derived rows (controller_family=dqn).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    try:
        import torch
        from trl import SFTTrainer
        from transformers import TrainingArguments
        from unsloth import FastLanguageModel
    except ImportError as exc:
        raise ImportError(
            "District LLM training requires 'unsloth' and 'trl'. "
            "Install them in the active environment before running this entry point."
        ) from exc

    controller_families = None if args.include_non_dqn_sources else ["dqn"]
    train_dataset = load_jsonl_text_dataset(
        args.dataset,
        controller_families=controller_families,
    )
    eval_dataset = (
        load_jsonl_text_dataset(
            args.eval_dataset,
            controller_families=controller_families,
        )
        if args.eval_dataset
        else None
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.model_name,
        max_seq_length=args.max_seq_length,
        dtype=None,
        load_in_4bit=bool(args.load_in_4bit),
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=args.lora_rank,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        dataset_num_proc=args.dataset_num_proc,
        packing=False,
        args=TrainingArguments(
            output_dir=str(output_dir),
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            warmup_steps=args.warmup_steps,
            max_steps=args.max_steps,
            learning_rate=args.learning_rate,
            logging_steps=args.logging_steps,
            save_steps=args.save_steps,
            bf16=bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()),
            fp16=bool(torch.cuda.is_available() and not torch.cuda.is_bf16_supported()),
            optim="paged_adamw_8bit",
            lr_scheduler_type="cosine",
            seed=args.seed,
            report_to="none",
            evaluation_strategy="steps" if eval_dataset is not None else "no",
            eval_steps=args.save_steps if eval_dataset is not None else None,
        ),
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    model.save_pretrained(str(output_dir))
    tokenizer.save_pretrained(str(output_dir))


if __name__ == "__main__":
    main()
