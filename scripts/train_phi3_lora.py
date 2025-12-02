import os

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ----------------------
# Paths / constants
# ----------------------
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

HF_HOME = os.environ.get(
    "HF_HOME",
    os.path.join(PROJECT_ROOT, ".cache", "huggingface"),
)

DATA_PATH = os.path.join(PROJECT_ROOT, "twitch_data", "processed", "supertf_all.txt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "phi3_supertf_lora")

# If you already have a snapshot downloaded, point here.
# You can override with: export PHI3_LOCAL_DIR=/path/to/snapshot
PHI3_LOCAL_DIR = os.environ.get(
    "PHI3_LOCAL_DIR",
    os.path.join(
        HF_HOME,
        "hub",
        "models--microsoft--Phi-3-mini-4k-instruct",
        "snapshots",
        "0a67737cc96d2554230f90338b163bc6380a2a85",  # adjust if your hash differs
    ),
)

# LoRA hyperparams
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05


# ----------------------
# Tokenizer / dataset
# ----------------------
def load_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        use_fast=True,
        trust_remote_code=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer


def load_dataset_tokenized(tokenizer, max_len: int = 32):
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    print(f"Loading raw text dataset from {DATA_PATH}")
    raw_datasets = load_dataset("text", data_files={"train": DATA_PATH})
    ds = raw_datasets["train"]

    # --- Filter to short, Twitchy lines ---
    def is_twitchy_short(example):
        text = example["text"].strip()
        if not text:
            return False

        # You can tune these thresholds
        num_words = len(text.split())
        num_ats = text.count("@")

        if num_words > 10:        # prefer ~3–10 word lines
            return False
        if num_ats > 1:           # avoid lots of @spam
            return False

        # optionally remove all-emoji / all punctuation lines if you want
        return True

    print("Filtering dataset for short, low-@ messages...")
    ds = ds.filter(is_twitchy_short)

    print(f"Kept {len(ds)} short messages after filtering.")

    # Small eval split
    split = ds.train_test_split(test_size=0.02, seed=42)
    train_ds = split["train"]
    eval_ds = split["test"]

    # --- Tokenization per line ---
    def tokenize_fn(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            max_length=max_len,
            padding="max_length",   # fixed-length sequences
        )

    print("Tokenizing dataset...")
    tokenized_train = train_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )
    tokenized_eval = eval_ds.map(
        tokenize_fn,
        batched=True,
        num_proc=1,
        remove_columns=["text"],
    )

    # For causal LM, labels = input_ids
    def add_labels(batch):
        batch["labels"] = batch["input_ids"].copy()
        return batch

    tokenized_train = tokenized_train.map(add_labels, batched=True)
    tokenized_eval = tokenized_eval.map(add_labels, batched=True)

    return tokenized_train, tokenized_eval


# ----------------------
# Model / QLoRA
# ----------------------
def load_model_for_qlora():
    print(f"HF_HOME resolved to: {HF_HOME}")
    print(f"Loading Phi-3 from local dir: {PHI3_LOCAL_DIR}")
    if not os.path.isdir(PHI3_LOCAL_DIR):
        raise RuntimeError(
            f"PHI3_LOCAL_DIR does not exist: {PHI3_LOCAL_DIR}\n"
            "Check your HF_HOME / snapshot hash or override PHI3_LOCAL_DIR."
        )

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    model = AutoModelForCausalLM.from_pretrained(
        PHI3_LOCAL_DIR,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
        local_files_only=True,  # <-- don't hit HF again
    )

    model.config.use_cache = False

    # prepare for k-bit training
    model = prepare_model_for_kbit_training(model)

    # LoRA config (standard Phi-style target modules for attention + MLP)
    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model


# ----------------------
# Main
# ----------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Using PROJECT_ROOT={PROJECT_ROOT}")
    print(f"Saving LoRA adapters to {OUTPUT_DIR}")

    tokenizer = load_tokenizer()
    # shorter sequences, one line at a time
    train_ds, eval_ds = load_dataset_tokenized(tokenizer, max_len=32)

    model = load_model_for_qlora()

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=1.0,          # you can bump to 2–3 later
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=8,
        eval_strategy="steps",
        eval_steps=200,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        learning_rate=2e-4,
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        report_to=["tensorboard"],
        logging_dir=os.path.join(OUTPUT_DIR, "logs"),
        fp16=False,
        gradient_checkpointing=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        data_collator=data_collator,
    )

    print("Starting training...")
    trainer.train()

    print("Saving final adapter + tokenizer...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done.")


if __name__ == "__main__":
    main()