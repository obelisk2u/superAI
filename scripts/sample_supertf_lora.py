#!/usr/bin/env python3
import os
import re
from typing import List

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# --------------------
# Config
# --------------------
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
ADAPTER_DIR = os.path.join(PROJECT_ROOT, "phi3_supertf_lora")

ROLLING_WINDOW = 25
NUM_MESSAGES = 100

MAX_MSG_CHARS = 140
MAX_MENTIONS = 2
MAX_RETRIES = 6


# --------------------
# Model Loader
# --------------------
def load_model():
    print("Loading 4-bit base model...")

    quant_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    base = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        quantization_config=quant_cfg,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=False,
    )

    print("Loading LoRA adapter…")
    model = PeftModel.from_pretrained(base, ADAPTER_DIR)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_ID,
        trust_remote_code=False,
        use_fast=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model


# --------------------
# Cleaning functions
# --------------------
def strip_to_one_line(text: str) -> str:
    if "\n" in text:
        text = text.split("\n", 1)[0]
    return text.strip()


def remove_embedded_user_tags(text: str) -> str:
    """Cut off if the model tries to generate another 'userXX:' inside the output."""
    match = re.search(r"\buser\d+\s*:", text)
    if match:
        return text[: match.start()].rstrip()
    return text


def collapse_repeats(text: str) -> str:
    text = re.sub(r"([?!.,])\1{3,}", r"\1\1\1", text)
    text = re.sub(r"(.)\1{6,}", r"\1" * 6, text)
    return text


def looks_like_garbage(text: str) -> bool:
    if not text:
        return True

    # Too many @
    if text.count("@") > MAX_MENTIONS:
        return True

    # Too symbolic
    alnum = sum(c.isalnum() for c in text)
    if alnum == 0:
        return True
    if alnum / len(text) < 0.30:
        return True

    if len(text) < 2:
        return True

    return False


def clean_message(text: str) -> str:
    text = strip_to_one_line(text)
    text = remove_embedded_user_tags(text)
    text = collapse_repeats(text)
    text = text[:MAX_MSG_CHARS].rstrip()

    if looks_like_garbage(text):
        return ""
    return text


# --------------------
# Prompt Builder
# --------------------
def build_prompt(history: List[str]) -> str:
    return "\n".join(history) + "\nuserX:"


# --------------------
# Generation
# --------------------
def generate_next_message(history, tokenizer, model):
    prompt = build_prompt(history)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    for _ in range(MAX_RETRIES):
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=10,
                temperature=0.5,
                top_p=0.85,
                no_repeat_ngram_size=3,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.12,
            )

        decoded = tokenizer.decode(out[0], skip_special_tokens=False)

        # Slice off the prompt safely
        if decoded.startswith(prompt):
            raw = decoded[len(prompt):].strip()
        else:
            # fallback for tokenization edge cases
            raw = decoded.split("userX:", 1)[-1].strip()

        msg = clean_message(raw)
        if msg:
            return msg

    return ""


# --------------------
# Main loop
# --------------------
def main():
    tokenizer, model = load_model()

    history = [
        "user1: PogChamp",
        "user2: PogChamp",
        "user3: PogChamp",
    ]

    print("\n=== GENERATED CHAT ===\n")

    uid = 4
    for _ in range(NUM_MESSAGES):
        msg = generate_next_message(history, tokenizer, model)
        if not msg:
            continue

        line = f"user{uid}: {msg}"
        print(line)
        uid += 1

        history.append(line)
        if len(history) > ROLLING_WINDOW:
            history = history[-ROLLING_WINDOW:]


if __name__ == "__main__":
    main()