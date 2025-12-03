#!/usr/bin/env python3
import os
import re
import json
import random
from typing import List, Dict, Any

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

# messages per chat
NUM_MESSAGES = 100

# how many chats to generate
NUM_CHATS = 50

MAX_MSG_CHARS = 140
MAX_MENTIONS = 2
MAX_RETRIES = 6

# where to save the frontend-ready logs
CHAT_LOGS_PATH = os.path.join(PROJECT_ROOT, "chat_logs.json")

# simple Twitch-y color palette for usernames
USERNAME_COLORS = [
    "#FF4F4F", "#33B9FF", "#A970FF", "#00FF7F", "#FFD93D",
    "#FF61A6", "#F97316", "#22C55E", "#6366F1", "#EC4899",
]


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
def generate_next_message(history: List[str], tokenizer, model) -> str:
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


def seconds_to_mmss(t: int) -> str:
    m = t // 60
    s = t % 60
    return f"{m:02d}:{s:02d}"


def generate_single_chat(
    chat_index: int,
    tokenizer,
    model,
    messages_per_chat: int = NUM_MESSAGES,
) -> Dict[str, Any]:
    """
    Generate a single chat log in the format:

    {
      "id": "log_0001",
      "title": "Generated Chat #1",
      "messages": [
        {"id": "1", "author": "user4", "color": "#ff4f4f", "text": "...", "timestamp": "00:01"},
        ...
      ]
    }
    """

    # Seed history with some starter spam so it feels like a running chat
    history: List[str] = [
        "user1: PogChamp",
        "user2: PogChamp",
        "user3: PogChamp",
    ]

    messages: List[Dict[str, Any]] = []

    # we start generating from user4 onward, like the original script
    next_uid = 4
    current_time_s = 0

    while len(messages) < messages_per_chat:
        msg_text = generate_next_message(history, tokenizer, model)
        if not msg_text:
            # skip garbage; try again without incrementing message count
            continue

        author = f"user{next_uid}"
        color = random.choice(USERNAME_COLORS)
        timestamp = seconds_to_mmss(current_time_s)

        message_obj = {
            "id": str(len(messages) + 1),
            "author": author,
            "color": color,
            "text": msg_text,
            "timestamp": timestamp,
        }
        messages.append(message_obj)

        # update rolling history for next generation
        history.append(f"{author}: {msg_text}")
        if len(history) > ROLLING_WINDOW:
            history = history[-ROLLING_WINDOW:]

        next_uid += 1
        # simple time progression: 1–5 seconds between messages
        current_time_s += random.randint(1, 5)

    chat_log = {
        "id": f"log_{chat_index:04d}",
        "title": f"Generated Chat #{chat_index}",
        "messages": messages,
    }

    return chat_log


# --------------------
# Main loop
# --------------------
def main():
    tokenizer, model = load_model()

    all_logs: List[Dict[str, Any]] = []

    for i in range(1, NUM_CHATS + 1):
        print(f"\n=== Generating chat {i}/{NUM_CHATS} ===")
        chat_log = generate_single_chat(i, tokenizer, model, messages_per_chat=NUM_MESSAGES)
        all_logs.append(chat_log)

    # Save to JSON in frontend-friendly format
    with open(CHAT_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_logs)} chats to {CHAT_LOGS_PATH}")


if __name__ == "__main__":
    main()