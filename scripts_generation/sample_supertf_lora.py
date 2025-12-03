#!/usr/bin/env python3
import os
import re
import json
import random
import sys
from typing import List, Dict, Any
from pathlib import Path

import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
)
from peft import PeftModel

# -------------------------------------------------
# Make repo root importable *before* local imports
# -------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from scripts_timing.timing_model import load_timing_model, simulate_timing, ChatEvent

# --------------------
# Config
# --------------------
MODEL_ID = "microsoft/Phi-3-mini-4k-instruct"

# if you want PROJECT_ROOT as a string for os.path.join:
PROJECT_ROOT_STR = str(PROJECT_ROOT)

ADAPTER_DIR = os.path.join(PROJECT_ROOT_STR, "phi3_supertf_lora")

ROLLING_WINDOW = 25

NUM_MESSAGES = 100
NUM_CHATS = 50

MAX_MSG_CHARS = 140
MAX_MENTIONS = 2
MAX_RETRIES = 6

CHAT_LOGS_PATH = os.path.join(PROJECT_ROOT_STR, "chat_logs.json")

TIMING_MODEL_PATH = PROJECT_ROOT / "twitch_data" / "processed" / "timing_model.json"
SYNTH_STREAM_DURATION_SEC = 3600.0  # 1h synthetic stream

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

# [cleaning functions unchanged]
# ...

def seconds_to_mmss(t: float) -> str:
    t_int = int(t)
    m = t_int // 60
    s = t_int % 60
    return f"{m:02d}:{s:02d}"


def generate_single_chat(
    chat_index: int,
    tokenizer,
    model,
    timing_model: Dict[str, Any],
    messages_per_chat: int = NUM_MESSAGES,
) -> Dict[str, Any]:
    """
    Generate a single chat log using:
      - Phi-3 for content
      - timing_model for timestamps
    """

    # seed history
    history: List[str] = [
        "user1: PogChamp",
        "user2: PogChamp",
        "user3: PogChamp",
    ]

    messages: List[Dict[str, Any]] = []

    # get synthetic event times
    events: List[ChatEvent] = simulate_timing(
        total_duration_sec=SYNTH_STREAM_DURATION_SEC,
        model=timing_model,
        use_empirical_gaps=True,
        use_empirical_batch=True,
    )

    if len(events) < messages_per_chat:
        # if you want to be strict you can re-simulate with a longer duration
        events = events[:messages_per_chat]
    else:
        events = events[:messages_per_chat]

    next_uid = 4

    for i, ev in enumerate(events):
        msg_text = generate_next_message(history, tokenizer, model)
        if not msg_text:
            # if we fail too many times, just skip this event
            continue

        author = f"user{next_uid}"
        color = random.choice(USERNAME_COLORS)
        timestamp_str = seconds_to_mmss(ev.timestamp)

        message_obj = {
            "id": str(len(messages) + 1),
            "author": author,
            "color": color,
            "text": msg_text,
            "timestamp": timestamp_str,
        }
        messages.append(message_obj)

        history.append(f"{author}: {msg_text}")
        if len(history) > ROLLING_WINDOW:
            history = history[-ROLLING_WINDOW:]

        next_uid += 1

        if len(messages) >= messages_per_chat:
            break

    chat_log = {
        "id": f"log_{chat_index:04d}",
        "title": f"Generated Chat #{chat_index}",
        "messages": messages,
    }

    return chat_log


def main():
    tokenizer, model = load_model()

    # load timing model once
    timing_model = load_timing_model(TIMING_MODEL_PATH)

    all_logs: List[Dict[str, Any]] = []

    for i in range(1, NUM_CHATS + 1):
        print(f"\n=== Generating chat {i}/{NUM_CHATS} ===")
        chat_log = generate_single_chat(
            i,
            tokenizer,
            model,
            timing_model=timing_model,
            messages_per_chat=NUM_MESSAGES,
        )
        all_logs.append(chat_log)

    with open(CHAT_LOGS_PATH, "w", encoding="utf-8") as f:
        json.dump(all_logs, f, ensure_ascii=False, indent=2)

    print(f"\nSaved {len(all_logs)} chats to {CHAT_LOGS_PATH}")


if __name__ == "__main__":
    main()