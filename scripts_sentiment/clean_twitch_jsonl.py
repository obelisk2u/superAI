#!/usr/bin/env python3
import argparse
import json
import re
from glob import glob
from pathlib import Path

import pandas as pd

EMOTE_SET = set()

USE_HEURISTIC_EMOTE_STRIP = True  # strip ALLCAPS “emote-like” tokens

# Emoji ranges
EMOJI_PATTERN = re.compile(
    "["
    "\U0001F300-\U0001F5FF"
    "\U0001F600-\U0001F64F"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "\u2600-\u26FF"
    "\u2700-\u27BF"
    "]+"
)


def strip_emotes(text: str) -> str:
    """Remove emoji + emote-like tokens from a message string."""
    if not isinstance(text, str):
        return ""

    text = EMOJI_PATTERN.sub(" ", text)

    tokens = text.split()
    cleaned_tokens = []

    for tok in tokens:
        if tok in EMOTE_SET:
            continue

        if USE_HEURISTIC_EMOTE_STRIP:
            if tok.isupper() and re.search("[A-Z]", tok) and 3 <= len(tok) <= 20:
                continue

        cleaned_tokens.append(tok)

    cleaned = " ".join(cleaned_tokens)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def load_jsonl_files(input_dir: Path) -> pd.DataFrame:
    files = sorted(glob(str(input_dir / "supertf_*.jsonl")))
    records = []

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if "username" not in obj or "message" not in obj:
                    continue

                records.append(
                    {
                        "timestamp_utc": obj.get("timestamp_utc"),
                        "username": obj.get("username"),
                        "message_raw": obj.get("message"),
                        "channel": obj.get("channel"),
                        "raw": obj.get("raw"),
                    }
                )

    if not records:
        raise RuntimeError(f"No valid messages found in {input_dir}")

    return pd.DataFrame(records)


def build_clean_dataset(df: pd.DataFrame, min_len: int = 20, min_user_msgs: int = 20) -> pd.DataFrame:
    df = df.copy()
 
    df["message_clean"] = df["message_raw"].astype(str).apply(strip_emotes)
 
    df["message_norm"] = (
        df["message_clean"]
        .str.lower()
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
 
    df = df[df["message_norm"].str.len() >= min_len]
    df = df[df["message_norm"] != ""]
 
    user_counts = df["username"].value_counts()
    allowed_users = user_counts[user_counts >= min_user_msgs].index
    df = df[df["username"].isin(allowed_users)]
 
    msg_user_counts = (
        df.groupby("message_norm")["username"]
        .nunique()
        .rename("unique_users")
    )
    cross_spam_messages = msg_user_counts[msg_user_counts > 1].index
    df["is_cross_spam"] = df["message_norm"].isin(cross_spam_messages)
 
    self_counts = (
        df.groupby(["username", "message_norm"])
        .size()
        .rename("self_count")
    )
    self_spam_pairs = self_counts[self_counts > 1].reset_index()[["username", "message_norm"]]
    self_spam_pairs["is_self_spam"] = True

    df = df.merge(
        self_spam_pairs,
        on=["username", "message_norm"],
        how="left",
    )
    df["is_self_spam"] = df["is_self_spam"].fillna(False)
 
    df = df[~df["is_cross_spam"] & ~df["is_self_spam"]]
 
    df = df.rename(columns={"message_clean": "message"})
    keep_cols = ["timestamp_utc", "username", "message", "channel"]
    df = df[keep_cols].reset_index(drop=True)

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Clean Twitch JSONL logs for sentiment analysis.")
    p.add_argument("input_dir", help="Directory containing supertf_*.jsonl files")
    p.add_argument("output_csv", help="Path to write cleaned CSV")
    p.add_argument(
        "--min-len",
        type=int,
        default=20,
        help="Minimum message length after cleaning (default: 20)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)

    df = load_jsonl_files(input_dir)
    clean_df = build_clean_dataset(df, min_len=args.min_len)
    clean_df.to_csv(args.output_csv, index=False)
    print(f"Loaded {len(df)} raw messages")
    print(f"Saved {len(clean_df)} cleaned messages to {args.output_csv}")


if __name__ == "__main__":
    main()