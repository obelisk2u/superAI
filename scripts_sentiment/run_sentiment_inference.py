#!/usr/bin/env python3
import argparse
from pathlib import Path

import pandas as pd
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm.auto import tqdm

# 3-way sentiment model: negative / neutral / positive
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"


def load_model_and_tokenizer(model_name: str = MODEL_NAME):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def predict_sentiment(
    df: pd.DataFrame,
    text_col: str = "message",
    batch_size: int = 32,
    max_length: int = 128,
    confidence_threshold: float = 0.9,
) -> pd.DataFrame:

    if text_col not in df.columns:
        raise ValueError(f"Input DataFrame must have a '{text_col}' column")

    tokenizer, model = load_model_and_tokenizer()
    id2label = model.config.id2label
    num_labels = len(id2label)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    texts = df[text_col].astype(str).tolist()
    n = len(texts)

    all_labels = []
    all_probs = [[] for _ in range(num_labels)]

    num_batches = (n + batch_size - 1) // batch_size

    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Scoring messages"):
            start = batch_idx * batch_size
            end = min(start + batch_size, n)
            batch_texts = texts[start:end]

            enc = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()

            for row_probs in probs:
                label_idx = int(row_probs.argmax())
                label = id2label[label_idx]
                all_labels.append(label)

                for j in range(num_labels):
                    all_probs[j].append(float(row_probs[j]))

    # Add prediction columns
    df = df.copy()
    df["sentiment_label"] = all_labels

    for j in range(num_labels):
        label_name = id2label[j]
        df[f"sentiment_prob_{label_name}"] = all_probs[j]

    # -------------------------------
    # FILTER BY CONFIDENCE ≥ 0.9
    # -------------------------------
    def get_row_conf(row):
        label = row["sentiment_label"]
        return row[f"sentiment_prob_{label}"]

    df["sentiment_confidence"] = df.apply(get_row_conf, axis=1)

    df = df[df["sentiment_confidence"] >= confidence_threshold]

    df = df.reset_index(drop=True)
    return df


def parse_args():
    p = argparse.ArgumentParser(
        description="Run twitter-roberta-base-sentiment-latest on a cleaned Twitch dataset."
    )
    p.add_argument("input_csv", help="Path to clean_sentiment_dataset.csv")
    p.add_argument("output_csv", help="Path to write filtered CSV")
    p.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    p.add_argument("--max-length", type=int, default=128, help="Max token length (default: 128)")
    p.add_argument("--threshold", type=float, default=0.9, help="Min probability to keep a message")
    return p.parse_args()


def main():
    args = parse_args()

    input_path = Path(args.input_csv)
    output_path = Path(args.output_csv)

    print("Loading CSV from:", input_path)
    df = pd.read_csv(input_path)
    print("Loaded", len(df), "rows")

    df_filtered = predict_sentiment(
        df,
        text_col="message",
        batch_size=args.batch_size,
        max_length=args.max_length,
        confidence_threshold=args.threshold,
    )

    print("After filtering:", len(df_filtered), "rows remain")

    df_filtered.to_csv(output_path, index=False)
    print("Saved filtered sentiment dataset to:", output_path)


if __name__ == "__main__":
    main()