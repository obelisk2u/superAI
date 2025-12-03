import os
import json
from glob import glob

PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))

RAW_DIR = os.path.join(PROJECT_ROOT, "twitch_data", "raw")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "twitch_data", "processed")
os.makedirs(PROCESSED_DIR, exist_ok=True)

CLEAN_JSONL_PATH = os.path.join(PROCESSED_DIR, "supertf_clean.jsonl")
CLEAN_TXT_PATH = os.path.join(PROCESSED_DIR, "supertf_all.txt")


def iter_raw_lines():
    pattern = os.path.join(RAW_DIR, "supertf_*.jsonl")
    files = sorted(glob(pattern))
    if not files:
        raise RuntimeError(f"No raw files found matching {pattern}")

    print("Found raw files:")
    for f in files:
        print("  -", f)

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield line


def clean_record(raw_line: str):
    try:
        data = json.loads(raw_line)
    except json.JSONDecodeError:
        return None

    msg = (data.get("message") or "").strip()
    username = (data.get("username") or "").strip()
    ts = data.get("timestamp_utc")

    if not msg or not username:
        return None

    return {
        "timestamp_utc": ts,
        "username": username,
        "message": msg,
    }


def main():
    num_raw = 0
    num_kept = 0

    with open(CLEAN_JSONL_PATH, "w", encoding="utf-8") as jsonl_out, \
         open(CLEAN_TXT_PATH, "w", encoding="utf-8") as txt_out:

        for line in iter_raw_lines():
            num_raw += 1
            rec = clean_record(line)
            if rec is None:
                continue

            num_kept += 1
            jsonl_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            txt_out.write(f"{rec['username']}: {rec['message']}\n")

    print(f"Processed {num_raw} raw lines, kept {num_kept}.")
    print(f"Wrote clean JSONL to: {CLEAN_JSONL_PATH}")
    print(f"Wrote flat text to:   {CLEAN_TXT_PATH}")


if __name__ == "__main__":
    main()