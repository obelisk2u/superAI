#!/usr/bin/env python3
import os
import ssl
import json
import time
import socket
import requests
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

###############################################################################
# CONFIG
###############################################################################

PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TWITCH_SERVER = "irc.chat.twitch.tv"
TWITCH_PORT = 6697

CHANNEL = "supertf"   # without '#'
CHANNEL_TAG = f"#{CHANNEL}"

# IRC login token
IRC_OAUTH = os.getenv("TWITCH_OAUTH_TOKEN")  # may be with or without "oauth:" prefix
IRC_NICK = os.getenv("TWITCH_NICK")

# Helix API credentials (already-issued app access token)
CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
API_ACCESS_TOKEN = os.getenv("TWITCH_API_OAUTH_TOKEN")

RAW_DIR = PROJECT_ROOT / "twitch_data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RECONNECT_DELAY = 10

# normalize IRC token
if IRC_OAUTH and not IRC_OAUTH.startswith("oauth:"):
    IRC_OAUTH = f"oauth:{IRC_OAUTH}"

###############################################################################
# OUTPUT FILE
###############################################################################

def current_log_path():
    date = datetime.utcnow().strftime("%Y-%m-%d")
    return RAW_DIR / f"{CHANNEL}_{date}.jsonl"


def write_jsonl(obj: dict):
    path = current_log_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

###############################################################################
# HELIX STREAM INFO
###############################################################################

_stream_cache = None
_stream_cache_expire = 0
STREAM_REFRESH_SECONDS = 30


def get_stream_info():
    """
    Returns:
      dict or None if offline:
      {
        "category_name": str or None,
        "category_id": str or None,
        "title": str or None,
        "viewer_count": int or None,
        "is_mature": bool or None
      }

    Uses CLIENT_ID + API_ACCESS_TOKEN directly (no refresh),
    and caches for STREAM_REFRESH_SECONDS to avoid spam.
    """
    global _stream_cache, _stream_cache_expire

    now = time.time()
    if _stream_cache and now < _stream_cache_expire:
        return _stream_cache

    if not CLIENT_ID or not API_ACCESS_TOKEN:
        print("[Helix WARN] Missing TWITCH_CLIENT_ID or TWITCH_API_OAUTH_TOKEN; no stream metadata.")
        return None

    headers = {
        "Client-Id": CLIENT_ID,
        "Authorization": f"Bearer {API_ACCESS_TOKEN}",
    }
    params = {"user_login": CHANNEL}

    try:
        r = requests.get(
            "https://api.twitch.tv/helix/streams",
            headers=headers,
            params=params,
            timeout=10,
        )
        r.raise_for_status()
        data = r.json().get("data", [])

        if not data:
            _stream_cache = None
        else:
            s = data[0]
            _stream_cache = {
                "category_name": s.get("game_name"),
                "category_id": s.get("game_id"),
                "title": s.get("title"),
                "viewer_count": s.get("viewer_count"),
                "is_mature": s.get("is_mature", False),
            }

    except Exception as e:
        print(f"[Helix WARN] Failed to fetch stream info: {e}")
        # keep old cache if any

    _stream_cache_expire = now + STREAM_REFRESH_SECONDS
    return _stream_cache

###############################################################################
# IRC (chat)
###############################################################################

def connect_irc():
    if not IRC_OAUTH or not IRC_NICK:
        raise RuntimeError("Missing TWITCH_OAUTH_TOKEN or TWITCH_NICK")

    print(f"[IRC] Connecting as {IRC_NICK}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    context = ssl.create_default_context()
    ssl_sock = context.wrap_socket(sock, server_hostname=TWITCH_SERVER)
    ssl_sock.connect((TWITCH_SERVER, TWITCH_PORT))

    ssl_sock.send(f"PASS {IRC_OAUTH}\r\n".encode("utf-8"))
    ssl_sock.send(f"NICK {IRC_NICK}\r\n".encode("utf-8"))
    ssl_sock.send(f"JOIN #{CHANNEL}\r\n".encode("utf-8"))

    # enable metadata tags
    ssl_sock.send("CAP REQ :twitch.tv/tags\r\n".encode("utf-8"))

    print(f"[IRC] Joined #{CHANNEL}")
    return ssl_sock


def parse_privmsg(raw: str):
    try:
        prefix_split = raw.split(" ", 3)
        if len(prefix_split) < 4:
            return None

        tags_part, _, cmd, rest = prefix_split
        if cmd != "PRIVMSG":
            return None

        # Parse tags
        tags = {}
        tags_str = tags_part.lstrip("@")
        for kv in tags_str.split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                tags[k] = v

        # Split channel and message
        chan_and_msg = rest.split(" :", 1)
        if len(chan_and_msg) != 2:
            return None

        message = chan_and_msg[1].rstrip("\r\n")
        username = tags.get("display-name") or tags.get("login") or "unknown"

        return {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "username": username,
            "message": message,
        }

    except Exception:
        return None

###############################################################################
# MAIN CHAT LOOP
###############################################################################

def run_loop():
    last_category = None

    while True:
        try:
            sock = connect_irc()
            buffer = ""

            while True:
                chunk = sock.recv(2048).decode("utf-8", errors="ignore")
                if not chunk:
                    print("[IRC] Connection dropped. Reconnecting...")
                    break

                buffer += chunk
                *lines, buffer = buffer.split("\r\n")

                for line in lines:
                    if not line:
                        continue

                    # Uncomment if you ever want to see all raw lines:
                    # if not line.startswith("PING") and "PRIVMSG" not in line:
                    #     print("[IRC RAW]", repr(line))

                    if line.startswith("PING"):
                        sock.send(line.replace("PING", "PONG").encode("utf-8") + b"\r\n")
                        continue

                    if "PRIVMSG" in line:
                        base = parse_privmsg(line)
                        if not base:
                            continue

                        # Add stream metadata
                        info = get_stream_info()

                        if info:
                            base.update(info)

                            # detect category change
                            cat = info.get("category_name")
                            if cat != last_category:
                                print(f"[Stream] Category changed: {last_category} → {cat}")
                                last_category = cat
                        else:
                            base.update(
                                {
                                    "category_name": None,
                                    "category_id": None,
                                    "title": None,
                                    "viewer_count": None,
                                    "is_mature": None,
                                }
                            )

                        write_jsonl(base)
                        print(f"[{base['timestamp_utc']}] {base['username']}: {base['message']}")

        except Exception as e:
            print(f"[ERROR] {e} — retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)
            continue

###############################################################################
# ENTRYPOINT
###############################################################################

if __name__ == "__main__":
    print(f"Logging Twitch chat + metadata for channel '{CHANNEL}'")
    print(f"Saving to {RAW_DIR}")
    run_loop()