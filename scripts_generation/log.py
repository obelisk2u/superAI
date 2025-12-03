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
 
PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(PROJECT_ROOT / ".env")

TWITCH_SERVER = "irc.chat.twitch.tv"
TWITCH_PORT = 6697

CHANNELS = ["supertf", "pge4", "emongg", "xQc", "aspen", "mL7support", "loltyler1", "erobb221"]
CHANNELS = [c.lower() for c in CHANNELS]

CHANNELS_JOIN = ",".join(f"#{c}" for c in CHANNELS)

IRC_OAUTH = os.getenv("TWITCH_OAUTH_TOKEN")
IRC_NICK = os.getenv("TWITCH_NICK")

CLIENT_ID = os.getenv("TWITCH_CLIENT_ID")
API_ACCESS_TOKEN = os.getenv("TWITCH_API_OAUTH_TOKEN")

RAW_DIR = PROJECT_ROOT / "twitch_data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RECONNECT_DELAY = 10
STREAM_REFRESH_SECONDS = 20   


if IRC_OAUTH and not IRC_OAUTH.startswith("oauth:"):
    IRC_OAUTH = f"oauth:{IRC_OAUTH}"
 
is_live = {c: False for c in CHANNELS}
current_stream_id = {c: None for c in CHANNELS}
current_stream_path = {c: None for c in CHANNELS}
 
_stream_cache = {}
_stream_cache_expire = {}
 
def start_new_stream_file(channel: str, started_at_iso: str): 
    safe_ts = started_at_iso.replace(":", "-")
    folder = RAW_DIR / channel
    folder.mkdir(parents=True, exist_ok=True)

    filepath = folder / f"{channel}_stream_{safe_ts}.jsonl"

    current_stream_id[channel] = started_at_iso
    current_stream_path[channel] = filepath

    print(f"[Stream {channel}] Starting new stream log → {filepath}")

    return filepath


def write_stream_jsonl(channel: str, obj: dict): 
    path = current_stream_path[channel]
    if not path:
        return   

    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

 
def get_stream_info(channel: str): 
    channel = channel.lower()
    now = time.time()
 
    if (
        channel in _stream_cache
        and channel in _stream_cache_expire
        and now < _stream_cache_expire[channel]
    ):
        return _stream_cache[channel]

    headers = {
        "Client-Id": CLIENT_ID,
        "Authorization": f"Bearer {API_ACCESS_TOKEN}",
    }
    params = {"user_login": channel}

    try:
        r = requests.get(
            "https://api.twitch.tv/helix/streams",
            headers=headers,
            params=params,
            timeout=10
        )
        r.raise_for_status()
        data = r.json().get("data", [])

        if not data:
            _stream_cache[channel] = None
        else:
            s = data[0]
            _stream_cache[channel] = {
                "category_name": s.get("game_name"),
                "category_id": s.get("game_id"),
                "title": s.get("title"),
                "viewer_count": s.get("viewer_count"),
                "started_at": s.get("started_at"), 
                "is_mature": s.get("is_mature", False),
            }

    except Exception as e:
        print(f"[Helix WARN] {channel}: {e}")

    _stream_cache_expire[channel] = now + STREAM_REFRESH_SECONDS
    return _stream_cache[channel]

 
def connect_irc():
    if not IRC_OAUTH or not IRC_NICK:
        raise RuntimeError("Missing Twitch IRC credentials.")

    print(f"[IRC] Connecting as {IRC_NICK}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    context = ssl.create_default_context()
    ssl_sock = context.wrap_socket(sock, server_hostname=TWITCH_SERVER)
    ssl_sock.connect((TWITCH_SERVER, TWITCH_PORT))

    ssl_sock.send(f"PASS {IRC_OAUTH}\r\n".encode("utf-8"))
    ssl_sock.send(f"NICK {IRC_NICK}\r\n".encode("utf-8"))
    ssl_sock.send(f"JOIN {CHANNELS_JOIN}\r\n".encode("utf-8"))
    ssl_sock.send(b"CAP REQ :twitch.tv/tags\r\n")

    print(f"[IRC] Joined: {', '.join('#' + c for c in CHANNELS)}")
    return ssl_sock
 
def parse_privmsg(raw: str):
    try:
        prefix_split = raw.split(" ", 3)
        if len(prefix_split) < 4:
            return None

        tags_part, _, cmd, rest = prefix_split
        if cmd != "PRIVMSG":
            return None
 
        tags = {}
        for kv in tags_part.lstrip("@").split(";"):
            if "=" in kv:
                k, v = kv.split("=", 1)
                tags[k] = v
 
        chan_and_msg = rest.split(" :", 1)
        if len(chan_and_msg) != 2:
            return None

        channel_raw = chan_and_msg[0].strip()
        message = chan_and_msg[1].rstrip("\r\n")

        if channel_raw.startswith("#"):
            channel = channel_raw[1:].lower()
        else:
            channel = channel_raw.lower()

        return {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "username": tags.get("display-name") or tags.get("login"),
            "message": message,
            "channel": channel
        }

    except Exception:
        return None
 

def run_loop():
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

                    if line.startswith("PING"):
                        sock.send(line.replace("PING", "PONG").encode("utf-8") + b"\r\n")
                        continue

                    if "PRIVMSG" in line:
                        base = parse_privmsg(line)
                        if not base:
                            continue

                        chan = base["channel"]
 
                        info = get_stream_info(chan)
 
                        if info is None:
                            is_live[chan] = False
                            current_stream_id[chan] = None
                            continue
 
                        if not is_live[chan]:
                            is_live[chan] = True
                            started_at = info["started_at"]  
                            start_new_stream_file(chan, started_at)
 
                        record = {
                            "timestamp_utc": base["timestamp_utc"],
                            "username": base["username"],
                            "message": base["message"],
                            "category_name": info["category_name"],
                            "viewer_count": info["viewer_count"],
                            "channel": chan,
                            "stream_id": current_stream_id[chan],
                        }

                        write_stream_jsonl(chan, record)

                        print(
                            f"[{record['timestamp_utc']}] #{chan} "
                            f"({record['category_name']} | {record['viewer_count']}): "
                            f"{record['username']}: {record['message']}"
                        )

        except Exception as e:
            print(f"[ERROR] {e} — retrying in {RECONNECT_DELAY}s...")
            time.sleep(RECONNECT_DELAY)
            continue

if __name__ == "__main__":
    print("Monitoring channels:")
    for c in CHANNELS:
        print(f" - #{c}")
    print(f"Logs saved under: {RAW_DIR}")
    run_loop()