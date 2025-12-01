import os
import ssl
import json
import socket
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[1]

load_dotenv(PROJECT_ROOT / ".env")

TWITCH_SERVER = "irc.chat.twitch.tv"
TWITCH_PORT = 6697  # TLS
CHANNEL = "#supertf"

OAUTH_TOKEN = os.environ.get("TWITCH_OAUTH_TOKEN")
TWITCH_NICK = os.environ.get("TWITCH_NICK")

if OAUTH_TOKEN and not OAUTH_TOKEN.startswith("oauth:"):
    OAUTH_TOKEN = f"oauth:{OAUTH_TOKEN}"

RAW_DIR = PROJECT_ROOT / "twitch_data" / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

RECONNECT_DELAY_SECONDS = 10


def current_log_path():
    date_str = datetime.utcnow().strftime("%Y-%m-%d")
    filename = f"supertf_{date_str}.jsonl"
    return RAW_DIR / filename


def log_line(data: dict):
    path = current_log_path()
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(data, ensure_ascii=False) + "\n")


def connect():
    if not OAUTH_TOKEN or not TWITCH_NICK:
        raise RuntimeError(
            "Missing TWITCH_OAUTH_TOKEN or TWITCH_NICK (check your .env file)."
        )

    print(f"Connecting to {TWITCH_SERVER}:{TWITCH_PORT} as {TWITCH_NICK} ...")
    sock = socket.socket()
    ssl_sock = ssl.wrap_socket(sock)

    ssl_sock.connect((TWITCH_SERVER, TWITCH_PORT))

    ssl_sock.send(f"PASS {OAUTH_TOKEN}\r\n".encode("utf-8"))
    ssl_sock.send(f"NICK {TWITCH_NICK}\r\n".encode("utf-8"))
    ssl_sock.send(f"JOIN {CHANNEL}\r\n".encode("utf-8"))

    ssl_sock.send("CAP REQ :twitch.tv/tags\r\n".encode("utf-8"))

    print(f"Joined {CHANNEL}")
    return ssl_sock


def parse_privmsg(raw_line: str):
    try:
        prefix_split = raw_line.split(" ", 3)
        if len(prefix_split) < 4:
            return None

        tags_part, _, cmd, rest = prefix_split

        if cmd != "PRIVMSG":
            return None

        tags_str = tags_part.lstrip("@")
        tags = {}
        for item in tags_str.split(";"):
            if "=" in item:
                k, v = item.split("=", 1)
                tags[k] = v

        chan_and_msg = rest.split(" :", 1)
        if len(chan_and_msg) != 2:
            return None

        channel = chan_and_msg[0].strip()
        message = chan_and_msg[1].rstrip("\r\n")

        username = tags.get("display-name") or tags.get("login") or "unknown"

        return {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "username": username,
            "message": message,
            "channel": channel,
            "raw": raw_line.rstrip("\r\n"),
        }
    except Exception:
        return None


def run_loop():
    while True:
        try:
            sock = connect()
            buffer = ""

            while True:
                data = sock.recv(2048).decode("utf-8", errors="ignore")
                if not data:
                    print("Connection closed by server, reconnecting...")
                    break

                buffer += data
                *lines, buffer = buffer.split("\r\n")

                for line in lines:
                    if not line:
                        continue

                    if line.startswith("PING"):
                        pong_reply = line.replace("PING", "PONG")
                        sock.send((pong_reply + "\r\n").encode("utf-8"))
                        continue

                    if "PRIVMSG" in line:
                        msg = parse_privmsg(line)
                        if msg:
                            log_line(msg)
                            print(f"[{msg['timestamp_utc']}] {msg['username']}: {msg['message']}")

        except Exception as e:
            print(f"[ERROR] {e}. Reconnecting in {RECONNECT_DELAY_SECONDS}s...")
            time.sleep(RECONNECT_DELAY_SECONDS)
            continue


if __name__ == "__main__":
    print(f"Logging Twitch chat from {CHANNEL} into {RAW_DIR}")
    run_loop()