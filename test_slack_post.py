#!/usr/bin/env python3
"""
Quick Slack API test script.
Reads SLACK_BOT_TOKEN from .env in the same folder and sends chat.postMessage.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from urllib import parse, request


def load_env_var(env_path: Path, key: str) -> str:
    if not env_path.exists():
        raise FileNotFoundError(f".env file not found at: {env_path}")

    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        name, value = line.split("=", 1)
        if name.strip() == key:
            cleaned = value.strip().strip("'").strip('"')
            if cleaned:
                return cleaned
            break
    raise ValueError(f"{key} not found in {env_path}")


def slack_api(method: str, token: str, payload: dict) -> dict:
    data = parse.urlencode(payload).encode("utf-8")
    req = request.Request(
        f"https://slack.com/api/{method}",
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with request.urlopen(req, timeout=20) as resp:
        return json.loads(resp.read().decode("utf-8"))


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    token = load_env_var(project_dir / ".env", "SLACK_BOT_TOKEN")

    auth = slack_api("auth.test", token, {})
    if not auth.get("ok"):
        raise RuntimeError(f"auth.test failed: {auth.get('error')}")

    # Send to the app/home DM using bot user id.
    channel = auth["user_id"]
    timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    message = f"Quick bot test via chat.postMessage at {timestamp}"

    post = slack_api("chat.postMessage", token, {"channel": channel, "text": message})
    if not post.get("ok"):
        raise RuntimeError(f"chat.postMessage failed: {post.get('error')}")

    print("Message posted successfully.")
    print("workspace:", auth.get("team"))
    print("channel:", post.get("channel"))
    print("message_ts:", post.get("ts"))


if __name__ == "__main__":
    main()
