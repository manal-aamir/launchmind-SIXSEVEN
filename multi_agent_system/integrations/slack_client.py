"""Slack integration for final CEO summary."""

import json
from typing import Dict
from urllib import parse, request


class SlackClient:
    def __init__(self, bot_token: str) -> None:
        self.bot_token = bot_token

    def post_message(self, channel: str, text: str) -> Dict:
        data = parse.urlencode({"channel": channel, "text": text}).encode("utf-8")
        req = request.Request(
            "https://slack.com/api/chat.postMessage",
            data=data,
            headers={
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload

    def post_block_message(self, channel: str, text: str, blocks: list) -> Dict:
        body = json.dumps({"channel": channel, "text": text, "blocks": blocks}).encode("utf-8")
        req = request.Request(
            "https://slack.com/api/chat.postMessage",
            data=body,
            headers={
                "Authorization": f"Bearer {self.bot_token}",
                "Content-Type": "application/json; charset=utf-8",
            },
            method="POST",
        )
        with request.urlopen(req, timeout=20) as resp:
            payload = json.loads(resp.read().decode("utf-8"))
        return payload

