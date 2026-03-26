"""
DeepSeek client (OpenAI-compatible) used as an optional fallback when Groq
is rate-limited.

IMPORTANT: The API key must come from environment/config (.env). Never hardcode.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from multi_agent_system.prompts import compose_system_prompt


class DeepSeekClient:
    def __init__(self, api_key: str, model: str = "deepseek-chat", base_url: str = "https://api.deepseek.com") -> None:
        self.model = model
        self.enabled = bool(api_key)
        self._client: Optional[OpenAI] = OpenAI(api_key=api_key, base_url=base_url) if self.enabled else None

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = (text or "").strip()
        if "```" in text:
            lines = [l for l in text.splitlines() if not l.strip().startswith("```")]
            text = "\n".join(lines).strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model response.")
        return json.loads(text[start : end + 1])

    def complete_json(self, role_prompt: str, user_prompt: str, mock_default: Dict[str, Any]) -> Dict[str, Any]:
        if not self.enabled or not self._client:
            return mock_default

        system_prompt = compose_system_prompt(role_prompt)
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = (resp.choices[0].message.content or "").strip()
        return self._extract_json(content)

    def complete_text(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.3,
    ) -> str:
        """Raw text completion (e.g. Product Agent JSON — caller parses)."""
        if not self.enabled or not self._client:
            return ""
        resp = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()

