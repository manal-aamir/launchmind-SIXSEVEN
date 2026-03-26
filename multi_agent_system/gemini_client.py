"""
Google Gemini client — optional fallback when Groq / DeepSeek fail.

Uses the new `google-genai` SDK.
Set GEMINI_API_KEY in .env (never commit keys).
Optional: GEMINI_MODEL (default gemini-2.0-flash).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from multi_agent_system.prompts import compose_system_prompt

try:
    from google import genai
    from google.genai import types
except ImportError:
    genai = None  # type: ignore
    types = None  # type: ignore


class GeminiClient:
    def __init__(self, api_key: str, model: str = "gemini-2.0-flash") -> None:
        self.model_name = model
        self.enabled = bool(api_key) and genai is not None and types is not None
        self._client: Optional[Any] = None
        if self.enabled:
            self._client = genai.Client(api_key=api_key)

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
        combined = (
            f"{system_prompt}\n\n---\n\n{user_prompt}\n\n"
            "Return ONLY a valid JSON object. No markdown fences. No extra text."
        )
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=combined,
                config=types.GenerateContentConfig(
                    temperature=0.2,
                    response_mime_type="application/json",
                ),
            )
            text = ((getattr(response, "text", None) or "")).strip()
            if text:
                return json.loads(text)
        except Exception:
            pass
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=combined,
                config=types.GenerateContentConfig(temperature=0.0),
            )
            text = ((getattr(response, "text", None) or "")).strip()
            return self._extract_json(text)
        except Exception:
            return mock_default

    def complete_text(self, system_prompt: str, user_prompt: str, temperature: float = 0.3) -> str:
        """Raw text (e.g. Product agent JSON — caller parses)."""
        if not self.enabled or not self._client:
            return ""
        combined = f"{system_prompt}\n\n{user_prompt}"
        try:
            response = self._client.models.generate_content(
                model=self.model_name,
                contents=combined,
                config=types.GenerateContentConfig(temperature=temperature),
            )
            return ((getattr(response, "text", None) or "")).strip()
        except Exception:
            return ""
