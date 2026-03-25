"""Shared models for agent communication.

Every inter-agent message follows the assignment-specified JSON schema:
  message_id, from_agent, to_agent, message_type, payload,
  timestamp, parent_message_id
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Assignment-specified message schema
# ---------------------------------------------------------------------------

def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def make_message(
    from_agent: str,
    to_agent: str,
    message_type: str,           # 'task' | 'result' | 'revision_request' | 'confirmation'
    payload: Dict[str, Any],
    parent_message_id: Optional[str] = None,
) -> Dict[str, Any]:
    """Build a fully-compliant inter-agent message dict."""
    return {
        "message_id":        f"msg-{uuid.uuid4().hex[:8]}",
        "from_agent":        from_agent,
        "to_agent":          to_agent,
        "message_type":      message_type,
        "payload":           payload,
        "timestamp":         _now_iso(),
        "parent_message_id": parent_message_id,
    }


# ---------------------------------------------------------------------------
# Message history log (in-memory + persisted to JSON)
# ---------------------------------------------------------------------------

class MessageBus:
    """Logs every message so the evaluator can inspect the full CEO history."""

    def __init__(self) -> None:
        self._messages: List[Dict[str, Any]] = []

    def send(
        self,
        from_agent: str,
        to_agent: str,
        message_type: str,
        payload: Dict[str, Any],
        parent_message_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        msg = make_message(from_agent, to_agent, message_type, payload, parent_message_id)
        self._messages.append(msg)
        return msg

    def all_messages(self) -> List[Dict[str, Any]]:
        return list(self._messages)

    def ceo_messages(self) -> List[Dict[str, Any]]:
        """Every message the CEO sent OR received."""
        return [
            m for m in self._messages
            if m["from_agent"] == "ceo" or m["to_agent"] == "ceo"
        ]

    def save(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = output_dir / f"message_log_{ts}.json"
        path.write_text(
            json.dumps(self._messages, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        return path


# ---------------------------------------------------------------------------
# Internal dataclasses (unchanged API surface for other modules)
# ---------------------------------------------------------------------------

@dataclass
class TaskMessage:
    task_id: str
    target_agent: str
    startup_idea: str
    task_brief: str
    expected_output: List[str]
    constraints: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AgentResult:
    agent_name: str
    task_id: str
    output: Dict[str, Any]
    revision_round: int = 0


@dataclass
class ReviewDecision:
    acceptable: bool
    score: int
    rationale: str
    follow_up_instruction: str


@dataclass
class DecisionLogEntry:
    stage: str
    detail: str
    data: Dict[str, Any] = field(default_factory=dict)
