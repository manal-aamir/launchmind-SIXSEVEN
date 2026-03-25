"""Base class for domain agents (Groq-powered)."""

import json
from typing import Any, Dict

from multi_agent_system.groq_client import GroqClient
from multi_agent_system.models import AgentResult, TaskMessage


class BaseAgent:
    agent_name = "base"

    def __init__(self, groq_client: GroqClient) -> None:
        self.groq = groq_client

    def build_role_prompt(self) -> str:
        raise NotImplementedError

    def mock_output(self, task: TaskMessage) -> Dict[str, Any]:
        return {"status": "mock", "task_brief": task.task_brief}

    def run(self, task: TaskMessage, revision_instruction: str = "") -> AgentResult:
        if not self.groq.enabled:
            return AgentResult(
                agent_name=self.agent_name,
                task_id=task.task_id,
                output=self.mock_output(task),
                revision_round=1 if revision_instruction else 0,
            )

        prompt = (
            f"Startup idea: {task.startup_idea}\n"
            f"Task brief: {task.task_brief}\n"
            f"Expected output: {task.expected_output}\n"
            f"Constraints: {task.constraints}\n"
            f"Context: {task.context}\n"
        )
        if revision_instruction:
            prompt += f"\nRevision instruction from CEO: {revision_instruction}\n"

        result = self.groq._complete_json(
            role_prompt=self.build_role_prompt(),
            user_prompt=prompt,
            mock_default=self.mock_output(task),
        )
        return AgentResult(
            agent_name=self.agent_name,
            task_id=task.task_id,
            output=result,
            revision_round=1 if revision_instruction else 0,
        )

    @staticmethod
    def to_pretty_json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=True)
