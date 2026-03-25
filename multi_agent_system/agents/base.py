"""Base class for domain agents."""

import json
from typing import Any, Dict

from multi_agent_system.llm_client import LLMClient
from multi_agent_system.models import AgentResult, TaskMessage


class BaseAgent:
    agent_name = "base"

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    def build_system_prompt(self) -> str:
        raise NotImplementedError

    def build_output_contract(self) -> str:
        raise NotImplementedError

    def mock_output(self, task: TaskMessage) -> Dict[str, Any]:
        return {"status": "mock", "task_brief": task.task_brief}

    def run(self, task: TaskMessage, revision_instruction: str = "") -> AgentResult:
        if not self.llm.enabled:
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

        response = self.llm._client.responses.create(
            model=self.llm.model,
            input=[
                {"role": "system", "content": self.build_system_prompt()},
                {"role": "user", "content": prompt + "\n" + self.build_output_contract()},
            ],
            temperature=0.3,
        )
        text = response.output_text
        data = self.llm._extract_json(text)
        return AgentResult(
            agent_name=self.agent_name,
            task_id=task.task_id,
            output=data,
            revision_round=1 if revision_instruction else 0,
        )

    @staticmethod
    def to_pretty_json(payload: Dict[str, Any]) -> str:
        return json.dumps(payload, indent=2, ensure_ascii=True)

