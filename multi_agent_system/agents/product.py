"""Product agent (Agent 2) — Groq-only content generation."""

from __future__ import annotations

from typing import Any, Dict

from multi_agent_system.groq_client import GroqClient
from multi_agent_system.models import AgentResult, TaskMessage


class ProductAgent:
    agent_name = "product"

    def __init__(self, groq_client: GroqClient) -> None:
        self.groq = groq_client

    def mock_output(self, task: TaskMessage) -> Dict[str, Any]:
        # Keep this mock aligned with the assignment required JSON keys.
        mock_features = [
            {
                "name": "Single invoice per client (team-facing)",
                "description": "Generate one professional invoice to the client while keeping split details internal/confidential.",
                "priority": 1,
            },
            {
                "name": "Hour-based internal payment splitting",
                "description": "Compute each member’s share proportionally to logged hours.",
                "priority": 2,
            },
            {
                "name": "Escalating reminder engine (Day 1 / Day 7 / Day 14)",
                "description": "Send Slack reminders to the team on Day 1 and Day 7, then dispatch a formal email on Day 14 if unpaid.",
                "priority": 3,
            },
            {
                "name": "AI-written reminders with tone control",
                "description": "Write polite → firm → formal messages based on overdue day.",
                "priority": 4,
            },
            {
                "name": "Payment notification + internal settlement",
                "description": "When paid, notify the team and record each member’s settlement amount based on hours.",
                "priority": 5,
            },
        ]

        return {
            "value_proposition": (
                "InvoiceHound helps freelance teams get paid on time by sending one clean invoice, escalating follow-ups automatically, and splitting earnings fairly by logged hours."
            ),
            "personas": [
                {
                    "name": "Aisha",
                    "role": "Freelance UI/UX designer",
                    "pain_point": "She’s always the one chasing clients for payment, which strains relationships.",
                },
                {
                    "name": "Bilal",
                    "role": "Full-stack developer (micro-agency)",
                    "pain_point": "He wastes time calculating payment splits and explaining them to the team.",
                },
                {
                    "name": "Sara",
                    "role": "Freelance copywriter",
                    "pain_point": "Reminder tone felt too harsh before, so clients ignore follow-ups.",
                },
            ],
            # Assignment expects `features`; repo historically used `core_features_ranked`.
            "features": mock_features,
            "core_features_ranked": mock_features,
            "user_stories": [
                "As a freelance team lead, I want one client invoice so internal split details stay private.",
                "As a team member, I want to preview my exact cut so I trust the settlement calculation.",
                "As a freelancer, I want automated escalating reminders so I avoid awkward payment conversations.",
            ],
            "confirmation_message": "Product spec generated for InvoiceHound.",
        }

    def run(self, task: TaskMessage, revision_instruction: str = "") -> AgentResult:
        if not self.groq.enabled:
            output = self.mock_output(task)
        else:
            output = self.groq.generate_product_spec(
                startup_idea=task.startup_idea,
                task_brief=task.task_brief,
                revision_instruction=revision_instruction,
            )

        return AgentResult(
            agent_name=self.agent_name,
            task_id=task.task_id,
            output=output,
            revision_round=1 if revision_instruction else 0,
        )

