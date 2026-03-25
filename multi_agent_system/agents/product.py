"""Product agent."""

from typing import Any, Dict

from multi_agent_system.agents.base import BaseAgent
from multi_agent_system.models import TaskMessage


class ProductAgent(BaseAgent):
    agent_name = "product"

    def build_system_prompt(self) -> str:
        return (
            "You are a Product Manager agent. Think in MVP terms. "
            "Return only JSON with practical product deliverables."
        )

    def build_output_contract(self) -> str:
        return (
            "Return JSON with keys: value_proposition, personas, core_features_ranked, "
            "user_stories, assumptions, confirmation_message. "
            "Personas must include realistic pain points about payment chasing and split fairness."
        )

    def mock_output(self, task: TaskMessage) -> Dict[str, Any]:
        return {
            "value_proposition": (
                "InvoiceHound helps freelance teams get paid on time by sending one clean invoice "
                "to clients and handling awkward follow-ups automatically."
            ),
            "personas": [
                {
                    "name": "Aisha",
                    "profile": "Freelance UI/UX designer collaborating with a developer.",
                    "pain_point": "She is always the one chasing clients for payment.",
                },
                {
                    "name": "Bilal",
                    "profile": "Full-stack developer running a 3-person micro-agency.",
                    "pain_point": "He spends hours manually calculating payment splits.",
                },
                {
                    "name": "Sara",
                    "profile": "Freelance copywriter working with a designer.",
                    "pain_point": "Reminder tone has hurt client relationships before.",
                },
            ],
            "core_features_ranked": [
                "Single invoice generation for the whole team",
                "Internal hour-based earnings split calculator",
                "Escalating reminder engine (Day 1 / Day 7 / Day 14)",
                "AI-written reminder emails (friendly / firm / formal)",
                "Team payment notification once client pays",
            ],
            "user_stories": [
                "As a freelance team lead, I want one client invoice so internal split details remain private.",
                "As a team member, I want to preview my exact cut before sending the invoice.",
                "As a freelancer, I want automated reminders so I avoid awkward payment conversations.",
            ],
            "assumptions": ["Early users are freelancer teams of 2-5 members."],
            "confirmation_message": "Product spec generated for InvoiceHound.",
        }

