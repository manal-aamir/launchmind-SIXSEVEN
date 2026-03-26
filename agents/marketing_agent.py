"""Marketing agent — Groq-only LLM generation + SendGrid/Slack actions."""

from __future__ import annotations

from typing import Any, Dict, List

from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.models import AgentResult, TaskMessage


class MarketingAgent:
    agent_name = "marketing"

    def __init__(
        self,
        groq_client: GroqClient,
        sendgrid_client: SendGridClient,
        slack_client: SlackClient,
        launches_channel_id: str,
        dry_run: bool = True,
    ) -> None:
        self.groq = groq_client
        self.sendgrid_client = sendgrid_client
        self.slack_client = slack_client
        self.launches_channel_id = launches_channel_id
        self.dry_run = dry_run

    def _build_launch_blocks(self, tagline: str, landing_description: str, pr_url: str) -> List[Dict[str, Any]]:
        """Block Kit template for #launches."""
        return [
            {"type": "header", "text": {"type": "plain_text", "text": "New Launch: InvoiceHound"}},
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        f"*{tagline}*\n"
                        f"{landing_description}\n"
                        f"<{pr_url}|View GitHub PR>"
                    ),
                },
            },
            {
                "type": "context",
                "elements": [{"type": "mrkdwn", "text": "Status: Ready for review"}],
            },
        ]

    @staticmethod
    def _plain_to_html(plain_text: str) -> str:
        return f"<p>{(plain_text or '').replace(chr(10), '<br>')}</p>"

    def run(self, task: TaskMessage, revision_instruction: str = "") -> AgentResult:
        product_spec = task.context.get("product_spec", {})
        pr_url = str(task.context.get("pr_url", "")) or ""

        assets = self.groq.generate_marketing_assets(
            startup_idea=task.startup_idea,
            product_spec=product_spec,
            pr_url=pr_url,
            revision_instruction=revision_instruction,
        )

        cold_email = assets.get("cold_email", {}) or {}
        subject = str(cold_email.get("subject", "InvoiceHound launch"))
        body = str(cold_email.get("body", ""))
        html = self._plain_to_html(body)

        email_receipt: Dict[str, Any] = {"ok": "False", "status_code": "dry-run", "error": ""}
        slack_receipt: Dict[str, Any] = {"ok": False, "reason": "dry-run_or_missing_channel"}

        if not self.dry_run:
            email_receipt = self.sendgrid_client.send_email(
                subject=subject, plain_text=body, html_text=html
            )
            if self.launches_channel_id:
                blocks = self._build_launch_blocks(
                    tagline=str(assets.get("tagline", "")),
                    landing_description=str(assets.get("landing_description", "")),
                    pr_url=pr_url,
                )
                slack_receipt = self.slack_client.post_block_message(
                    channel=self.launches_channel_id,
                    text="InvoiceHound launch update",
                    blocks=blocks,
                )

        output: Dict[str, Any] = {
            "tagline": assets.get("tagline", ""),
            "landing_description": assets.get("landing_description", ""),
            "cold_email": assets.get("cold_email", {}),
            "social_posts": assets.get("social_posts", {}),
            "pr_url": pr_url,
            "email_receipt": email_receipt,
            "slack_receipt": slack_receipt,
        }
        return AgentResult(
            agent_name=self.agent_name,
            task_id=task.task_id,
            output=output,
            revision_round=1 if revision_instruction else 0,
        )


__all__ = ["MarketingAgent"]

