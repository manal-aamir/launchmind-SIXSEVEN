"""Marketing agent with email + Slack launch actions."""

from typing import Dict

from multi_agent_system.agents.base import BaseAgent
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.models import AgentResult, TaskMessage


class MarketingAgent(BaseAgent):
    agent_name = "marketing"

    def __init__(
        self,
        llm,
        sendgrid_client: SendGridClient,
        slack_client: SlackClient,
        launches_channel_id: str,
        dry_run: bool = True,
    ) -> None:
        super().__init__(llm)
        self.sendgrid_client = sendgrid_client
        self.slack_client = slack_client
        self.launches_channel_id = launches_channel_id
        self.dry_run = dry_run

    def build_system_prompt(self) -> str:
        return "You are a marketing strategist. Return JSON."

    def build_output_contract(self) -> str:
        return "Return JSON with campaign assets."

    def run(self, task: TaskMessage, revision_instruction: str = "") -> AgentResult:
        product_spec = task.context.get("product_spec", {})
        pr_url = str(task.context.get("pr_url", ""))
        assets = self.llm.generate_marketing_assets(task.startup_idea, product_spec, pr_url)

        subject = str(assets.get("cold_email", {}).get("subject", "InvoiceHound launch"))
        body = str(assets.get("cold_email", {}).get("body", ""))
        html = f"<p>{body.replace(chr(10), '<br>')}</p>"

        email_receipt = {"ok": "False", "status_code": "dry-run", "error": ""}
        slack_receipt: Dict[str, object] = {"ok": False, "reason": "dry-run_or_missing_channel"}
        if not self.dry_run:
            email_receipt = self.sendgrid_client.send_email(subject=subject, plain_text=body, html_text=html)
            blocks = [
                {"type": "header", "text": {"type": "plain_text", "text": "New Launch: InvoiceHound 🐕"}},
                {
                    "type": "section",
                    "text": {
                        "type": "mrkdwn",
                        "text": (
                            f"*{assets.get('tagline', '')}*\n"
                            f"{assets.get('landing_description', '')}\n"
                            f"<{pr_url}|View GitHub PR>"
                        ),
                    },
                },
                {"type": "context", "elements": [{"type": "mrkdwn", "text": "Status: Ready for review"}]},
            ]
            if self.launches_channel_id:
                slack_receipt = self.slack_client.post_block_message(
                    channel=self.launches_channel_id,
                    text="InvoiceHound launch update",
                    blocks=blocks,
                )

        output = {
            "tagline": assets.get("tagline", ""),
            "landing_description": assets.get("landing_description", ""),
            "cold_email": assets.get("cold_email", {}),
            "social_posts": assets.get("social_posts", {}),
            "pr_url": pr_url,
            "email_receipt": email_receipt,
            "slack_receipt": slack_receipt,
        }
        return AgentResult(agent_name=self.agent_name, task_id=task.task_id, output=output)

