#!/usr/bin/env python3
"""
Run the InvoiceHound 5-agent workflow end-to-end.

The startup idea and settings are read from startup_config.json — no terminal
input is needed. Just run:

    python3 run_multi_agent_demo.py

To enable real GitHub / SendGrid / Slack side-effects, set
  "execute_actions": true  in startup_config.json
or add OPENAI_API_KEY to .env.
"""

import json
from pathlib import Path

from multi_agent_system.agents.ceo import CEOAgent
from multi_agent_system.env_utils import load_dotenv_file
from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.github_client import GitHubClient
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.llm_client import LLMClient


def load_config(project_root: Path) -> dict:
    config_path = project_root / "startup_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    env = load_dotenv_file(project_root / ".env")
    config = load_config(project_root)

    # All inputs come from startup_config.json — no terminal prompts needed.
    startup_idea = config.get(
        "startup_idea",
        "InvoiceHound — a tool for freelance teams that generates one invoice for the client, "
        "internally splits earnings by hours logged, and auto-sends escalating payment reminders "
        "until the client pays.",
    )
    execute_actions = config.get("execute_actions", False)
    dry_run = config.get("dry_run", True)

    print("=" * 60)
    print("InvoiceHound Multi-Agent System")
    print("=" * 60)
    print(f"Startup idea : {startup_idea[:80]}...")
    print(f"Live actions : {'YES — real GitHub / Slack / Email' if execute_actions else 'NO  — dry-run (safe)'}")
    print(f"Slack post   : {'YES' if not dry_run else 'NO  — dry-run'}")
    print("=" * 60)
    print()

    llm = LLMClient(
        api_key=env.get("OPENAI_API_KEY", ""),
        model=env.get("OPENAI_MODEL", "gpt-4o-mini"),
    )
    groq_client = GroqClient(
        api_key=env.get("GROQ_API_KEY", ""),
        model=env.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    )
    github_client = GitHubClient(
        token=env.get("GITHUB_TOKEN", ""),
        repo=env.get("GITHUB_REPO", "manal-aamir/launchmind-SIXSEVEN"),
    )
    sendgrid_client = SendGridClient(
        api_key=env.get("SENDGRID_API_KEY", ""),
        from_email=env.get("SENDGRID_FROM_EMAIL", ""),
        to_email=env.get("SENDGRID_TO_EMAIL", ""),
    )
    slack_client = SlackClient(bot_token=env.get("SLACK_BOT_TOKEN", ""))

    ceo = CEOAgent(
        llm=llm,
        groq_client=groq_client,
        slack_client=slack_client,
        github_client=github_client,
        sendgrid_client=sendgrid_client,
        slack_channel_id=env.get("SLACK_CHANNEL_ID", ""),
        launches_channel_id=env.get("LAUNCHES_CHANNEL_ID", ""),
        output_dir=project_root,
        dry_run_actions=(not execute_actions),
        max_revisions=2,
    )

    result = ceo.run(startup_idea=startup_idea, dry_run=dry_run)

    print("=== CEO ORCHESTRATION COMPLETE ===")
    print("QA passed    :", result["qa"]["passed"])
    print("Decision log :", result["decision_log_path"])
    print("Slack posted :", bool(result["slack_response"].get("ok")))
    print()
    print("Final summary:")
    print(result["final_summary_text"])
    print()
    print("Task messages sent to each agent:")
    for agent, msg in result["task_messages"].items():
        print(f"  [{agent.upper()}] {msg['task_brief'][:90]}")


if __name__ == "__main__":
    main()

