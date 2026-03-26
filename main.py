#!/usr/bin/env python3
"""
Single entry point (assignment requirement) to run the full system end-to-end.

This runs the same 5-agent orchestration used by the Flask UI, using inputs from
`startup_config.json` and environment variables in `.env`.
"""

from __future__ import annotations

import json
from pathlib import Path

from agents.ceo_agent import CEOAgent
from multi_agent_system.deepseek_client import DeepSeekClient
from multi_agent_system.env_utils import load_dotenv_file
from multi_agent_system.gemini_client import GeminiClient
from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.github_client import GitHubClient
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.redis_bus import RedisBus


def load_config(project_root: Path) -> dict:
    config_path = project_root / "startup_config.json"
    if config_path.exists():
        return json.loads(config_path.read_text(encoding="utf-8"))
    return {}


def main() -> None:
    project_root = Path(__file__).resolve().parent
    env = load_dotenv_file(project_root / ".env")
    config = load_config(project_root)

    execute_actions = bool(config.get("execute_actions", False))
    dry_run = bool(config.get("dry_run", not execute_actions))

    deepseek_client = DeepSeekClient(
        api_key=env.get("DEEPSEEK_API_KEY", ""),
        model=env.get("DEEPSEEK_MODEL", "deepseek-chat"),
    )
    gemini_client = GeminiClient(
        api_key=env.get("GEMINI_API_KEY", ""),
        model=env.get("GEMINI_MODEL", "gemini-2.0-flash"),
    )
    groq_client = GroqClient(
        api_key=env.get("GROQ_API_KEY", ""),
        model=env.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
        fallback=deepseek_client,
        gemini_fallback=gemini_client,
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

    redis_bus = RedisBus(
        host=env.get("REDIS_HOST", "localhost"),
        port=int(env.get("REDIS_PORT", "6379")),
    )

    startup_idea = config.get(
        "startup_idea",
        "InvoiceHound — a tool for freelance teams that generates one invoice for the client, "
        "internally splits earnings by hours logged, and auto-sends escalating payment reminders "
        "(Day 1 polite Slack, Day 7 firm Slack, Day 14 formal email with invoice) until the client pays.",
    )

    ceo = CEOAgent(
        groq_client=groq_client,
        deepseek_client=deepseek_client,
        gemini_client=gemini_client,
        redis_bus=redis_bus,
        slack_client=slack_client,
        github_client=github_client,
        sendgrid_client=sendgrid_client,
        slack_channel_id=env.get("SLACK_CHANNEL_ID", ""),
        launches_channel_id=env.get("LAUNCHES_CHANNEL_ID", ""),
        output_dir=project_root,
        dry_run_actions=(dry_run or (not execute_actions)),
        max_revisions=2,
    )

    result = ceo.run(startup_idea=startup_idea, dry_run=dry_run)
    print("\n=== InvoiceHound MAS complete ===")
    print(f"QA passed   : {result.get('qa', {}).get('passed')}")
    print(f"Decision log: {result.get('decision_log_path')}")
    print(f"Message log : {result.get('message_log_path')}")
    print("\nFinal summary:\n")
    print(result.get("final_summary_text", ""))


if __name__ == "__main__":
    main()

