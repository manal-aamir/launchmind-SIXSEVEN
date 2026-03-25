#!/usr/bin/env python3
"""
InvoiceHound end-to-end demo.

Demonstrates:
  1. Invoice generation with team split by hours
  2. Day 1  — polite Slack reminder (client hasn't paid)
  3. Day 7  — firm Slack nudge
  4. Day 14 — formal SendGrid email with full HTML invoice
  5. Payment received — Slack split notification to team

Usage:
  python3 invoicehound_demo.py                  # dry-run, no real side effects
  python3 invoicehound_demo.py --execute-actions # real Slack + SendGrid calls
  python3 invoicehound_demo.py --day 14          # simulate a specific overdue day
  python3 invoicehound_demo.py --mark-paid       # show payment distribution
"""

import argparse
import json
from pathlib import Path

from multi_agent_system.env_utils import load_dotenv_file
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.invoice_engine import InvoiceEngine, LineItem, TeamMember
from multi_agent_system.reminder_engine import ReminderEngine


# ---------------------------------------------------------------------------
# Sample data
# ---------------------------------------------------------------------------

SAMPLE_TEAM = [
    TeamMember(name="Aisha",  email="aisha@team.dev",  hours_worked=40, role="UI/UX Designer"),
    TeamMember(name="Bilal",  email="bilal@team.dev",  hours_worked=60, role="Full-Stack Dev"),
    TeamMember(name="Sara",   email="sara@team.dev",   hours_worked=20, role="Copywriter"),
]

SAMPLE_ITEMS = [
    LineItem(description="Website redesign + development", quantity=1,  unit_price=3000),
    LineItem(description="Copywriting & content strategy",  quantity=1,  unit_price=800),
    LineItem(description="UI/UX design sprints (x2)",       quantity=2,  unit_price=600),
]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="InvoiceHound reminder + split demo")
    parser.add_argument("--execute-actions", action="store_true",
                        help="Send real Slack messages and SendGrid emails.")
    parser.add_argument("--day", type=int, default=None,
                        help="Simulate N days overdue (1, 7, or 14).")
    parser.add_argument("--mark-paid", action="store_true",
                        help="Skip reminders and demo payment distribution instead.")
    parser.add_argument("--save-invoice", action="store_true",
                        help="Save invoice HTML to disk.")
    return parser


def main() -> None:
    project_root = Path(__file__).resolve().parent
    env = load_dotenv_file(project_root / ".env")
    args = build_parser().parse_args()

    dry_run = not args.execute_actions

    # Build clients
    slack_client = SlackClient(bot_token=env.get("SLACK_BOT_TOKEN", ""))
    sendgrid_client = SendGridClient(
        api_key=env.get("SENDGRID_API_KEY", ""),
        from_email=env.get("SENDGRID_FROM_EMAIL", ""),
        to_email=env.get("SENDGRID_TO_EMAIL", ""),
    )

    # Build invoice
    engine = InvoiceEngine()
    inv = engine.create_invoice(
        project_name="Brand Redesign — Acme Corp",
        client_name="Acme Corp",
        client_email=env.get("SENDGRID_TO_EMAIL", "client@example.com"),
        team_members=SAMPLE_TEAM,
        line_items=SAMPLE_ITEMS,
        days_until_due=14,
    )

    print("=" * 60)
    print("INVOICEHOUND DEMO")
    print("=" * 60)
    print(f"Invoice ID  : {inv.invoice_id}")
    print(f"Project     : {inv.project_name}")
    print(f"Client      : {inv.client_name}  ({inv.client_email})")
    print(f"Total Due   : {inv.currency} {inv.total_amount:,.2f}")
    print(f"Issue Date  : {inv.issue_date}")
    print(f"Due Date    : {inv.due_date}")
    print()
    print("Payment split by hours:")
    splits = inv.payment_splits
    for name, data in splits.items():
        print(f"  {name:12} {data['role']:20} {data['hours']:5}h  "
              f"{data['percentage']:5.1f}%  → {inv.currency} {data['amount']:,.2f}")

    # Optionally save HTML invoice
    if args.save_invoice:
        html_path = project_root / f"invoice_{inv.invoice_id}.html"
        html_path.write_text(engine.generate_html(inv), encoding="utf-8")
        print(f"\nInvoice HTML saved → {html_path}")

    # Build reminder engine
    reminder = ReminderEngine(
        slack_client=slack_client,
        sendgrid_client=sendgrid_client,
        slack_channel_id=env.get("SLACK_CHANNEL_ID", "") or env.get("LAUNCHES_CHANNEL_ID", ""),
        dry_run=dry_run,
    )

    print()
    print("─" * 60)

    if args.mark_paid:
        print("SCENARIO: Client pays — distributing to team\n")
        result = reminder.distribute_payment(inv)
        print("Splits sent to team:")
        print(json.dumps(result["splits"], indent=2))
        print("\nSlack receipt:", result["slack_receipt"])
        return

    # Reminder simulation
    days = args.day
    if days is None:
        # Run all three scenarios sequentially for demo
        for d in [1, 7, 14]:
            _run_reminder(reminder, inv, d, dry_run)
    else:
        _run_reminder(reminder, inv, days, dry_run)


def _run_reminder(reminder: ReminderEngine, inv, days: int, dry_run: bool) -> None:
    label = {1: "DAY 1 — Polite Slack reminder",
             7: "DAY 7 — Firm Slack nudge",
             14: "DAY 14 — Formal email + invoice"}.get(days, f"DAY {days}")
    print(f"\nSCENARIO: {label}")
    result = reminder.check_and_send(inv, simulate_days_overdue=days)
    print(f"  Action taken  : {result['action_taken']}")
    print(f"  Dry run       : {dry_run}")
    print(f"  Receipts      : {json.dumps(result['receipts'], indent=4)}")
    print("─" * 60)


if __name__ == "__main__":
    main()
