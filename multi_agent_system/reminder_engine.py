"""
Reminder engine for InvoiceHound.

Escalation schedule (measured from invoice due_date):
  Day 1  → Polite Slack message to the team channel (client hasn't paid yet)
  Day 7  → Firmer Slack message (nudge tone)
  Day 14 → Formal SendGrid email to client with full HTML invoice embedded

All actions are real side-effects when dry_run=False.
"""

from __future__ import annotations

import base64
import json
from datetime import date, timedelta
from typing import Any, Dict

from multi_agent_system.invoice_engine import InvoiceEngine, InvoiceRecord
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient


class ReminderEngine:
    def __init__(
        self,
        slack_client: SlackClient,
        sendgrid_client: SendGridClient,
        slack_channel_id: str,
        dry_run: bool = True,
    ) -> None:
        self.slack_client = slack_client
        self.sendgrid_client = sendgrid_client
        self.slack_channel_id = slack_channel_id
        self.dry_run = dry_run
        self._invoice_engine = InvoiceEngine()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _days_overdue(self, inv: InvoiceRecord, today: date | None = None) -> int:
        due = date.fromisoformat(inv.due_date)
        return max(0, ((today or date.today()) - due).days)

    def _slack_blocks_day1(self, inv: InvoiceRecord) -> list:
        return [
            {"type": "header",
             "text": {"type": "plain_text", "text": f"Payment reminder: {inv.project_name}"}},
            {"type": "section",
             "text": {"type": "mrkdwn",
                      "text": (
                          f"Hi team — friendly heads up that invoice *{inv.invoice_id}* "
                          f"sent to *{inv.client_name}* is now due.\n\n"
                          f"*Amount:* {inv.currency} {inv.total_amount:,.2f}\n"
                          f"*Due date:* {inv.due_date}\n\n"
                          "Client has been notified. We'll follow up automatically."
                      )}},
            {"type": "context",
             "elements": [{"type": "mrkdwn",
                           "text": "InvoiceHound • Day 1 reminder • Polite tone"}]},
        ]

    def _slack_blocks_day7(self, inv: InvoiceRecord) -> list:
        return [
            {"type": "header",
             "text": {"type": "plain_text",
                      "text": f"Payment still pending: {inv.project_name}"}},
            {"type": "section",
             "text": {"type": "mrkdwn",
                      "text": (
                          f"Invoice *{inv.invoice_id}* for *{inv.client_name}* is now "
                          f"*7 days overdue*.\n\n"
                          f"*Amount outstanding:* {inv.currency} {inv.total_amount:,.2f}\n\n"
                          "A firmer follow-up has been sent. "
                          "If unpaid by Day 14, a formal email with the invoice will be dispatched."
                      )}},
            {"type": "context",
             "elements": [{"type": "mrkdwn",
                           "text": "InvoiceHound • Day 7 reminder • Firm nudge"}]},
        ]

    def _email_subject_day14(self, inv: InvoiceRecord) -> str:
        return (
            f"FINAL NOTICE — Invoice {inv.invoice_id} | {inv.project_name} | "
            f"{inv.currency} {inv.total_amount:,.2f} — 14 Days Overdue"
        )

    def _email_body_day14(self, inv: InvoiceRecord) -> str:
        return f"""Dear {inv.client_name},

This is a FINAL FORMAL NOTICE regarding the outstanding payment for services
rendered under the project "{inv.project_name}".

Despite our previous reminders on Day 1 and Day 7, invoice {inv.invoice_id}
remains unpaid as of today — now 14 days past the agreed due date.

─────────────────────────────────────────
INVOICE DETAILS
─────────────────────────────────────────
  Invoice ID   : {inv.invoice_id}
  Project      : {inv.project_name}
  Issue Date   : {inv.issue_date}
  Due Date     : {inv.due_date}
  Amount Due   : {inv.currency} {inv.total_amount:,.2f}
  Status       : OVERDUE — 14 DAYS
─────────────────────────────────────────

This matter now requires your IMMEDIATE attention. The full invoice is
attached to this email for your reference.

We expect payment to be settled within 48 hours of receiving this notice.
Failure to do so may result in one or more of the following actions:

  1. Formal escalation to a collections process.
  2. Suspension of all ongoing or future work under this engagement.
  3. Legal proceedings to recover the outstanding amount plus any
     accrued interest and associated costs.

If payment has already been made, please reply to this email immediately
with your payment confirmation and receipt so we can update our records
and close this matter.

If you are experiencing difficulty making payment, we ask that you contact
us within 24 hours to discuss a resolution. We are open to structured
payment arrangements, but this must be agreed in writing before the
48-hour deadline.

This is an automated final notice generated by InvoiceHound on behalf of
the project team. No further reminders will be sent after this notice.

Regards,
InvoiceHound Automated Collections
(on behalf of: {inv.project_name} team)
"""

    def _email_html_day14(self, inv: InvoiceRecord) -> str:
        invoice_html = self._invoice_engine.generate_html(inv)
        from datetime import date as _date
        today = _date.today().isoformat()
        header = f"""
<div style="font-family:Arial,sans-serif;max-width:720px;margin:0 auto;padding:0">

  <!-- Red alert banner -->
  <div style="background:#7f1d1d;color:#fff;padding:14px 28px;text-align:center">
    <div style="font-size:13px;letter-spacing:2px;font-weight:700;text-transform:uppercase">
      ⚠ Final Notice — Overdue Payment
    </div>
  </div>

  <!-- Header bar -->
  <div style="background:#991b1b;color:#fff;padding:20px 28px;display:flex;justify-content:space-between;align-items:center">
    <div>
      <div style="font-size:20px;font-weight:900;letter-spacing:-0.5px">INVOICEHOUND</div>
      <div style="font-size:11px;opacity:.8;margin-top:2px">Automated Collections Notice</div>
    </div>
    <div style="text-align:right;font-size:12px;opacity:.9">
      <div>Invoice: <strong>{inv.invoice_id}</strong></div>
      <div>Issued: {inv.issue_date}</div>
      <div>Notice Date: {today}</div>
    </div>
  </div>

  <!-- Body -->
  <div style="padding:28px;background:#fff;border:1px solid #fecaca;border-top:none">

    <p style="font-size:14px;margin-bottom:18px">Dear <strong>{inv.client_name}</strong>,</p>

    <p style="font-size:14px;margin-bottom:16px;line-height:1.7">
      This is a <strong style="color:#dc2626">FINAL FORMAL NOTICE</strong> regarding your
      outstanding payment for services rendered under the project
      <em><strong>{inv.project_name}</strong></em>. Despite reminders sent on
      <strong>Day 1</strong> and <strong>Day 7</strong>, the invoice below remains
      <strong style="color:#dc2626">14 days overdue</strong>.
    </p>

    <!-- Invoice summary box -->
    <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;padding:18px 20px;margin:20px 0">
      <table style="width:100%;font-size:13px;border-collapse:collapse">
        <tr><td style="padding:5px 0;color:#6b7280">Invoice ID</td>
            <td style="padding:5px 0;font-weight:700">{inv.invoice_id}</td></tr>
        <tr><td style="padding:5px 0;color:#6b7280">Project</td>
            <td style="padding:5px 0">{inv.project_name}</td></tr>
        <tr><td style="padding:5px 0;color:#6b7280">Issue Date</td>
            <td style="padding:5px 0">{inv.issue_date}</td></tr>
        <tr><td style="padding:5px 0;color:#6b7280">Due Date</td>
            <td style="padding:5px 0;color:#dc2626;font-weight:700">{inv.due_date}</td></tr>
        <tr><td style="padding:5px 0;color:#6b7280">Days Overdue</td>
            <td style="padding:5px 0;color:#dc2626;font-weight:700">14 days</td></tr>
        <tr><td style="padding:5px 0;font-size:15px;font-weight:700;padding-top:12px;border-top:1px solid #fecaca">
              AMOUNT DUE</td>
            <td style="padding:5px 0;font-size:18px;font-weight:900;color:#dc2626;padding-top:12px;border-top:1px solid #fecaca">
              {inv.currency} {inv.total_amount:,.2f}</td></tr>
      </table>
    </div>

    <p style="font-size:14px;font-weight:700;color:#991b1b;margin:16px 0">
      IMMEDIATE ACTION REQUIRED — Payment must be received within 48 hours.
    </p>

    <p style="font-size:13px;color:#374151;margin-bottom:12px;line-height:1.7">
      Failure to settle this invoice within 48 hours of receiving this notice may result in:
    </p>
    <ol style="font-size:13px;color:#374151;line-height:2;padding-left:20px;margin-bottom:18px">
      <li>Formal escalation to a <strong>third-party collections agency</strong>.</li>
      <li><strong>Suspension of all ongoing and future work</strong> under this engagement.</li>
      <li><strong>Legal proceedings</strong> to recover the outstanding amount, plus interest
          and associated costs.</li>
    </ol>

    <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:16px">
      If payment has already been made, please reply immediately with your
      <strong>payment confirmation and receipt</strong> so we can close this matter.
    </p>
    <p style="font-size:13px;color:#374151;line-height:1.7;margin-bottom:24px">
      If you are experiencing difficulty, contact us within <strong>24 hours</strong>
      to discuss a structured payment arrangement. This must be agreed in writing
      before the deadline.
    </p>

    <div style="background:#f9fafb;border-top:1px solid #e5e7eb;padding:14px 0;font-size:12px;color:#9ca3af">
      This is a final automated notice from <strong>InvoiceHound</strong> on behalf of
      the <em>{inv.project_name}</em> team. No further reminders will be sent.
    </div>
  </div>

  <!-- Full invoice below -->
  <div style="padding:16px 28px;background:#f9fafb;border:1px solid #fecaca;border-top:none;
              font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:1px">
    Full Invoice — Attached Below
  </div>
</div>
<br>
"""
        return header + invoice_html

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_and_send(
        self, inv: InvoiceRecord, simulate_days_overdue: int | None = None
    ) -> Dict[str, Any]:
        """
        Check overdue status and trigger the appropriate reminder action.

        Args:
            inv: The InvoiceRecord to check.
            simulate_days_overdue: Override actual overdue days for testing.

        Returns:
            dict with keys: days_overdue, action_taken, receipts.
        """
        if inv.paid:
            return {"days_overdue": 0, "action_taken": "none_paid", "receipts": {}}

        days = simulate_days_overdue if simulate_days_overdue is not None else self._days_overdue(inv)
        result: Dict[str, Any] = {"days_overdue": days, "action_taken": "none", "receipts": {}}

        if days >= 14:
            result["action_taken"] = "day_14_email"
            subject = self._email_subject_day14(inv)
            body_text = self._email_body_day14(inv)
            body_html = self._email_html_day14(inv)
            if not self.dry_run:
                receipt = self.sendgrid_client.send_email(
                    subject=subject,
                    plain_text=body_text,
                    html_text=body_html,
                )
                result["receipts"]["email"] = receipt
            else:
                result["receipts"]["email"] = {"ok": "dry-run", "subject": subject}

        elif days >= 7:
            result["action_taken"] = "day_7_slack"
            blocks = self._slack_blocks_day7(inv)
            if not self.dry_run and self.slack_channel_id:
                receipt = self.slack_client.post_block_message(
                    channel=self.slack_channel_id,
                    text=f"Payment overdue (7 days): {inv.invoice_id}",
                    blocks=blocks,
                )
                result["receipts"]["slack"] = receipt
            else:
                result["receipts"]["slack"] = {"ok": "dry-run", "day": 7}

        elif days >= 1:
            result["action_taken"] = "day_1_slack"
            blocks = self._slack_blocks_day1(inv)
            if not self.dry_run and self.slack_channel_id:
                receipt = self.slack_client.post_block_message(
                    channel=self.slack_channel_id,
                    text=f"Payment reminder (Day 1): {inv.invoice_id}",
                    blocks=blocks,
                )
                result["receipts"]["slack"] = receipt
            else:
                result["receipts"]["slack"] = {"ok": "dry-run", "day": 1}

        return result

    def distribute_payment(self, inv: InvoiceRecord) -> Dict[str, Any]:
        """
        Called when client pays. Returns split summary.
        Posts a Slack notification to the team.
        """
        inv.paid = True
        splits = inv.calculate_splits()

        lines = "\n".join(
            f"  • {name}: {inv.currency} {data['amount']:,.2f} ({data['hours']}h, {data['percentage']}%)"
            for name, data in splits.items()
        )
        slack_text = (
            f"Payment received for *{inv.project_name}* (Invoice {inv.invoice_id}) — "
            f"{inv.currency} {inv.total_amount:,.2f}\n\n"
            f"*Team split:*\n{lines}"
        )

        blocks = [
            {"type": "header",
             "text": {"type": "plain_text", "text": f"Payment received: {inv.project_name}"}},
            {"type": "section",
             "text": {"type": "mrkdwn", "text": slack_text}},
            {"type": "context",
             "elements": [{"type": "mrkdwn",
                           "text": "InvoiceHound • Splits calculated by hours worked"}]},
        ]

        receipt: Dict[str, Any] = {"ok": "dry-run"}
        if not self.dry_run and self.slack_channel_id:
            receipt = self.slack_client.post_block_message(
                channel=self.slack_channel_id,
                text=f"Payment received: {inv.project_name}",
                blocks=blocks,
            )
        return {"splits": splits, "slack_receipt": receipt}
