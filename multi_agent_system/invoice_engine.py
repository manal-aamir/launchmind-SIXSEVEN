"""
Invoice engine for InvoiceHound.

Responsibilities:
- Accept a project, client info, and list of team members + hours worked.
- Calculate each member's share from the total amount (proportional to hours).
- Generate a professional HTML invoice that can be sent as-is or embedded in email.
- Return a structured InvoiceRecord for the reminder engine to use.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import List


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class TeamMember:
    name: str
    email: str
    hours_worked: float
    role: str = "Team Member"


@dataclass
class LineItem:
    description: str
    quantity: float
    unit_price: float

    @property
    def total(self) -> float:
        return round(self.quantity * self.unit_price, 2)


@dataclass
class InvoiceRecord:
    invoice_id: str
    project_name: str
    client_name: str
    client_email: str
    issue_date: str
    due_date: str
    line_items: List[LineItem]
    team_members: List[TeamMember]
    currency: str = "PKR"
    paid: bool = False
    payment_splits: dict = field(default_factory=dict)

    @property
    def total_amount(self) -> float:
        return round(sum(item.total for item in self.line_items), 2)

    def calculate_splits(self) -> dict:
        """Distribute total by hours worked proportion."""
        total_hours = sum(m.hours_worked for m in self.team_members)
        if total_hours == 0:
            return {}
        splits = {}
        for member in self.team_members:
            share = round((member.hours_worked / total_hours) * self.total_amount, 2)
            splits[member.name] = {
                "email": member.email,
                "role": member.role,
                "hours": member.hours_worked,
                "percentage": round((member.hours_worked / total_hours) * 100, 1),
                "amount": share,
            }
        self.payment_splits = splits
        return splits


# ---------------------------------------------------------------------------
# Invoice generator
# ---------------------------------------------------------------------------

class InvoiceEngine:
    def create_invoice(
        self,
        project_name: str,
        client_name: str,
        client_email: str,
        team_members: List[TeamMember],
        line_items: List[LineItem],
        days_until_due: int = 14,
        currency: str = "PKR",
    ) -> InvoiceRecord:
        today = date.today()
        invoice = InvoiceRecord(
            invoice_id=f"INV-{uuid.uuid4().hex[:8].upper()}",
            project_name=project_name,
            client_name=client_name,
            client_email=client_email,
            issue_date=today.isoformat(),
            due_date=(today + timedelta(days=days_until_due)).isoformat(),
            line_items=line_items,
            team_members=team_members,
            currency=currency,
        )
        invoice.calculate_splits()
        return invoice

    def generate_html(
        self,
        inv: InvoiceRecord,
        include_internal_split: bool = False,
        show_status_banner: bool = True,
    ) -> str:
        subtotal = inv.total_amount
        gst      = round(subtotal * 0.10, 2)
        total    = round(subtotal + gst, 2)
        status_color = "#16a34a" if inv.paid else "#dc2626"
        status_label = "PAID"    if inv.paid else "UNPAID"

        # Derive days_until_due for professional footer text
        try:
            from datetime import date as _date
            issue = _date.fromisoformat(inv.issue_date)
            due   = _date.fromisoformat(inv.due_date)
            days_until_due = (due - issue).days
        except Exception:
            days_until_due = 14

        item_rows = ""
        for idx, item in enumerate(inv.line_items, 1):
            bg = "#ffffff" if idx % 2 else "#f9fafb"
            qty_display = "-" if item.quantity == 1 else (
                int(item.quantity) if item.quantity == int(item.quantity) else item.quantity
            )
            item_rows += f"""
            <tr style="background:{bg}">
              <td style="text-align:center;border:1px solid #d1d5db;padding:9px 8px">{idx}</td>
              <td style="border:1px solid #d1d5db;padding:9px 12px">{item.description}</td>
              <td style="text-align:center;border:1px solid #d1d5db;padding:9px 8px">{qty_display}</td>
              <td style="text-align:right;border:1px solid #d1d5db;padding:9px 12px">Rs {item.unit_price:,.2f}</td>
              <td style="text-align:right;border:1px solid #d1d5db;padding:9px 12px;font-weight:600">Rs {item.total:,.2f}</td>
            </tr>"""

        split_rows = ""
        for name, data in inv.payment_splits.items():
            split_rows += f"""
            <tr>
              <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb">{name}</td>
              <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;color:#6b7280">{data['role']}</td>
              <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;text-align:center">{data['hours']}h</td>
              <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;text-align:center">{data['percentage']}%</td>
              <td style="padding:7px 10px;border-bottom:1px solid #e5e7eb;text-align:right;color:#16a34a;font-weight:700">Rs {data['amount']:,.2f}</td>
            </tr>"""

        internal_split_html = (
            '<div class="split-section"><h3>🔒 Internal Payment Split — Confidential (not sent to client)</h3>'
            '<table class="split"><thead><tr><th>Name</th><th>Role</th><th style="text-align:center">Hours</th>'
            '<th style="text-align:center">Share</th><th style="text-align:right">Amount</th></tr></thead><tbody>'
            + split_rows
            + '</tbody></table></div>'
        ) if (include_internal_split and inv.payment_splits) else ""

        status_banner_html = (
            f'<div class="status-bar">{status_label}</div>'
            if show_status_banner else ""
        )

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width,initial-scale=1">
  <title>Invoice {inv.invoice_id} — {inv.project_name}</title>
  <style>
    * {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, Helvetica, sans-serif; font-size: 13px; color: #1f2937;
            background: #f3f4f6; padding: 32px 16px; }}
    .page {{ max-width: 860px; margin: 0 auto; background: #ffffff;
             box-shadow: 0 4px 24px rgba(0,0,0,.12); }}

    /* ── Brand accent bar ── */
    .accent-bar {{ height: 4px; background: #1a6b3c; }}

    /* ── Header ── */
    .header {{ display: flex; justify-content: space-between; align-items: flex-start;
               padding: 28px 32px 20px; border-bottom: 3px solid #1d4f3e; }}
    .brand {{ display: flex; align-items: center; gap: 14px; }}
    .logo-box {{ width: 64px; height: 64px; background: #1d4f3e; border-radius: 6px;
                 display: flex; align-items: center; justify-content: center; }}
    .logo-box span {{ color: #fff; font-size: 22px; font-weight: 900; letter-spacing: -1px; }}
    .brand-lines {{ font-size: 11px; color: #4b5563; line-height: 1.7; text-transform: uppercase; letter-spacing: .5px; }}
    .brand-name {{ font-size: 18px; font-weight: 800; color: #1d4f3e; margin-top: 4px; }}
    .doc-title {{ text-align: right; }}
    .doc-title h1 {{ font-size: 22px; font-weight: 800; color: #1d4f3e; letter-spacing: 1px;
                     text-transform: uppercase; margin-bottom: 8px; }}
    .doc-title table {{ margin-left: auto; font-size: 12px; }}
    .doc-title td {{ padding: 2px 6px; }}
    .doc-title td:first-child {{ color: #6b7280; }}

    /* ── Status badge ── */
    .status-bar {{ background: {status_color}; color: #fff; text-align: center; padding: 5px;
                   font-weight: 700; font-size: 13px; letter-spacing: 2px; }}

    /* ── Party info ── */
    .parties {{ display: grid; grid-template-columns: 1fr 1fr; border-bottom: 1px solid #d1d5db; }}
    .party {{ padding: 18px 32px; }}
    .party:first-child {{ border-right: 1px solid #d1d5db; }}
    .party-label {{ font-size: 10px; font-weight: 700; color: #9ca3af; text-transform: uppercase;
                    letter-spacing: 1px; margin-bottom: 8px; }}
    .party-name {{ font-size: 15px; font-weight: 700; color: #111827; margin-bottom: 6px; }}
    .party-detail {{ font-size: 12px; color: #4b5563; line-height: 1.8; }}

    /* ── Items table ── */
    .items-section {{ padding: 0 32px 20px; }}
    .section-title {{ font-size: 11px; font-weight: 700; color: #6b7280; text-transform: uppercase;
                      letter-spacing: 1px; padding: 16px 0 10px; }}
    table.items {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.items thead tr {{ background: #1d4f3e; color: #fff; }}
    table.items thead th {{ padding: 10px 12px; text-align: left; font-size: 11px;
                            font-weight: 600; letter-spacing: .5px; border: 1px solid #166534; }}
    table.items thead th.num {{ text-align: center; }}
    table.items tfoot td {{ padding: 9px 12px; border: 1px solid #d1d5db; }}

    /* ── Totals ── */
    .totals {{ padding: 0 32px 20px; display: flex; justify-content: flex-end; }}
    .totals-box {{ width: 280px; }}
    .totals-row {{ display: flex; justify-content: space-between; padding: 7px 12px;
                   font-size: 13px; border-bottom: 1px solid #e5e7eb; }}
    .totals-row.grand {{ background: #1d4f3e; color: #fff; font-weight: 700; font-size: 14px;
                          border-radius: 0 0 6px 6px; }}

    /* ── Internal split ── */
    .split-section {{ margin: 0 32px 28px; background: #f0fdf4; border: 1px solid #bbf7d0;
                       border-radius: 8px; padding: 16px 20px; }}
    .split-section h3 {{ font-size: 12px; font-weight: 700; color: #166534; text-transform: uppercase;
                          letter-spacing: 1px; margin-bottom: 10px; }}
    table.split {{ width: 100%; border-collapse: collapse; font-size: 12px; }}
    table.split th {{ padding: 7px 10px; text-align: left; color: #4b5563; background: #dcfce7;
                       font-size: 11px; text-transform: uppercase; letter-spacing: .5px; }}

    /* ── Footer ── */
    .footer {{ background: #f9fafb; border-top: 1px solid #e5e7eb; padding: 14px 32px;
               display: flex; justify-content: space-between; align-items: center; font-size: 11px;
               color: #9ca3af; }}
    @media print {{
      body {{ background: #fff; padding: 0; }}
      .page {{ box-shadow: none; }}
    }}
  </style>
</head>
<body>
<div class="page">

  <!-- Brand accent bar -->
  <div class="accent-bar"></div>

  <!-- Header -->
  <div class="header">
    <div class="brand">
      <div class="logo-box"><span>IH</span></div>
      <div>
        <div class="brand-lines">
          Invoicing &amp; Payments<br>
          Team Split Engine<br>
          Automated Reminders<br>
          Freelance Finance
        </div>
        <div class="brand-name">INVOICEHOUND</div>
      </div>
    </div>
    <div class="doc-title">
      <h1>{'Tax Invoice' if inv.paid else 'Invoice'}</h1>
      <table>
        <tr><td>Invoice No:</td><td><strong>{inv.invoice_id}</strong></td></tr>
        <tr><td>Project:</td><td><strong>{inv.project_name}</strong></td></tr>
        <tr><td>Issue Date:</td><td>{inv.issue_date}</td></tr>
        <tr><td>Due Date:</td><td><strong>{inv.due_date}</strong></td></tr>
      </table>
    </div>
  </div>

  <!-- Status bar (dashboard only, not shown on client PDF) -->
  {status_banner_html}

  <!-- Parties -->
  <div class="parties">
    <div class="party">
      <div class="party-label">Bill To</div>
      <div class="party-name">{inv.client_name}</div>
      <div class="party-detail">
        {inv.client_email}<br>
        Payment due: <strong>{inv.due_date}</strong>
      </div>
    </div>
    <div class="party">
      <div class="party-label">From</div>
      <div class="party-name">{inv.project_name} Team</div>
      <div class="party-detail">
        <strong>Powered by InvoiceHound</strong><br>
        Issued: {inv.issue_date}
      </div>
    </div>
  </div>

  <!-- Items -->
  <div class="items-section">
    <div class="section-title">Invoice Items</div>
    <table class="items">
      <thead>
        <tr>
          <th class="num" style="width:52px">Item No.</th>
          <th>Item Description</th>
          <th class="num" style="width:52px">Quantity</th>
          <th style="text-align:right;width:130px">Unit Price (Rs)</th>
          <th style="text-align:right;width:110px">Item Total</th>
        </tr>
      </thead>
      <tbody>{item_rows}</tbody>
      <tfoot>
        <tr>
          <td colspan="3" style="border-color:transparent"></td>
          <td style="text-align:right;color:#6b7280">Subtotal</td>
          <td style="text-align:right;font-weight:600">Rs {subtotal:,.2f}</td>
        </tr>
        <tr>
          <td colspan="3" style="border-color:transparent"></td>
          <td style="text-align:right;color:#6b7280">GST (10%)</td>
          <td style="text-align:right">Rs {gst:,.2f}</td>
        </tr>
      </tfoot>
    </table>
  </div>

  <!-- Grand total -->
  <div class="totals">
    <div class="totals-box">
      <div class="totals-row grand">
        <span>Total Due</span>
        <span>Rs {total:,.2f}</span>
      </div>
    </div>
  </div>

  <!-- Internal split (confidential, internal only) -->
  {internal_split_html}

  <!-- Footer -->
  <div class="footer">
    <span>Thank you for your business.</span>
    <span>Payment is due within {days_until_due} days of invoice date.</span>
  </div>

</div>
</body>
</html>"""

    def generate_member_invoice_html(self, inv: InvoiceRecord, member_name: str) -> str:
        """Generate a personal earnings statement for one team member."""
        data = inv.payment_splits.get(member_name)
        if not data:
            return f"<p>No split data found for {member_name}</p>"

        subtotal = data["amount"]
        gst      = round(subtotal * 0.10, 2)
        total    = round(subtotal + gst, 2)
        total_hours = sum(v["hours"] for v in inv.payment_splits.values())

        return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Earnings Statement — {member_name}</title>
  <style>
    *, *::before, *::after {{ box-sizing: border-box; margin: 0; padding: 0; }}
    body {{ font-family: Arial, Helvetica, sans-serif; font-size: 13px; color: #1f2937;
            background: #f3f4f6; padding: 32px 16px; }}
    .page {{ max-width: 720px; margin: 0 auto; background: #fff;
             box-shadow: 0 4px 24px rgba(0,0,0,.12); }}
    .header {{ display: flex; justify-content: space-between; align-items: flex-start;
               padding: 28px 32px 20px; border-bottom: 3px solid #1d4f3e; }}
    .logo-box {{ width: 56px; height: 56px; background: #1d4f3e; border-radius: 6px;
                 display: flex; align-items: center; justify-content: center;
                 font-size: 18px; font-weight: 900; color: #fff; letter-spacing: -1px; }}
    .brand-name {{ font-size: 16px; font-weight: 800; color: #1d4f3e; margin-top: 4px; }}
    .brand-lines {{ font-size: 10px; color: #6b7280; text-transform: uppercase; letter-spacing: .5px; }}
    .doc-title h1 {{ font-size: 20px; font-weight: 800; color: #1d4f3e; text-align: right;
                     text-transform: uppercase; letter-spacing: 1px; margin-bottom: 8px; }}
    .doc-title table {{ margin-left: auto; font-size: 12px; }}
    .doc-title td {{ padding: 2px 6px; }}
    .doc-title td:first-child {{ color: #6b7280; }}
    .confidential {{ background: #f0fdf4; border: 1px solid #bbf7d0; margin: 0 32px 20px;
                     padding: 10px 16px; border-radius: 6px; font-size: 12px; color: #166534;
                     font-weight: 600; }}
    .member-card {{ margin: 0 32px 20px; border: 1px solid #d1d5db; border-radius: 8px;
                    padding: 18px 20px; display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
    .field-label {{ font-size: 10px; color: #9ca3af; text-transform: uppercase; letter-spacing: 1px; }}
    .field-value {{ font-size: 14px; font-weight: 600; color: #111827; margin-top: 2px; }}
    .items-section {{ padding: 0 32px 20px; }}
    .section-title {{ font-size: 11px; font-weight: 700; color: #6b7280; text-transform: uppercase;
                      letter-spacing: 1px; padding: 0 0 10px; }}
    table.items {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    table.items thead tr {{ background: #1d4f3e; color: #fff; }}
    table.items thead th {{ padding: 9px 12px; font-size: 11px; font-weight: 600; border: 1px solid #166534; }}
    table.items td {{ padding: 9px 12px; border: 1px solid #d1d5db; }}
    .totals {{ padding: 0 32px 20px; display: flex; justify-content: flex-end; }}
    .totals-box {{ width: 260px; border: 1px solid #d1d5db; border-radius: 6px; overflow: hidden; }}
    .totals-row {{ display: flex; justify-content: space-between; padding: 8px 14px;
                   font-size: 13px; border-bottom: 1px solid #e5e7eb; }}
    .totals-row.grand {{ background: #1d4f3e; color: #fff; font-weight: 700; font-size: 14px; border: none; }}
    .footer {{ background: #f9fafb; border-top: 1px solid #e5e7eb; padding: 12px 32px;
               font-size: 11px; color: #9ca3af; display: flex; justify-content: space-between; }}
  </style>
</head>
<body>
<div class="page">
  <!-- Header -->
  <div class="header">
    <div style="display:flex;gap:14px;align-items:center">
      <div class="logo-box">IH</div>
      <div>
        <div class="brand-lines">Freelance Earnings Statement</div>
        <div class="brand-name">INVOICEHOUND</div>
      </div>
    </div>
    <div class="doc-title">
      <h1>Earnings Statement</h1>
      <table>
        <tr><td>Project Invoice:</td><td><strong>{inv.invoice_id}</strong></td></tr>
        <tr><td>Project:</td><td>{inv.project_name}</td></tr>
        <tr><td>Issue Date:</td><td>{inv.issue_date}</td></tr>
        <tr><td>Due Date:</td><td><strong>{inv.due_date}</strong></td></tr>
      </table>
    </div>
  </div>

  <div class="confidential">🔒 CONFIDENTIAL — Internal document only. Do not share with client.</div>

  <!-- Member info -->
  <div class="member-card">
    <div>
      <div class="field-label">Team Member</div>
      <div class="field-value">{member_name}</div>
    </div>
    <div>
      <div class="field-label">Role</div>
      <div class="field-value">{data['role']}</div>
    </div>
    <div>
      <div class="field-label">Hours Worked</div>
      <div class="field-value">{data['hours']}h of {total_hours}h total</div>
    </div>
    <div>
      <div class="field-label">Share Percentage</div>
      <div class="field-value" style="color:#16a34a">{data['percentage']}%</div>
    </div>
  </div>

  <!-- Earnings breakdown -->
  <div class="items-section">
    <div class="section-title">Earnings Breakdown</div>
    <table class="items">
      <thead>
        <tr>
          <th style="text-align:left">Description</th>
          <th style="text-align:center">Your Share</th>
          <th style="text-align:right">Project Total</th>
          <th style="text-align:right">Your Earnings</th>
        </tr>
      </thead>
      <tbody>
        {"".join(
            f"<tr style='background:{'#fff' if i%2==0 else '#f9fafb'}'>"
            f"<td>{item.description}</td>"
            f"<td style='text-align:center'>{data['percentage']}%</td>"
            f"<td style='text-align:right'>{inv.currency} {item.total:,.2f}</td>"
            f"<td style='text-align:right;font-weight:600'>{inv.currency} {round(item.total * data['percentage']/100, 2):,.2f}</td>"
            f"</tr>"
            for i, item in enumerate(inv.line_items)
        )}
      </tbody>
    </table>
  </div>

  <!-- Totals -->
  <div class="totals">
    <div class="totals-box">
      <div class="totals-row"><span>Subtotal</span><span>{inv.currency} {subtotal:,.2f}</span></div>
      <div class="totals-row"><span>GST (10%)</span><span>{inv.currency} {gst:,.2f}</span></div>
      <div class="totals-row grand"><span>Your Total ({inv.currency})</span><span>{inv.currency} {total:,.2f}</span></div>
    </div>
  </div>

  <div class="footer">
    <span>Generated by <strong>InvoiceHound</strong></span>
    <span>Project invoice: {inv.invoice_id} · Client: {inv.client_name}</span>
  </div>
</div>
</body>
</html>"""

    def to_json(self, inv: InvoiceRecord) -> str:
        return json.dumps({
            "invoice_id": inv.invoice_id,
            "project_name": inv.project_name,
            "client_name": inv.client_name,
            "client_email": inv.client_email,
            "issue_date": inv.issue_date,
            "due_date": inv.due_date,
            "total_amount": inv.total_amount,
            "currency": inv.currency,
            "paid": inv.paid,
            "line_items": [
                {"description": i.description, "quantity": i.quantity,
                 "unit_price": i.unit_price, "total": i.total}
                for i in inv.line_items
            ],
            "payment_splits": inv.payment_splits,
        }, indent=2)
