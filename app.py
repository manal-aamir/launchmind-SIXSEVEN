#!/usr/bin/env python3
"""
InvoiceHound Web App
====================
Form-based UI. You enter project + hours, InvoiceHound does everything else:
  - LLM writes the invoice email (Groq — not hardcoded)
  - SendGrid sends it to the client
  - Slack Day 1 / Day 7 / Day 14 reminders
  - CEO agent pipeline → Engineer creates GitHub PR, Marketing posts launch Slack
  - Payment received → auto-splits by hours + Slack notification to team

Run:
    python3 app.py
Then open http://127.0.0.1:5000
"""

import json
import threading
from datetime import date
from pathlib import Path

from flask import Flask, flash, redirect, render_template, request, url_for, Response

# In-memory pipeline state dict: { invoice_id: { ...status fields... } }
_pipeline_states: dict = {}

from multi_agent_system.agents.ceo import CEOAgent
from multi_agent_system.env_utils import load_dotenv_file
from multi_agent_system.deepseek_client import DeepSeekClient
from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.github_client import GitHubClient
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.invoice_engine import InvoiceEngine, LineItem, TeamMember
from multi_agent_system.redis_bus import RedisBus
from multi_agent_system.reminder_engine import ReminderEngine

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------

app = Flask(__name__)
app.secret_key = "invoicehound-secret-2026"

PROJECT_ROOT = Path(__file__).resolve().parent
STORE_DIR = PROJECT_ROOT / "invoice_store"
STORE_DIR.mkdir(exist_ok=True)

env = load_dotenv_file(PROJECT_ROOT / ".env")
config = json.loads((PROJECT_ROOT / "startup_config.json").read_text()) if (PROJECT_ROOT / "startup_config.json").exists() else {}
execute_actions = config.get("execute_actions", False)

# Clients
slack_client    = SlackClient(bot_token=env.get("SLACK_BOT_TOKEN", ""))
sendgrid_client = SendGridClient(
    api_key=env.get("SENDGRID_API_KEY", ""),
    from_email=env.get("SENDGRID_FROM_EMAIL", ""),
    to_email=env.get("SENDGRID_TO_EMAIL", ""),
)
deepseek_client = DeepSeekClient(
    api_key=env.get("DEEPSEEK_API_KEY", ""),
    model=env.get("DEEPSEEK_MODEL", "deepseek-chat"),
)
groq_client = GroqClient(
    api_key=env.get("GROQ_API_KEY", ""),
    model=env.get("GROQ_MODEL", "llama-3.3-70b-versatile"),
    fallback=deepseek_client,
)
github_client = GitHubClient(
    token=env.get("GITHUB_TOKEN", ""),
    repo=env.get("GITHUB_REPO", "manal-aamir/launchmind-SIXSEVEN"),
)
redis_bus = RedisBus(
    host=env.get("REDIS_HOST", "localhost"),
    port=int(env.get("REDIS_PORT", "6379")),
)

invoice_engine = InvoiceEngine()
reminder_engine = ReminderEngine(
    slack_client=slack_client,
    sendgrid_client=sendgrid_client,
    slack_channel_id=env.get("SLACK_CHANNEL_ID", "") or env.get("LAUNCHES_CHANNEL_ID", ""),
    dry_run=not execute_actions,
)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def _save(inv_data: dict) -> None:
    path = STORE_DIR / f"{inv_data['invoice_id']}.json"
    path.write_text(json.dumps(inv_data, indent=2), encoding="utf-8")


def _load(invoice_id: str) -> dict:
    path = STORE_DIR / f"{invoice_id}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _all_invoices() -> list:
    records = []
    for f in sorted(STORE_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        data = json.loads(f.read_text(encoding="utf-8"))
        due = date.fromisoformat(data["due_date"])
        data["days_overdue"] = max(0, (date.today() - due).days) if not data.get("paid") else 0
        records.append(data)
    return records


def _build_team_assignments(team_members: list, product_spec: dict) -> list:
    """
    Create simple task assignments from Product spec features.
    Assignment rule: sort members by hours desc, then round-robin features.
    """
    features = product_spec.get("features") or product_spec.get("core_features_ranked") or []
    if not team_members or not features:
        return []

    members = sorted(team_members, key=lambda m: float(m.get("hours_worked", 0)), reverse=True)
    assignments = []
    for idx, feat in enumerate(features):
        member = members[idx % len(members)]
        assignments.append(
            {
                "member_name": member.get("name", ""),
                "member_role": member.get("role", ""),
                "member_hours": member.get("hours_worked", 0),
                "task_name": feat.get("name", ""),
                "task_description": feat.get("description", ""),
                "priority": feat.get("priority", ""),
            }
        )
    return assignments


def _invoice_to_record(inv):
    """Convert InvoiceRecord object to a plain dict for storage."""
    return {
        "invoice_id":     inv.invoice_id,
        "project_name":   inv.project_name,
        "client_name":    inv.client_name,
        "client_email":   inv.client_email,
        "issue_date":     inv.issue_date,
        "due_date":       inv.due_date,
        "total_amount":   inv.total_amount,
        "currency":       inv.currency,
        "paid":           inv.paid,
        "payment_splits": inv.payment_splits,
        "team_members": [
            {"name": m.name, "email": m.email,
             "hours_worked": m.hours_worked, "role": m.role}
            for m in inv.team_members
        ],
        "line_items": [
            {"description": i.description, "quantity": i.quantity,
             "unit_price": i.unit_price, "total": i.total}
            for i in inv.line_items
        ],
        "email_sent":       False,
        "slack_reminders":  [],
        "pr_url":           "",
        "agent_log":        "",
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    """Default entry goes to the CEO invoice console."""
    return redirect(url_for("new_invoice_page"))


@app.route("/invoice")
def new_invoice_page():
    """Full invoice form (detailed)."""
    return render_template("new_invoice.html")


@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html", invoices=_all_invoices())


@app.route("/pipeline/<invoice_id>")
def pipeline_status(invoice_id):
    """Live pipeline status page — auto-refreshes every 5 s until done."""
    state = _pipeline_states.get(invoice_id, {})
    # Merge persisted record fields (pr_url, issue_url, etc.)
    try:
        rec = _load(invoice_id)
        state.setdefault("project_name", rec.get("project_name", ""))
        state.setdefault("client_name",  rec.get("client_name", ""))
        state["invoice_id"] = invoice_id
        if rec.get("pr_url"):
            state["pr_url"] = rec["pr_url"]
        if rec.get("issue_url"):
            state["issue_url"] = rec["issue_url"]
        if rec.get("agent_error"):
            state["agent_error"] = rec["agent_error"]
            state["done"] = True
    except Exception:
        pass
    state["invoice_id"] = invoice_id
    return render_template("pipeline.html", pipeline=state)


# ── Legacy /submit route (from full invoice form) ────────────────────────────

@app.route("/submit", methods=["POST"])
def submit():
    f = request.form
    development_request = f.get("development_request", "").strip()

    descs  = f.getlist("item_desc[]")
    qtys   = f.getlist("item_qty[]")
    prices = f.getlist("item_price[]")
    items  = [
        LineItem(description=d, quantity=float(q), unit_price=float(p))
        for d, q, p in zip(descs, qtys, prices)
        if d.strip()
    ]
    if not items:
        flash("Add at least one invoice item.", "error")
        return redirect(url_for("new_invoice_page"))

    names  = f.getlist("member_name[]")
    roles  = f.getlist("member_role[]")
    emails = f.getlist("member_email[]")
    hours  = f.getlist("member_hours[]")
    team   = [
        TeamMember(name=n, email=e, hours_worked=float(h), role=r)
        for n, r, e, h in zip(names, roles, emails, hours)
        if n.strip()
    ]
    if not team:
        flash("Add at least one team member.", "error")
        return redirect(url_for("new_invoice_page"))

    inv = invoice_engine.create_invoice(
        project_name=f["project_name"],
        client_name=f["client_name"],
        client_email=f["client_email"],
        team_members=team,
        line_items=items,
        days_until_due=int(f.get("days_due", 14)),
    )
    record = _invoice_to_record(inv)
    record["project_description"] = development_request
    _save(record)

    do_email  = "send_email"  in f
    do_slack  = "post_slack"  in f
    do_agents = "run_agents"  in f

    if do_email:
        email_copy = groq_client.write_invoice_email(
            project_name=inv.project_name,
            client_name=inv.client_name,
            total_amount=inv.total_amount,
            currency=inv.currency,
            due_date=inv.due_date,
            invoice_id=inv.invoice_id,
            line_items=record["line_items"],
        )
        subject  = email_copy.get("subject", f"Invoice {inv.invoice_id}")
        body     = email_copy.get("body", "")
        html_inv = invoice_engine.generate_html(inv, include_internal_split=False)
        html_body = f"<p>{body.replace(chr(10), '<br>')}</p><hr>" + html_inv
        client_sg = SendGridClient(
            api_key=env.get("SENDGRID_API_KEY", ""),
            from_email=env.get("SENDGRID_FROM_EMAIL", ""),
            to_email=inv.client_email,
        )
        receipt = client_sg.send_email(subject=subject, plain_text=body, html_text=html_body) \
                  if execute_actions else {"ok": "dry-run"}
        record["email_sent"]    = True
        record["email_receipt"] = receipt
        _save(record)

    if do_slack:
        result = reminder_engine.check_and_send(inv, simulate_days_overdue=1)
        record["slack_reminders"].append("Day 1")
        record["reminder_receipts"] = {"day_1": result}
        _save(record)

    if do_agents:
        invoice_id = inv.invoice_id
        _pipeline_states[invoice_id] = {
            "invoice_id":   invoice_id,
            "project_name": inv.project_name,
            "client_name":  inv.client_name,
            "done":         False,
            "agent_done":   {},
            "agent_active": None,
        }

        def _run_agents(inv_id: str):
            state = _pipeline_states[inv_id]
            try:
                ceo = CEOAgent(
                    groq_client=groq_client,
                    redis_bus=redis_bus,
                    slack_client=slack_client,
                    github_client=github_client,
                    sendgrid_client=sendgrid_client,
                    slack_channel_id=env.get("SLACK_CHANNEL_ID", ""),
                    launches_channel_id=env.get("LAUNCHES_CHANNEL_ID", ""),
                    output_dir=PROJECT_ROOT,
                    dry_run_actions=not execute_actions,
                    max_revisions=2,
                )
                original_rwr = ceo._run_with_review

                def _tracked_rwr(task):
                    state["agent_active"] = task.target_agent
                    result = original_rwr(task)
                    state["agent_done"][task.target_agent] = True
                    state["agent_active"] = None
                    return result

                ceo._run_with_review = _tracked_rwr
                state["agent_active"] = "ceo"
                base_request = development_request or inv.project_name
                idea = (
                    f"InvoiceHound — Build request: {base_request}. "
                    f"Client: {inv.client_name}."
                )
                result = ceo.run(startup_idea=idea, dry_run=not execute_actions)
                state["agent_done"]["ceo"] = True

                eng = result.get("agent_outputs", {}).get("engineer", {}) or {}
                mkt = result.get("agent_outputs", {}).get("marketing", {}) or {}
                qa  = result.get("qa", {}) or {}
                product = result.get("agent_outputs", {}).get("product", {}) or {}

                # Mark all agents done (QA runs inside ceo.run, not via _run_with_review)
                state["agent_done"]["product"] = True
                state["agent_done"]["engineer"] = True
                state["agent_done"]["marketing"] = True
                state["agent_done"]["qa"] = True

                state["pr_url"]    = eng.get("pr_url", "")
                state["issue_url"] = eng.get("issue_url", "")

                state["email_sent"] = bool(mkt.get("email_receipt", {}).get("ok"))
                state["slack_ok"]   = bool(mkt.get("slack_receipt", {}).get("ok"))

                state["qa_passed"] = qa.get("passed")
                state["slack_summary_ok"] = bool(result.get("slack_response", {}).get("ok"))

                comments_raw = (
                    qa.get("report", {})
                      .get("review_receipt", {})
                      .get("comments", [])
                )
                state["qa_comments"] = [
                    c.get("body", "") for c in comments_raw if isinstance(c, dict)
                ][:2]

                state["done"] = True
                rec = _load(inv_id)
                rec["pr_url"]    = state["pr_url"]
                rec["issue_url"] = state["issue_url"]
                rec["agent_log"] = result.get("decision_log_path", "")
                rec["product_spec"] = product
                rec["team_assignments"] = _build_team_assignments(
                    rec.get("team_members", []), product
                )
                _save(rec)
            except Exception as exc:
                state["agent_error"] = str(exc)
                state["done"] = True

        threading.Thread(target=_run_agents, args=(invoice_id,), daemon=True).start()
        flash(f"Invoice {inv.invoice_id} created — pipeline running.", "success")
        return redirect(url_for("pipeline_status", invoice_id=inv.invoice_id))

    flash(f"Invoice {inv.invoice_id} created for {inv.client_name}.", "success")
    return redirect(url_for("dashboard"))


@app.route("/remind/<invoice_id>/<int:day>", methods=["POST"])
def remind(invoice_id, day):
    try:
        record = _load(invoice_id)
        inv = _rebuild_invoice(record)
        result = reminder_engine.check_and_send(inv, simulate_days_overdue=day)

        if day == 14:
            # Day 14 uses SendGrid — LLM writes the message
            msg_copy = groq_client.write_reminder_message(
                client_name=record["client_name"],
                project_name=record["project_name"],
                invoice_id=invoice_id,
                total_amount=record["total_amount"],
                currency=record["currency"],
                days_overdue=14,
            )
            html_inv = invoice_engine.generate_html(_rebuild_invoice(record))
            html_body = f"<p>{msg_copy.get('body','').replace(chr(10),'<br>')}</p><hr>" + html_inv
            client_sg = SendGridClient(
                api_key=env.get("SENDGRID_API_KEY", ""),
                from_email=env.get("SENDGRID_FROM_EMAIL", ""),
                to_email=record["client_email"],
            )
            if execute_actions:
                client_sg.send_email(
                    subject=msg_copy.get("subject", f"Overdue: {invoice_id}"),
                    plain_text=msg_copy.get("body", ""),
                    html_text=html_body,
                )

        label = f"Day {day}"
        if label not in record.get("slack_reminders", []):
            record.setdefault("slack_reminders", []).append(label)
        _save(record)
        flash(f"Day {day} reminder triggered for {invoice_id}.", "success")
    except Exception as e:
        flash(f"Reminder failed: {e}", "error")
    return redirect(url_for("dashboard"))


@app.route("/download/<invoice_id>/full")
def download_invoice(invoice_id):
    """Download the full client-facing invoice as HTML."""
    try:
        record = _load(invoice_id)
        inv = _rebuild_invoice(record)
        # Compute splits so the confidential section shows in the HTML
        if not inv.payment_splits and inv.team_members:
            inv.calculate_splits()
        html = invoice_engine.generate_html(inv, include_internal_split=True)
        return Response(
            html,
            mimetype="text/html",
            headers={"Content-Disposition": f"attachment; filename=invoice_{invoice_id}.html"},
        )
    except Exception as e:
        flash(f"Download failed: {e}", "error")
        return redirect(url_for("dashboard"))


@app.route("/download/<invoice_id>/member/<member_name>")
def download_member_invoice(invoice_id, member_name):
    """Download a single team member's earnings statement."""
    try:
        record = _load(invoice_id)
        inv = _rebuild_invoice(record)
        # Compute splits on the fly if not yet stored
        if not inv.payment_splits and inv.team_members:
            inv.calculate_splits()
        html = invoice_engine.generate_member_invoice_html(inv, member_name)
        safe_name = member_name.replace(" ", "_")
        return Response(
            html,
            mimetype="text/html",
            headers={"Content-Disposition": f"attachment; filename=earnings_{safe_name}_{invoice_id}.html"},
        )
    except Exception as e:
        flash(f"Download failed: {e}", "error")
        return redirect(url_for("dashboard"))


@app.route("/paid/<invoice_id>", methods=["POST"])
def mark_paid(invoice_id):
    try:
        record = _load(invoice_id)
        inv = _rebuild_invoice(record)
        result = reminder_engine.distribute_payment(inv)
        record["paid"] = True
        record["payment_splits"] = result["splits"]
        record["split_slack_receipt"] = result["slack_receipt"]
        _save(record)
        flash(f"Payment received! Team splits calculated and posted to Slack.", "success")
    except Exception as e:
        flash(f"Error: {e}", "error")
    return redirect(url_for("dashboard"))


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------

def _rebuild_invoice(record: dict):
    """Reconstruct an InvoiceRecord-like object from stored JSON for the reminder engine."""
    from multi_agent_system.invoice_engine import InvoiceRecord
    items = [
        LineItem(
            description=i["description"],
            quantity=i["quantity"],
            unit_price=i["unit_price"],
        )
        for i in record.get("line_items", [])
    ]
    # Prefer stored team_members; fall back to rebuilding from payment_splits
    raw_team = record.get("team_members") or [
        {"name": name, "email": data.get("email", ""),
         "hours_worked": data.get("hours", 0), "role": data.get("role", "")}
        for name, data in record.get("payment_splits", {}).items()
    ]
    team = [
        TeamMember(
            name=m["name"],
            email=m.get("email", ""),
            hours_worked=float(m.get("hours_worked", 0)),
            role=m.get("role", ""),
        )
        for m in raw_team
    ]
    inv = InvoiceRecord(
        invoice_id=record["invoice_id"],
        project_name=record["project_name"],
        client_name=record["client_name"],
        client_email=record["client_email"],
        issue_date=record["issue_date"],
        due_date=record["due_date"],
        line_items=items,
        team_members=team,
        currency=record.get("currency", "USD"),
        paid=record.get("paid", False),
    )
    inv.payment_splits = record.get("payment_splits", {})
    return inv


@app.route("/messages")
def message_history():
    """Show every inter-agent message (evaluator view: CEO sent/received)."""
    from multi_agent_system.models import MessageBus as _MB
    # Load all message logs from disk
    logs_dir = Path(__file__).parent / "logs"
    all_msgs: list = []
    if logs_dir.exists():
        for f in sorted(logs_dir.glob("message_log_*.json"), reverse=True):
            try:
                all_msgs.extend(json.loads(f.read_text()))
            except Exception:
                pass

    ceo_msgs = [m for m in all_msgs if m.get("from_agent") == "ceo" or m.get("to_agent") == "ceo"]
    return render_template("messages.html", all_messages=all_msgs, ceo_messages=ceo_msgs)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  InvoiceHound 🐕  →  http://127.0.0.1:8080")
    print(f"  Live actions : {'YES — Slack/Email/GitHub active' if execute_actions else 'NO  — dry-run (set execute_actions: true in startup_config.json)'}")
    print("=" * 55)
    app.run(debug=True, port=8080)
