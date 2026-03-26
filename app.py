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
import re

try:
    from weasyprint import HTML as WeasyprintHTML
    _WEASYPRINT_OK = True
except Exception:
    _WEASYPRINT_OK = False

from flask import Flask, flash, redirect, render_template, request, url_for, Response

# In-memory pipeline state dict: { invoice_id: { ...status fields... } }
_pipeline_states: dict = {}

from agents.ceo_agent import CEOAgent
from multi_agent_system.env_utils import load_dotenv_file
from multi_agent_system.deepseek_client import DeepSeekClient
from multi_agent_system.gemini_client import GeminiClient
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
    Assign product spec features to team members by role similarity,
    then fall back to round-robin so every feature gets an owner.

    Role matching logic:
      - Feature name/description keywords → matched against member role string
      - Keywords for design: ui, ux, design, interface, visual, wireframe, frontend, front-end, css
      - Keywords for dev/eng: backend, dev, engineer, api, database, logic, server, integration, code
      - Keywords for qa: qa, test, review, quality
      - Keywords for marketing: marketing, copy, social, email, campaign
      - Keywords for product/pm: product, pm, strategy, roadmap, spec, research
    """
    features = product_spec.get("features") or product_spec.get("core_features_ranked") or []
    if not team_members or not features:
        return []

    # Build role buckets — map lowercased role keyword → member
    ROLE_KEYWORDS = {
        "design":    ["ui", "ux", "design", "interface", "visual", "wireframe", "frontend", "front-end", "css", "figma"],
        "dev":       ["backend", "developer", "dev", "engineer", "api", "database", "logic", "server", "integration", "code", "fullstack"],
        "qa":        ["qa", "test", "quality", "review", "assurance"],
        "marketing": ["marketing", "copy", "social", "email", "campaign", "growth"],
        "product":   ["product", "pm", "manager", "strategy", "roadmap", "spec", "research", "analyst"],
    }

    def _best_member(feat: dict) -> dict:
        feat_text = (feat.get("name", "") + " " + feat.get("description", "")).lower()
        best = None
        best_score = -1
        for member in team_members:
            role_lower = member.get("role", "").lower()
            score = 0
            for bucket, kws in ROLE_KEYWORDS.items():
                if any(kw in role_lower for kw in kws):
                    score += sum(1 for kw in kws if kw in feat_text)
            if score > best_score:
                best_score = score
                best = member
        return best or team_members[0]

    # Sort members by hours worked descending for tie-breaking
    members_sorted = sorted(team_members, key=lambda m: float(m.get("hours_worked", 0)), reverse=True)

    assignments = []
    # Track how many tasks each member already owns for load-balancing
    load: dict = {m.get("name", i): 0 for i, m in enumerate(members_sorted)}

    for idx, feat in enumerate(features):
        member = _best_member(feat)
        # If tie (score=0 means no match), use the least-loaded member
        if load[member.get("name", 0)] > 0:
            candidate = min(members_sorted, key=lambda m: load[m.get("name", "")])
            if load[candidate.get("name", "")] < load[member.get("name", "")]:
                member = candidate
        load[member.get("name", "")] = load.get(member.get("name", ""), 0) + 1
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
    """Live pipeline/debug page — auto-refreshes every 5 s while running."""
    state = _pipeline_states.get(invoice_id, {})

    def _safe_load_json(path_str: str):
        if not path_str:
            return None
        try:
            p = Path(path_str)
            if p.exists():
                return json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            return None
        return None

    # Merge persisted record fields (source of truth for debug view)
    try:
        rec = _load(invoice_id)
        state.setdefault("project_name", rec.get("project_name", ""))
        state.setdefault("client_name", rec.get("client_name", ""))
        state.setdefault("project_description", rec.get("project_description", ""))
        state["invoice_id"] = invoice_id

        # Base persisted outputs
        if rec.get("pr_url"):
            state["pr_url"] = rec["pr_url"]
        if rec.get("issue_url"):
            state["issue_url"] = rec["issue_url"]
        if rec.get("product_spec"):
            state["product_spec"] = rec["product_spec"]
        if rec.get("task_messages"):
            state["task_messages"] = rec["task_messages"]
        if rec.get("marketing_copy"):
            state["marketing_copy"] = rec["marketing_copy"]
        if rec.get("marketing_recipient"):
            state["marketing_recipient"] = rec["marketing_recipient"]
        if rec.get("ceo_summary_text"):
            state["ceo_summary_text"] = rec["ceo_summary_text"]
        if rec.get("landing_html"):
            state["landing_html"] = rec["landing_html"]
        if rec.get("qa_report"):
            state["qa_report"] = rec["qa_report"]
        if rec.get("qa"):
            state["qa"] = rec["qa"]
        if rec.get("engineer_output"):
            state["engineer_output"] = rec["engineer_output"]

        if rec.get("agent_error"):
            state["agent_error"] = rec["agent_error"]
            state["done"] = True

        # Infer completion even if in-memory state was lost after restart.
        if rec.get("ceo_summary_text") or rec.get("qa_report"):
            state["done"] = True

        # Resolve message log path (new key first, then derive from decision log).
        message_log_path = rec.get("message_log_path", "")
        if not message_log_path and rec.get("agent_log"):
            guess = str(rec["agent_log"]).replace("ceo_decisions_", "message_log_")
            if Path(guess).exists():
                message_log_path = guess

        messages = _safe_load_json(message_log_path) or []
        state["message_bus_history"] = messages

        # Build CEO decomposition block from saved task messages or message log.
        ceo_decompose = {
            "product_task": {"task_brief": ""},
            "engineer_task": {"task_brief": ""},
            "marketing_task": {"task_brief": ""},
        }
        tm = state.get("task_messages", {}) or {}
        for key in ("product", "engineer", "marketing"):
            obj = tm.get(key, {}) if isinstance(tm, dict) else {}
            if isinstance(obj, dict) and obj.get("task_brief"):
                ceo_decompose[f"{key}_task"]["task_brief"] = obj.get("task_brief", "")
        if messages:
            wanted = {"product": "product_task", "engineer": "engineer_task", "marketing": "marketing_task"}
            for m in messages:
                if m.get("from_agent") == "ceo" and m.get("message_type") in {"task", "revision_request"}:
                    to_agent = m.get("to_agent")
                    bucket = wanted.get(to_agent)
                    brief = (m.get("payload") or {}).get("brief", "")
                    if bucket and brief and not ceo_decompose[bucket]["task_brief"]:
                        ceo_decompose[bucket]["task_brief"] = brief
        state["ceo_task_decomposition"] = ceo_decompose

        # Engineer output view model.
        eng_out = state.get("engineer_output", {}) if isinstance(state.get("engineer_output"), dict) else {}
        branch = eng_out.get("branch") or rec.get("branch_name") or "agent-landing-page"
        html_blob = state.get("landing_html") or eng_out.get("html") or ""
        state["engineer_debug"] = {
            "issue_url": state.get("issue_url", ""),
            "pr_url": state.get("pr_url", ""),
            "branch_name": branch,
            "html_preview": str(html_blob)[:500],
        }

        # QA view model (verdict/issues/comments).
        qa_report = state.get("qa_report", {}) if isinstance(state.get("qa_report"), dict) else {}
        html_review = qa_report.get("html_review", {}) if isinstance(qa_report.get("html_review"), dict) else {}
        copy_review = qa_report.get("copy_review", {}) if isinstance(qa_report.get("copy_review"), dict) else {}
        qa_obj = state.get("qa", {}) if isinstance(state.get("qa"), dict) else {}
        qa_passed = qa_obj.get("passed")
        if qa_passed is None:
            qa_passed = (str(html_review.get("verdict", "")).lower() == "pass"
                         and str(copy_review.get("verdict", "")).lower() == "pass")
        issues = list(html_review.get("issues", []) or []) + list(copy_review.get("issues", []) or [])
        comments = list(html_review.get("comments", []) or []) + list(copy_review.get("comments", []) or [])
        state["qa_debug"] = {
            "verdict": "PASS" if qa_passed else "FAIL",
            "issues": issues,
            "comments": comments,
        }

        # CEO review decisions timeline from decision log.
        ceo_review_log = []
        decision_entries = _safe_load_json(rec.get("agent_log", "")) or []
        for entry in decision_entries:
            if not isinstance(entry, dict):
                continue
            if entry.get("stage") != "review":
                continue
            detail = str(entry.get("detail", ""))
            data = entry.get("data", {}) if isinstance(entry.get("data"), dict) else {}
            # detail format: "product round 0: acceptable=True score=9/10"
            m = re.match(r"([a-z_]+)\s+round\s+\d+:\s+acceptable=(True|False)\s+score=(\d+)/10", detail)
            ceo_review_log.append(
                {
                    "agent": m.group(1) if m else detail.split(" ")[0] if detail else "unknown",
                    "score": int(m.group(3)) if m else data.get("score"),
                    "acceptable": (m.group(2) == "True") if m else bool(data.get("score", 0) >= 8),
                    "rationale": data.get("rationale", ""),
                    "follow_up_instruction": data.get("follow_up_instruction", ""),
                }
            )
        state["ceo_review_log"] = ceo_review_log

        # Message bus feed model: show keys only (not full payload).
        feed = []
        for m in messages:
            payload = m.get("payload") if isinstance(m, dict) else {}
            keys = list(payload.keys()) if isinstance(payload, dict) else []
            feed.append(
                {
                    "from_agent": m.get("from_agent", ""),
                    "to_agent": m.get("to_agent", ""),
                    "message_type": m.get("message_type", ""),
                    "timestamp": m.get("timestamp", ""),
                    "payload_keys": keys,
                }
            )
        state["message_feed"] = feed

    except Exception:
        pass

    if state.get("agent_error"):
        state["pipeline_status"] = "error"
    elif state.get("done"):
        state["pipeline_status"] = "complete"
    else:
        state["pipeline_status"] = "running"

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
        try:
            email_copy = groq_client.write_invoice_email(
                project_name=inv.project_name,
                client_name=inv.client_name,
                total_amount=inv.total_amount,
                currency=inv.currency,
                due_date=inv.due_date,
                invoice_id=inv.invoice_id,
                line_items=record["line_items"],
            )
            subject = email_copy.get("subject", f"Invoice {inv.invoice_id}")

            # Plain text body — clean and professional, no inline invoice HTML
            plain_body = (
                f"Dear {inv.client_name},\n\n"
                f"Please find your invoice attached for {inv.project_name}.\n\n"
                f"Invoice ID: {inv.invoice_id}\n"
                f"Total: Rs {inv.total_amount:,.2f}\n"
                f"Due Date: {inv.due_date}\n\n"
                f"Please review the attached PDF and let us know if you have any questions.\n\n"
                f"Best regards,\nThe InvoiceHound Team"
            )

            # Generate PDF from the invoice HTML (no status banner for client-facing PDF)
            html_inv = invoice_engine.generate_html(inv, include_internal_split=False, show_status_banner=False)
            pdf_bytes = None
            if _WEASYPRINT_OK:
                try:
                    pdf_bytes = WeasyprintHTML(string=html_inv).write_pdf()
                    print(f"[PDF] Invoice PDF generated for {inv.invoice_id}")
                except Exception as pdf_err:
                    print(f"[PDF] weasyprint failed: {pdf_err} — sending without attachment")

            client_sg = SendGridClient(
                api_key=env.get("SENDGRID_API_KEY", ""),
                from_email=env.get("SENDGRID_FROM_EMAIL", ""),
                to_email=inv.client_email,
            )
            receipt = client_sg.send_email(
                subject=subject,
                plain_text=plain_body,
                html_text=f"<pre style='font-family:Arial,sans-serif;font-size:14px'>{plain_body}</pre>",
                pdf_bytes=pdf_bytes,
                pdf_filename=f"InvoiceHound_{inv.invoice_id}.pdf",
            ) if execute_actions else {"ok": "dry-run"}
            record["email_sent"] = bool(receipt.get("ok"))
            record["email_receipt"] = receipt
            if not record["email_sent"]:
                record["email_error"] = str(receipt.get("error", "Email send failed"))
                flash(f"Email send failed: {record['email_error']}", "error")
        except Exception as exc:
            record["email_sent"] = False
            record["email_receipt"] = {"ok": False, "error": str(exc)}
            record["email_error"] = str(exc)
            flash(f"Email send failed: {exc}", "error")
        _save(record)

    if do_slack:
        try:
            result = reminder_engine.check_and_send(inv, simulate_days_overdue=1)
            record["slack_reminders"].append("Day 1")
            record["reminder_receipts"] = {"day_1": result}
        except Exception as exc:
            record["reminder_receipts"] = {"day_1": {"ok": False, "error": str(exc)}}
            flash(f"Slack reminder failed: {exc}", "error")
        _save(record)

    if do_agents:
        invoice_id = inv.invoice_id
        _pipeline_states[invoice_id] = {
            "invoice_id":   invoice_id,
            "project_name": inv.project_name,
            "client_name":  inv.client_name,
            "project_description": development_request,
            "done":         False,
            "agent_done":   {},
            "agent_active": None,
        }

        def _run_agents(inv_id: str):
            state = _pipeline_states[inv_id]
            try:
                print(f"[Pipeline:{inv_id}] START")
                print(f"[Pipeline:{inv_id}] Build request: {development_request}")
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
                    output_dir=PROJECT_ROOT,
                    dry_run_actions=not execute_actions,
                    max_revisions=2,
                )
                original_rwr = ceo._run_with_review

                def _tracked_rwr(task):
                    state["agent_active"] = task.target_agent
                    print(
                        f"[Pipeline:{inv_id}] {task.target_agent.upper()} input -> "
                        f"brief={task.task_brief!r}"
                    )
                    result = original_rwr(task)
                    state["agent_done"][task.target_agent] = True
                    state["agent_active"] = None
                    print(f"[Pipeline:{inv_id}] {task.target_agent.upper()} done")
                    return result

                ceo._run_with_review = _tracked_rwr
                state["agent_active"] = "ceo"
                # IMPORTANT: the agent pipeline is always for the startup product (InvoiceHound),
                # not the client's one-off project request.
                idea = (
                    "InvoiceHound — a tool for freelance teams that generates one professional client invoice, "
                    "splits earnings internally by hours worked, and automatically sends Day 1 / Day 7 / Day 14 "
                    "payment reminders (Slack nudges + a formal overdue email with the HTML invoice)."
                )
                result = ceo.run(startup_idea=idea, dry_run=not execute_actions)
                state["agent_done"]["ceo"] = True
                state["task_messages"] = result.get("task_messages", {})

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
                state["ceo_summary_text"] = result.get("final_summary_text", "")
                state["marketing_copy"] = {
                    "tagline": mkt.get("tagline", ""),
                    "landing_description": mkt.get("landing_description", ""),
                    "cold_email": mkt.get("cold_email", {}),
                    "social_posts": mkt.get("social_posts", {}),
                }
                state["marketing_recipient"] = env.get("SENDGRID_TO_EMAIL", "")
                state["landing_html"] = eng.get("html", "")
                state["qa_report"] = result.get("qa_report", {})

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
                rec["message_log_path"] = result.get("message_log_path", "")
                rec["task_messages"] = state.get("task_messages", {})
                rec["product_spec"] = product
                rec["qa"] = result.get("qa", {})
                rec["engineer_output"] = {
                    "issue_url": state["issue_url"],
                    "pr_url": state["pr_url"],
                    "branch": eng.get("branch", "agent-landing-page"),
                }
                rec["team_assignments"] = _build_team_assignments(
                    rec.get("team_members", []), product
                )
                rec["marketing_copy"] = state["marketing_copy"]
                rec["marketing_recipient"] = state["marketing_recipient"]
                rec["ceo_summary_text"] = state["ceo_summary_text"]
                rec["landing_html"] = state["landing_html"]
                rec["qa_report"] = state["qa_report"]
                _save(rec)
                print(f"[Pipeline:{inv_id}] ISSUE: {state['issue_url']}")
                print(f"[Pipeline:{inv_id}] PR: {state['pr_url']}")
                print(f"[Pipeline:{inv_id}] Marketing email_sent={state['email_sent']} slack_ok={state['slack_ok']}")
                print(f"[Pipeline:{inv_id}] QA passed={state['qa_passed']}")
                print(f"[Pipeline:{inv_id}] CEO summary posted={state['slack_summary_ok']}")
                print(f"[Pipeline:{inv_id}] COMPLETE")
            except Exception as exc:
                state["agent_error"] = str(exc)
                state["done"] = True
                print(f"[Pipeline:{inv_id}] ERROR: {exc}")

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
        record.setdefault("team_payout_email_receipts", [])

        # Email each team member their earnings statement as a PDF attachment
        if execute_actions:
            if not _WEASYPRINT_OK:
                record["team_payout_email_receipts"].append(
                    {"ok": False, "error": "weasyprint not available; cannot generate payout PDFs"}
                )
            else:
                for member in inv.team_members:
                    # IMPORTANT: payment_splits keys are created from the exact
                    # `member.name` strings used during invoice creation.
                    # Do not strip/normalize here; otherwise .get(name) won't match.
                    name = member.name or ""
                    to_email = (member.email or "").strip()
                    split = (result.get("splits") or {}).get(name) or {}
                    amount = float(split.get("amount", 0) or 0)
                    if not name or not to_email:
                        continue
                    try:
                        html_member = invoice_engine.generate_member_invoice_html(inv, name)
                        pdf_bytes = WeasyprintHTML(string=html_member).write_pdf()
                        subject = f"Earnings statement — {inv.project_name} ({inv.invoice_id})"
                        plain_body = (
                            f"Dear {name},\n\n"
                            f"Payment has been received for {inv.project_name} (Invoice {inv.invoice_id}).\n\n"
                            f"Your earnings: Rs {amount:,.2f}\n"
                            f"Hours logged: {split.get('hours', member.hours_worked)}\n\n"
                            "Please find your PDF earnings statement attached.\n\n"
                            "Best regards,\nInvoiceHound"
                        )
                        sg = SendGridClient(
                            api_key=env.get("SENDGRID_API_KEY", ""),
                            from_email=env.get("SENDGRID_FROM_EMAIL", ""),
                            to_email=to_email,
                        )
                        file_member = name.strip().replace(" ", "_")
                        receipt = sg.send_email(
                            subject=subject,
                            plain_text=plain_body,
                            html_text=f"<pre style='font-family:Arial,sans-serif;font-size:14px'>{plain_body}</pre>",
                            pdf_bytes=pdf_bytes,
                            pdf_filename=f"Earnings_{inv.invoice_id}_{file_member}.pdf",
                        )
                        record["team_payout_email_receipts"].append(
                            {"member": name, "email": to_email, "amount": amount, "receipt": receipt}
                        )
                    except Exception as exc:
                        record["team_payout_email_receipts"].append(
                            {"member": name, "email": to_email, "amount": amount, "error": str(exc)}
                        )
        _save(record)
        flash(
            "Payment received! Team splits calculated, Slack notified, and payout PDFs emailed to team.",
            "success",
        )
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
        currency=record.get("currency", "PKR"),
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
