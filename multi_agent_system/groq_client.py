"""
Groq LLM client — powers ALL five agents via llama-3.3-70b-versatile.

  CEO       → Groq  (orchestration, decomposition, review, summarise)
  Product   → Groq  (product spec generation)
  Engineer  → Groq  (HTML landing page + GitHub issue/PR text)
  Marketing → Groq  (tagline, copy, cold email, social posts)
  QA        → Groq  (HTML + copy review, PR comments)

Falls back to structured mock outputs when no API key is provided.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from groq import Groq

from multi_agent_system.models import ReviewDecision
from multi_agent_system.deepseek_client import DeepSeekClient
from multi_agent_system.prompts import (
    QA_HTML_ROLE_PROMPT,
    QA_COPY_ROLE_PROMPT,
    CEO_DECOMPOSE_ROLE_PROMPT,
    CEO_REVIEW_ROLE_PROMPT,
    PRODUCT_ROLE_PROMPT,
    ENGINEER_ROLE_PROMPT,
    MARKETING_ROLE_PROMPT,
    compose_system_prompt,
)


class GroqClient:
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile",
        fallback: Optional[DeepSeekClient] = None,
    ) -> None:
        self.model = model
        self.enabled = bool(api_key)
        self._client: Optional[Groq] = Groq(api_key=api_key) if self.enabled else None
        self.fallback = fallback

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        # strip markdown code fences if present
        if "```" in text:
            lines = text.splitlines()
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError(f"No JSON found in Groq response: {text[:200]}")
        return json.loads(text[start: end + 1])

    def _complete_json(
        self,
        role_prompt: str,
        user_prompt: str,
        mock_default: Dict[str, Any],
        temperature: float = 0.2,
    ) -> Dict[str, Any]:
        if not self.enabled or not self._client:
            print(f"[GROQ FALLBACK] Using mock for {role_prompt[:50]}")
            return mock_default

        system_prompt = compose_system_prompt(role_prompt)
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user",   "content": user_prompt},
                ],
                temperature=temperature,
            )
            text = response.choices[0].message.content or ""
            parsed = self._extract_json(text)
            print(f"[GROQ LLM] Successfully called LLM for {role_prompt[:50]}")
            return parsed
        except Exception as e:
            error_msg = str(e)
            print(f"[GROQ ERROR] ⚠️ {error_msg[:200]}")

            if "rate_limit" in error_msg or "429" in error_msg:
                print("[GROQ ERROR] 🚫 Token/rate limit hit — using mock fallback")
                print("[GROQ ERROR] Solution: wait 10 min OR switch agent to different LLM")

            if self.fallback and self.fallback.enabled:
                try:
                    print("[GROQ FALLBACK] Trying DeepSeek fallback...")
                    return self.fallback.complete_json(role_prompt, user_prompt, mock_default)
                except Exception as e2:
                    print(f"[GROQ FALLBACK] DeepSeek also failed: {e2}")

            print(f"[GROQ MOCK] Returning mock for role: {role_prompt[:60]}")
            return mock_default

    # ------------------------------------------------------------------
    # QA-specific methods
    # ------------------------------------------------------------------

    def review_html(
        self, product_spec: Dict[str, Any], html_content: str
    ) -> Dict[str, Any]:
        mock = {
            "verdict": "pass",
            "issues": [],
            "comments": [
                "Headline matches the InvoiceHound value proposition.",
                "All five core features are present in the features section.",
                "Reminder escalation schedule (Day 1 / Day 7 / Day 14) is explicitly mentioned.",
                "CTA 'Start Chasing Invoices Free' is clear and brand-consistent.",
            ],
        }
        return self._complete_json(
            role_prompt=QA_HTML_ROLE_PROMPT,
            user_prompt=(
                f"Product spec:\n{json.dumps(product_spec, indent=2)}\n\n"
                f"HTML to review (first 4000 chars):\n{html_content[:4000]}\n\n"
                "Review this InvoiceHound landing page. Check:\n"
                "1. Does the headline match the value proposition?\n"
                "2. Are all five core features present?\n"
                "3. Is the Day 1 / Day 7 / Day 14 reminder escalation explained?\n"
                "4. Is the CTA strong and brand-consistent?\n\n"
                "Return JSON: { verdict: 'pass'|'fail', issues: [...], comments: [...] }"
            ),
            mock_default=mock,
        )

    def review_copy(self, marketing_copy: Dict[str, Any]) -> Dict[str, Any]:
        mock = {
            "verdict": "pass",
            "issues": [],
            "comments": [
                "Tagline is under 10 words and punchy.",
                "Cold email has a clear call-to-action.",
                "Tone is appropriate — speaks to freelancers frustrated by late payments.",
            ],
        }
        return self._complete_json(
            role_prompt=QA_COPY_ROLE_PROMPT,
            user_prompt=(
                f"Marketing copy JSON:\n{json.dumps(marketing_copy, indent=2)}\n\n"
                "Review the InvoiceHound marketing copy. Check:\n"
                "1. Is the tagline under 10 words and compelling?\n"
                "2. Does the cold email have a clear call to action?\n"
                "3. Is the tone right for freelancers who hate chasing clients?\n\n"
                "Return JSON: { verdict: 'pass'|'fail', issues: [...], comments: [...] }"
            ),
            mock_default=mock,
        )

    def write_invoice_email(
        self,
        project_name: str,
        client_name: str,
        total_amount: float,
        currency: str,
        due_date: str,
        invoice_id: str,
        line_items: list,
    ) -> Dict[str, str]:
        """LLM writes the initial invoice email — nothing hardcoded."""
        items_text = "\n".join(
            f"  - {item.get('description','')}: {currency} {item.get('total', 0):,.2f}"
            for item in line_items
        )
        mock = {
            "subject": f"Invoice {invoice_id} for {project_name} — {currency} {total_amount:,.2f} due {due_date}",
            "body": (
                f"Dear {client_name},\n\n"
                f"Please find attached the invoice for {project_name}.\n\n"
                f"Total due: {currency} {total_amount:,.2f}\nDue date: {due_date}\n\n"
                "Looking forward to your prompt payment.\n\nBest regards,\nThe Team"
            ),
        }
        return self._complete_json(
            role_prompt="You write professional, warm invoice emails. Return ONLY valid JSON.",
            user_prompt=(
                f"Write an invoice email with these details:\n"
                f"Client: {client_name}\n"
                f"Project: {project_name}\n"
                f"Invoice ID: {invoice_id}\n"
                f"Total: {currency} {total_amount:,.2f}\n"
                f"Due date: {due_date}\n"
                f"Line items:\n{items_text}\n\n"
                "Tone: professional but warm. Mention the invoice ID and due date. "
                "Do NOT be aggressive. Keep it under 150 words.\n\n"
                'Return JSON: { "subject": "...", "body": "..." }'
            ),
            mock_default=mock,
        )

    def write_reminder_message(
        self,
        client_name: str,
        project_name: str,
        invoice_id: str,
        total_amount: float,
        currency: str,
        days_overdue: int,
    ) -> Dict[str, str]:
        """LLM writes a contextual reminder — tone escalates with days overdue."""
        tone_map = {
            1: "friendly and informative — invoice just sent, letting them know the total and due date, no pressure",
            7: "firm but professional",
            14: "formal and urgent",
        }
        tone = tone_map.get(days_overdue, "professional")

        if days_overdue == 1:
            mock = {
                "subject": f"Invoice {invoice_id} for {project_name} — just a heads up!",
                "body": (
                    f"Hi {client_name},\n\n"
                    f"Just wanted to share the invoice for {project_name}. "
                    f"Total is {currency} {total_amount:,.2f}, due on the date shown on the invoice. "
                    f"Let us know if you have any questions — happy to help!\n\n"
                    "Best regards,\nThe InvoiceHound Team"
                ),
            }
        else:
            mock = {
                "subject": f"Payment reminder: Invoice {invoice_id} — {days_overdue} day(s) overdue",
                "body": f"Dear {client_name}, your invoice {invoice_id} for {project_name} "
                        f"is {days_overdue} day(s) overdue. Amount: {currency} {total_amount:,.2f}.",
            }

        day1_constraint = (
            "IMPORTANT for Day 1: Do NOT say the invoice is overdue or due today. "
            "Say it was just sent and mention the total amount and due date as a friendly heads-up. "
            "Keep it warm and professional. "
            f"Example tone: 'Just wanted to share the invoice for {project_name}. "
            f"Total is {currency} {total_amount:,.2f}, due on [date]. Let us know if you have any questions!'\n\n"
            if days_overdue == 1 else ""
        )

        return self._complete_json(
            role_prompt="You write payment reminder messages. Return ONLY valid JSON.",
            user_prompt=(
                f"Write a {tone} payment reminder:\n"
                f"Client: {client_name}\n"
                f"Project: {project_name}\n"
                f"Invoice: {invoice_id}\n"
                f"Amount: {currency} {total_amount:,.2f}\n"
                f"Days since invoice: {days_overdue}\n\n"
                + day1_constraint +
                "Keep under 120 words. Do not include payment links.\n"
                'Return JSON: { "subject": "...", "body": "..." }'
            ),
            mock_default=mock,
        )

    # ------------------------------------------------------------------
    # CEO orchestration methods (Groq as CEO's LLM — multi-LLM bonus)
    # ------------------------------------------------------------------

    def decompose_startup_idea(self, startup_idea: str) -> Dict[str, Any]:
        """CEO uses Groq to break the idea into product/engineer/marketing tasks."""
        mock = {
            "product_task": {
                "task_brief": (
                    "Generate the InvoiceHound product spec JSON including: a one-sentence value proposition, "
                    "3 user personas (freelancers/team leads) with named pain points around payment chasing and splits, "
                    "5 core features ranked by priority explicitly covering the Day 1/Day 7/Day 14 reminder "
                    "escalation and the hour-based internal payment split, and 3 user stories in standard "
                    "As a / I want / So that format."
                ),
                "expected_output": [
                    "value_proposition: one sentence describing the product and its users",
                    "personas: 3 user personas each with name, role, and pain_point",
                    "features: 5 core features ranked by priority (1=highest)",
                    "user_stories: 3 stories in As a / I want / So that format",
                ],
                "constraints": [
                    "Must explicitly cover Day 1/Day 7/Day 14 escalation reminders.",
                    "Must explicitly cover internal hour-based split logic.",
                ],
            },
            "engineer_task": {
                "task_brief": (
                    "Build a complete index.html landing page for InvoiceHound and execute real GitHub workflow: "
                    "create issue, branch, commit, and open PR."
                ),
                "expected_output": ["Landing page HTML", "Issue URL", "PR URL"],
                "constraints": [
                    "Branch name must be agent-landing-page.",
                    "Author commit as EngineerAgent <agent@invoicehound.ai>.",
                ],
            },
            "marketing_task": {
                "task_brief": (
                    "Generate launch copy for freelancers, send cold outreach email via SendGrid, and post "
                    "Slack Block Kit launch message to #launches with PR link."
                ),
                "expected_output": [
                    "Tagline under 10 words",
                    "Landing page description",
                    "Cold outreach email",
                    "Three social posts",
                    "Delivery receipts",
                ],
                "constraints": [
                    "Tone must target freelancers frustrated by chasing clients.",
                ],
            },
        }
        return self._complete_json(
            role_prompt=CEO_DECOMPOSE_ROLE_PROMPT,
            user_prompt=(
                f"Startup idea: {startup_idea}\n\n"
                "Return JSON with keys: product_task, engineer_task, marketing_task.\n"
                "Each must include: task_brief (string), expected_output (array), constraints (array).\n"
                "For product_task, expected_output must contain EXACTLY these four items and nothing else:\n"
                "- value_proposition: one sentence describing the product and its users\n"
                "- personas: 3 user personas each with name, role, and pain_point\n"
                "- features: 5 core features ranked by priority (1=highest)\n"
                "- user_stories: 3 stories in As a / I want / So that format\n"
                "Do NOT include wireframes, user journey maps, or UX deliverables in expected_output. "
                "This is a product spec agent, not a UX agent.\n"
                "For product_task, use this exact task_brief:\n"
                "\"Generate the InvoiceHound product spec JSON including: a one-sentence value proposition, "
                "3 user personas (freelancers/team leads) with named pain points around payment chasing and splits, "
                "5 core features ranked by priority explicitly covering the Day 1/Day 7/Day 14 reminder escalation "
                "and the hour-based internal payment split, and 3 user stories in standard As a / I want / So that format.\"\n"
                "For InvoiceHound, enforce explicit reminder escalation (Day 1/7/14) and split-logic requirements.\n"
                "Do NOT change domain/use-case (e.g., never switch to unrelated apps like healthcare, fintech, etc.). "
                "All tasks must stay strictly aligned to the startup idea above."
            ),
            mock_default=mock,
        )

    def review_output(
        self,
        startup_idea: str,
        task_brief: str,
        agent_name: str,
        agent_output: Dict[str, Any],
    ) -> ReviewDecision:
        """CEO uses Groq to review any agent's output."""
        mock_pass = {
            "acceptable": True, "score": 6,
            "rationale": "[MOCK - Groq limit hit] Output returned mock fallback data.",
            "follow_up_instruction": "",
        }
        mock_fail = {
            "acceptable": False, "score": 3,
            "rationale": "[MOCK - Groq limit hit] Cannot review — mock fallback active.",
            "follow_up_instruction": "Retry when Groq token limit resets.",
        }
        mock_default = mock_pass if len(json.dumps(agent_output)) > 180 else mock_fail
        result = self._complete_json(
            role_prompt=CEO_REVIEW_ROLE_PROMPT,
            user_prompt=(
                f"Startup: {startup_idea}\nAgent: {agent_name}\nTask: {task_brief}\n"
                f"Output: {json.dumps(agent_output)[:3000]}\n\n"
                "Return JSON with keys:\n"
                "- acceptable (boolean)\n"
                "- score (integer 1-10)\n"
                "- rationale (string, 1-2 sentences)\n"
                "- follow_up_instruction (string — specific improvement, empty if acceptable)"
            ),
            mock_default=mock_default,
        )
        return ReviewDecision(
            acceptable=bool(result.get("acceptable")),
            score=int(result.get("score", 0)),
            rationale=str(result.get("rationale", "")),
            follow_up_instruction=str(result.get("follow_up_instruction", "")),
        )

    def review_product_spec(
        self, startup_idea: str, product_output: Dict[str, Any]
    ) -> ReviewDecision:
        """CEO uses Groq to review the Product agent's spec for InvoiceHound specifics."""
        mock_default = {
            "acceptable": True, "score": 6,
            "rationale": "[MOCK - Groq limit hit] Cannot review — mock fallback active.",
            "follow_up_instruction": "",
        }
        result = self._complete_json(
            role_prompt=CEO_REVIEW_ROLE_PROMPT,
            user_prompt=(
                f"Startup: {startup_idea}\nProduct spec: {json.dumps(product_output)[:3000]}\n\n"
                "Check ALL of the following. Fail if ANY are missing:\n"
                "1. Does it have at least 3 user personas with named pain points?\n"
                "2. Are there 5 ranked features?\n"
                "3. Does it explicitly mention Day 1/Day 7/Day 14 reminder escalation?\n"
                "4. Does it explicitly mention internal payment splitting by hours?\n"
                "5. Are there at least 3 user stories?\n\n"
                "Return JSON with keys: acceptable (bool), score (1-10), rationale, follow_up_instruction"
            ),
            mock_default=mock_default,
        )
        return ReviewDecision(
            acceptable=bool(result.get("acceptable")),
            score=int(result.get("score", 0)),
            rationale=str(result.get("rationale", "")),
            follow_up_instruction=str(result.get("follow_up_instruction", "")),
        )

    # ------------------------------------------------------------------
    # Product-spec generation (for Agent 2) — Groq only (+2% capability)
    # ------------------------------------------------------------------

    def generate_product_spec(
        self,
        startup_idea: str,
        task_brief: str = "",
        revision_instruction: str = "",
    ) -> Dict[str, Any]:
        """
        Generate the Product agent JSON spec using Groq.

        Required keys (assignment):
          - value_proposition: string
          - personas: [{name, role, pain_point}, ...] (2-3 items)
          - features: [{name, description, priority}, ...] (5 items, priority 1..5)
          - user_stories: ["As a ... I want ... so that ...", ...] (3 items)

        This repo historically used `core_features_ranked`; we also return
        `core_features_ranked` as an alias of `features`.
        """
        mock_features = [
            {
                "name": "Single invoice per client (team-facing)",
                "description": "Generate one professional invoice to the client while keeping team split details internal/confidential.",
                "priority": 1,
            },
            {
                "name": "Hour-based split calculator (internal)",
                "description": "Compute each member’s share proportionally to hours logged and show it as a confidential split panel/statement.",
                "priority": 2,
            },
            {
                "name": "Escalating reminder engine (Day 1 / Day 7 / Day 14)",
                "description": "Send Slack reminders to the team on Day 1 and Day 7, and dispatch a formal email on Day 14 if unpaid.",
                "priority": 3,
            },
            {
                "name": "AI-written reminders with tone control",
                "description": "Write polite → firm → formal messages without hardcoding text, using the business logic for each overdue stage.",
                "priority": 4,
            },
            {
                "name": "Payment notification + internal settlement",
                "description": "When the client pays, distribute the amount to team members based on hours and notify the team on Slack.",
                "priority": 5,
            },
        ]

        mock = {
            "value_proposition": (
                "InvoiceHound helps freelance teams get paid on time by sending one clean invoice and automatically escalating follow-ups—while splitting earnings fairly by hours."
            ),
            "personas": [
                {
                    "name": "Aisha",
                    "role": "Freelance UI/UX designer",
                    "pain_point": "She’s always the one chasing clients for payment, which damages relationships.",
                },
                {
                    "name": "Bilal",
                    "role": "Full-stack developer (micro-agency)",
                    "pain_point": "He spends hours calculating payment splits and explaining them to the team.",
                },
                {
                    "name": "Sara",
                    "role": "Freelance copywriter",
                    "pain_point": "Past reminder messages felt too harsh, so clients ignore them.",
                },
            ],
            "features": mock_features,
            "core_features_ranked": mock_features,
            "user_stories": [
                "As a freelance team lead, I want one client invoice so internal split details stay private.",
                "As a team member, I want to preview my exact cut so I trust the settlement calculation.",
                "As a freelancer, I want automated escalating reminders so I avoid awkward payment conversations.",
            ],
            "confirmation_message": "Product spec generated for InvoiceHound.",
        }

        user_prompt = (
            f"Startup idea:\n{startup_idea}\n\n"
            f"Task brief:\n{task_brief}\n\n"
        )
        if revision_instruction:
            user_prompt += f"Revision instruction from CEO:\n{revision_instruction}\n\n"

        user_prompt += (
            "Generate the product spec JSON with EXACTLY these keys:\n"
            "value_proposition (string)\n"
            "personas (array of 2-3 objects: name, role, pain_point)\n"
            "features (array of exactly 5 objects: name, description, priority (1-5 where 1=highest))\n"
            "user_stories (array of exactly 3 strings in format: 'As a [user], I want to [action] so that [benefit]')\n"
            "confirmation_message (string)\n\n"
            "You are generating the product spec for the startup product named InvoiceHound.\n"
            "Do NOT generate specs for a client's one-off project (e.g., 'Acme Corp website redesign').\n\n"
            "Business requirements you MUST include in features or user stories:\n"
            "- Single invoice to client: generate one professional invoice to send to the client (team split details are internal).\n"
            "- Payment reminder escalation: Day 1 polite Slack reminder, Day 7 firm Slack reminder, Day 14 formal email with full HTML invoice.\n"
            "- Internal payment split logic: split earnings among developers/designers based on logged hours.\n"
        )

        print("[PRODUCT AGENT] Calling LLM for product spec...")
        result = self._complete_json(
            role_prompt=PRODUCT_ROLE_PROMPT,
            user_prompt=user_prompt,
            mock_default=mock,
        )

        if result.get("value_proposition") != mock.get("value_proposition"):
            print("[PRODUCT AGENT] LLM returned real product spec")
        else:
            print("[PRODUCT AGENT] Product spec looks like mock - check GROQ_API_KEY and token quota")

        # Always provide `core_features_ranked` alias for this repo’s older expectations.
        if isinstance(result, dict) and "features" in result and "core_features_ranked" not in result:
            result["core_features_ranked"] = result["features"]
        return result

    # ------------------------------------------------------------------
    # Engineer generation (Agent 3) — Groq-only
    # ------------------------------------------------------------------

    def generate_engineer_assets(
        self,
        startup_idea: str,
        product_spec: Dict[str, Any],
        revision_instruction: str = "",
    ) -> Dict[str, Any]:
        """
        Engineer LLM output contract:
          - html:         full production-quality index.html (inline CSS, all sections)
          - branch_name:  kebab-case branch name (e.g. 'feat/invoicehound-landing-v1')
          - issue_title:  concise GitHub issue title
          - issue_body:   string (LLM-generated, markdown ok)
          - pr_title:     string (LLM-generated)
          - pr_body:      string (LLM-generated, markdown ok)
        """
        mock_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>InvoiceHound — Get Paid Without the Awkward Follow-Up</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; }
    body { font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #0b1220; color: #e5e7eb; line-height: 1.6; }
    a { color: inherit; text-decoration: none; }
    /* NAV */
    nav { display: flex; justify-content: space-between; align-items: center; padding: 18px 40px;
          border-bottom: 1px solid #1e293b; position: sticky; top: 0; background: #0b1220; z-index: 10; }
    .nav-brand { font-size: 20px; font-weight: 800; color: #22c55e; }
    .nav-cta { padding: 9px 20px; border-radius: 8px; background: #22c55e; color: #052e16; font-weight: 700; font-size: 14px; }
    /* HERO */
    .hero { max-width: 820px; margin: 80px auto 60px; padding: 0 24px; text-align: center; }
    .hero h1 { font-size: 52px; font-weight: 900; line-height: 1.15; margin-bottom: 18px;
                background: linear-gradient(135deg,#22c55e,#3b82f6); -webkit-background-clip:text; -webkit-text-fill-color:transparent; }
    .hero p { font-size: 20px; color: #94a3b8; max-width: 580px; margin: 0 auto 32px; }
    .hero-cta { display: inline-block; padding: 16px 36px; border-radius: 10px; background: #22c55e;
                color: #052e16; font-weight: 800; font-size: 17px; transition: opacity .2s; }
    .hero-cta:hover { opacity: .88; }
    /* SECTION */
    section { max-width: 1080px; margin: 0 auto; padding: 60px 24px; }
    h2 { font-size: 30px; font-weight: 800; margin-bottom: 10px; }
    .section-sub { color: #64748b; margin-bottom: 36px; }
    /* FEATURES GRID */
    .grid { display: grid; gap: 20px; grid-template-columns: repeat(auto-fit, minmax(240px, 1fr)); }
    .card { border: 1px solid #1e293b; padding: 24px; border-radius: 14px; background: #111827;
            transition: border-color .2s; }
    .card:hover { border-color: #22c55e; }
    .card-icon { font-size: 28px; margin-bottom: 10px; }
    .card h3 { font-size: 16px; font-weight: 700; margin: 0 0 8px; color: #f1f5f9; }
    .card p { font-size: 13px; color: #64748b; margin: 0; }
    /* REMINDERS */
    .reminder-list { display: flex; flex-direction: column; gap: 16px; }
    .reminder-row { display: flex; align-items: flex-start; gap: 18px; padding: 20px;
                    background: #111827; border-radius: 12px; border-left: 4px solid; }
    .reminder-row.day1 { border-color: #22c55e; }
    .reminder-row.day7 { border-color: #f59e0b; }
    .reminder-row.day14 { border-color: #ef4444; }
    .reminder-day { font-size: 22px; font-weight: 900; min-width: 60px; }
    .reminder-day.d1 { color: #22c55e; }
    .reminder-day.d7 { color: #f59e0b; }
    .reminder-day.d14 { color: #ef4444; }
    .reminder-body h4 { margin: 0 0 4px; font-size: 15px; color: #f1f5f9; }
    .reminder-body p { margin: 0; font-size: 13px; color: #64748b; }
    /* HOW IT WORKS */
    .steps { display: flex; gap: 0; flex-wrap: wrap; }
    .step { flex: 1; min-width: 200px; text-align: center; padding: 24px 16px; position: relative; }
    .step:not(:last-child)::after { content: '→'; position: absolute; right: -12px; top: 50%;
                                     font-size: 24px; color: #334155; transform: translateY(-50%); }
    .step-num { width: 48px; height: 48px; border-radius: 50%; background: #1e293b; border: 2px solid #22c55e;
                display: flex; align-items: center; justify-content: center; margin: 0 auto 12px;
                font-weight: 800; color: #22c55e; font-size: 18px; }
    .step h4 { font-size: 15px; font-weight: 700; margin: 0 0 6px; }
    .step p { font-size: 12px; color: #64748b; margin: 0; }
    /* SPLITS */
    .split-box { background: #111827; border: 1px solid #1e293b; border-radius: 14px; padding: 32px; }
    .split-row { display: flex; justify-content: space-between; align-items: center;
                 padding: 12px 0; border-bottom: 1px solid #1e293b; font-size: 14px; }
    .split-row:last-child { border-bottom: none; }
    .split-label { color: #94a3b8; }
    .split-val { font-weight: 700; color: #22c55e; }
    /* FOOTER */
    footer { text-align: center; padding: 40px 24px; color: #334155; font-size: 13px; border-top: 1px solid #1e293b; }
  </style>
</head>
<body>

<nav>
  <span class="nav-brand">InvoiceHound</span>
  <a class="nav-cta" href="#start">Get Started Free</a>
</nav>

<!-- HERO -->
<div class="hero">
  <h1>Your team did the work.<br>We make sure you get paid.</h1>
  <p>One invoice to the client. Automatic escalating reminders. Fair hour-based splits. Zero awkward conversations.</p>
  <a class="hero-cta" href="#start" id="start">Start Chasing Invoices Free</a>
</div>

<!-- FEATURES -->
<section>
  <h2>Everything your freelance team needs</h2>
  <p class="section-sub">From the moment you finish the project to the moment everyone gets paid.</p>
  <div class="grid">
    <div class="card">
      <div class="card-icon">&#x1F4CB;</div>
      <h3>Single client invoice</h3>
      <p>Generate one professional invoice for the whole team and send it to the client instantly.</p>
    </div>
    <div class="card">
      <div class="card-icon">&#x23F0;</div>
      <h3>Escalating reminders</h3>
      <p>Automated follow-ups on Day 1, Day 7, and Day 14 — each one firmer and more formal.</p>
    </div>
    <div class="card">
      <div class="card-icon">&#x2696;&#xFE0F;</div>
      <h3>Hour-based splits</h3>
      <p>When the client pays, earnings are distributed internally based on each member's logged hours.</p>
    </div>
    <div class="card">
      <div class="card-icon">&#x1F916;</div>
      <h3>AI-written emails</h3>
      <p>Every reminder is written by AI — professional tone that escalates appropriately without burning bridges.</p>
    </div>
    <div class="card">
      <div class="card-icon">&#x1F4E3;</div>
      <h3>Slack notifications</h3>
      <p>Your team gets notified on Slack the moment reminders go out and when payment lands.</p>
    </div>
  </div>
</section>

<!-- REMINDERS -->
<section style="background:#0f172a; max-width:100%; padding: 60px 0;">
  <div style="max-width:1080px;margin:0 auto;padding:0 24px;">
    <h2>Reminder escalation schedule</h2>
    <p class="section-sub">InvoiceHound never lets an invoice go cold.</p>
    <div class="reminder-list">
      <div class="reminder-row day1">
        <div class="reminder-day d1">D1</div>
        <div class="reminder-body">
          <h4>Day 1 — Polite Slack nudge</h4>
          <p>A friendly reminder posted to your Slack channel with the invoice total. Sets the expectation without pressure.</p>
        </div>
      </div>
      <div class="reminder-row day7">
        <div class="reminder-day d7">D7</div>
        <div class="reminder-body">
          <h4>Day 7 — Firm Slack follow-up</h4>
          <p>A firmer message referencing the original due date and outstanding amount. Professional, not aggressive.</p>
        </div>
      </div>
      <div class="reminder-row day14">
        <div class="reminder-day d14">D14</div>
        <div class="reminder-body">
          <h4>Day 14 — Formal email with embedded invoice</h4>
          <p>A formal overdue notice sent via email with the full HTML invoice attached. Appropriate for escalation or legal follow-up.</p>
        </div>
      </div>
    </div>
  </div>
</section>

<!-- HOW IT WORKS -->
<section>
  <h2>How it works</h2>
  <p class="section-sub">Three steps to getting paid without the back-and-forth.</p>
  <div class="steps">
    <div class="step">
      <div class="step-num">1</div>
      <h4>Log your hours</h4>
      <p>Each team member logs hours in the invoice form. InvoiceHound calculates everyone's share.</p>
    </div>
    <div class="step">
      <div class="step-num">2</div>
      <h4>Send the invoice</h4>
      <p>One professional invoice goes to the client. Reminders kick in automatically on schedule.</p>
    </div>
    <div class="step">
      <div class="step-num">3</div>
      <h4>Get paid &amp; split</h4>
      <p>When payment arrives, InvoiceHound distributes earnings by hours. Everyone gets what they earned.</p>
    </div>
  </div>
</section>

<!-- TEAM SPLITS -->
<section>
  <h2>Transparent team splits</h2>
  <p class="section-sub">No awkward payout conversations. Hours in, money out.</p>
  <div class="split-box">
    <div class="split-row"><span class="split-label">Total invoice paid by client</span><span class="split-val">USD 5,000</span></div>
    <div class="split-row"><span class="split-label">Developer (32 hrs / 80 total)</span><span class="split-val">USD 2,000 — 40%</span></div>
    <div class="split-row"><span class="split-label">Designer (24 hrs / 80 total)</span><span class="split-val">USD 1,500 — 30%</span></div>
    <div class="split-row"><span class="split-label">PM (16 hrs / 80 total)</span><span class="split-val">USD 1,000 — 20%</span></div>
    <div class="split-row"><span class="split-label">QA (8 hrs / 80 total)</span><span class="split-val">USD 500 — 10%</span></div>
  </div>
</section>

<footer>
  <strong>InvoiceHound</strong> &mdash; Professional invoicing &amp; automated reminders for freelance teams.
</footer>

</body>
</html>"""

        mock = {
            "html": mock_html,
            "branch_name": "feat/invoicehound-landing-page",
            "issue_title": "Initial landing page",
            "issue_body": (
                "## Task\n"
                "Create the initial InvoiceHound marketing landing page (`index.html`).\n\n"
                "## Requirements\n"
                "- Hero section with headline, subheadline, and CTA button\n"
                "- Features section (at least 5 cards from product spec)\n"
                "- Reminder escalation schedule (Day 1 / Day 7 / Day 14)\n"
                "- How It Works (3-step flow)\n"
                "- Team splits explainer section\n"
                "- Responsive inline CSS, dark theme\n\n"
                "## Acceptance criteria\n"
                "- Opens correctly in any browser\n"
                "- All sections visible without JavaScript\n"
            ),
            "pr_title": "Initial landing page",
            "pr_body": (
                "## Summary\n"
                "- Adds `index.html` — full InvoiceHound marketing landing page\n"
                "- Sections: Hero, Features, Reminder Schedule, How It Works, Team Splits, Footer\n"
                "- Inline CSS, dark theme, responsive grid layout\n\n"
                "## Changes\n"
                "- New file: `index.html`\n\n"
                "## Test plan\n"
                "- [ ] Open `index.html` in browser — all sections render\n"
                "- [ ] Responsive on mobile viewport\n"
                "- [ ] QA agent posts inline review comments on this PR\n"
            ),
        }

        user_prompt = (
            f"Startup idea:\n{startup_idea}\n\n"
            f"Product spec JSON:\n{json.dumps(product_spec, indent=2)[:6000]}\n\n"
        )
        if revision_instruction:
            user_prompt += f"Revision instruction from CEO:\n{revision_instruction}\n\n"

        user_prompt += (
            "Generate a COMPLETE, PRODUCTION-QUALITY landing page and GitHub metadata.\n"
            "Return ONLY valid JSON with exactly these keys:\n"
            "- html: complete index.html (must include: nav, hero with H1+CTA, features grid from spec, "
            "  reminder schedule Day1/Day7/Day14, how-it-works steps, team-splits section, footer; all inline CSS)\n"
            "- branch_name: short kebab-case git branch name starting with 'feat/' (e.g. 'feat/invoicehound-landing-v1')\n"
            "- issue_title: must be exactly 'Initial landing page'\n"
            "- issue_body: detailed GitHub issue description (markdown, include requirements + acceptance criteria)\n"
            "- pr_title: pull request title (keep concise)\n"
            "- pr_body: PR description in markdown (summary, changes, test plan)\n\n"
            "The HTML must reflect the actual product spec features provided above — not generic placeholders.\n"
            "This landing page is for the startup product InvoiceHound — NOT a client-specific project website.\n"
            "The page title and branding must be 'InvoiceHound'. Do not use a client project name.\n"
            "Minimum HTML length: 3000 characters. Include all 6 required sections.\n"
        )

        return self._complete_json(
            role_prompt=ENGINEER_ROLE_PROMPT,
            user_prompt=user_prompt,
            mock_default=mock,
        )

    # ------------------------------------------------------------------
    # Marketing generation (Agent 4) — Groq-only
    # ------------------------------------------------------------------

    def generate_marketing_assets(
        self,
        startup_idea: str,
        product_spec: Dict[str, Any],
        pr_url: str = "",
        revision_instruction: str = "",
    ) -> Dict[str, Any]:
        """
        Marketing agent LLM output contract:
          - tagline: string (< 10 words, compelling)
          - landing_description: string (2-3 sentences)
          - cold_email: { subject, body }
          - social_posts: { twitter, linkedin, instagram }
          - pr_url: echoed back
        """
        mock = {
            "tagline": "Get paid. Without the awkward follow-up.",
            "landing_description": (
                "InvoiceHound handles everything after project delivery. "
                "One invoice goes to the client, reminders escalate automatically, "
                "and team earnings split the moment payment lands."
            ),
            "cold_email": {
                "subject": "Stop chasing client invoices manually",
                "body": (
                    "Hi there,\n\nInvoiceHound helps freelance teams send one professional invoice, "
                    "auto-follow up on late payments, and split earnings fairly by logged hours.\n\n"
                    "Want a quick demo?\n"
                ),
            },
            "social_posts": {
                "twitter": (
                    "You delivered the project on time.\n"
                    "The client loved it.\n"
                    "But now you are stuck sending your fourth payment follow-up instead of starting the next paid job.\n\n"
                    "There is a better way: InvoiceHound runs the entire collection workflow for you.\n"
                    "✨ One professional invoice sent instantly after delivery\n"
                    "⚡ Automated Day 1, Day 7, and Day 14 reminder escalation\n"
                    "💰 Hour-based team split calculation when payment lands\n"
                    "🔔 Real-time alerts so your whole team knows what is due and what is paid\n"
                    "Start your first flow today and stop chasing manually.\n"
                    "#FreelanceLife #GetPaid #InvoiceHound #StartupTools #SaaS"
                ),
                "linkedin": (
                    "A lot has happened since I last posted on LinkedIn...\n\n"
                    "✅ We built InvoiceHound to automate the full post-delivery payment workflow for freelance teams.\n"
                    "✅ One clean invoice now goes to the client while internal split details stay private and organized.\n"
                    "✅ Reminder escalation is now structured: Day 1 polite nudge, Day 7 firmer follow-up, Day 14 formal email.\n"
                    "✅ Team payouts are calculated by logged hours automatically, so no one argues over manual sheets.\n"
                    "✅ Every payment status update is pushed to Slack so the full team sees what is due and what is paid.\n"
                    "✅ AI-written reminders adapt tone by stage so follow-ups stay professional and relationship-safe.\n\n"
                    "And something else exciting is still in progress 👀\n\n"
                    "Building this made one thing obvious: freelancers rarely struggle with the quality of their work; "
                    "they struggle with everything that comes after delivery. We have all seen the same cycle: "
                    "invoice sent, silence, awkward follow-up, delayed cash flow, and stress for everyone waiting on their share. "
                    "For small teams, this delay is not just inconvenient, it can block payroll, planning, and momentum. "
                    "We wanted to remove that invisible tax on creative and technical teams by turning payment follow-up "
                    "into a consistent system rather than a personal confrontation every week.\n\n"
                    "Grateful for the lessons, the people, and the progress so far.\n\n"
                    "#InvoiceHound #Freelance #SaaS #FreelanceTools #StartupLife #GetPaid #ProductLaunch #B2B #Automation #WorkSmart"
                ),
                "instagram": (
                    "Have you ever finished excellent client work and still had to beg for payment week after week 😩\n\n"
                    "You send the invoice. ✅\n"
                    "You wait and refresh your inbox. ⏳\n"
                    "You send a polite check-in. 😬\n"
                    "You wait again and start worrying about team payouts. 😟\n"
                    "You send a firmer follow-up and hope it does not hurt the relationship. 😣\n"
                    "You open your banking app again, still unpaid. 💀\n\n"
                    "That cycle drains time, confidence, and momentum for every freelancer and micro-agency.\n\n"
                    "We built InvoiceHound for exactly this moment.\n\n"
                    "✨ Send one professional invoice to the client in minutes\n"
                    "⚡ Trigger staged reminders automatically on Day 1, Day 7, and Day 14\n"
                    "💰 Split incoming payment fairly across your team by logged hours\n"
                    "🔔 Notify everyone instantly when reminders go out and payment lands\n"
                    "✨ Keep internal split logic private while client communication stays clean\n\n"
                    "No more awkward chasing. No more payment confusion. Build confidently, deliver proudly, and get paid on time. 🙌\n\n"
                    "#FreelancerLife #GetPaid #InvoiceHound #Freelance #SmallBusiness #FreelanceDesigner #FreelanceDev #AgencyLife #ClientWork #InvoicingTips #FreelanceTips #PaidInFull #Automation #WorkSmart #HustleSmart"
                ),
            },
            "pr_url": pr_url,
        }

        user_prompt = (
            "You are marketing InvoiceHound — a freelance team invoicing and payment reminder tool.\n"
            "Do NOT reference the client's project name or their one-off build request. Always refer to the product as InvoiceHound.\n\n"
            f"Startup idea (InvoiceHound):\n{startup_idea}\n\n"
            f"PRODUCT SPEC (InvoiceHound — use the actual feature names, personas, and value proposition below):\n"
            f"{json.dumps(product_spec, indent=2)[:4000]}\n\n"
            f"GitHub PR URL: {pr_url}\n\n"
        )
        if revision_instruction:
            user_prompt += f"Revision instruction from CEO:\n{revision_instruction}\n\n"

        user_prompt += (
            "Generate launch marketing assets for InvoiceHound.\n"
            "Do NOT write copy for a client project. Do NOT use a client's company name.\n"
            "Name the product InvoiceHound and reference its actual features.\n\n"
            "Return ONLY valid JSON with these exact keys:\n"
            "  tagline         — string, under 10 words, punchy, specific to this product\n"
            "  landing_description — string, 3-4 sentences about THIS product's value\n"
            "  cold_email      — object: { subject, body } (120-220 words, plain text, include CTA)\n"
            "  social_posts    — object: { twitter, linkedin, instagram }\n"
            "    twitter  → 80-120 words minimum; start with relatable freelancer pain story (2-3 lines),\n"
            "               then hook/solution reveal, then 3-4 specific feature lines with emojis,\n"
            "               then CTA + 4-5 hashtags\n"
            "    linkedin → 250-350 words minimum; start with 'A lot has happened...' momentum opener,\n"
            "               include 5-7 ✅ bullet points about specific features,\n"
            "               include a teaser line with 👀,\n"
            "               include a personal journey paragraph about freelancer payment pain,\n"
            "               include grateful closing line,\n"
            "               end with 8-10 hashtags on the final line\n"
            "    instagram → 200-280 words minimum; start with pain-point question ending in 😩,\n"
            "               include a 5-6 line emotional sequence with escalating emojis,\n"
            "               include transition line: 'We built [product] for exactly this moment',\n"
            "               include 4-5 feature bullet points with ✨ ⚡ 💰 🔔 emojis,\n"
            "               include empowerment closing line,\n"
            "               end with 12-15 hashtags on the final line\n"
            "  pr_url          — echo back the PR URL provided above\n"
        )

        user_prompt += (
            "Generate completely fresh, original copy. Do not repeat phrases from previous runs. "
            "Use different opening lines, different examples, and different phrasing each time.\n"
        )

        print("[MARKETING AGENT] Calling LLM for marketing assets...")
        result = self._complete_json(
            role_prompt=MARKETING_ROLE_PROMPT,
            user_prompt=user_prompt,
            mock_default=mock,
            temperature=0.85,
        )
        social = result.get("social_posts", {})
        print(f"[MARKETING AGENT] social_posts keys: {list(social.keys())}")
        twitter_len = len(social.get("twitter", ""))
        print(f"[MARKETING AGENT] Twitter post length: {twitter_len} chars")
        if twitter_len > 0 and social.get("twitter") != mock.get("social_posts", {}).get("twitter"):
            print("[MARKETING AGENT] LLM returned fresh marketing copy")
        else:
            print("[MARKETING AGENT] Marketing copy looks like mock — check GROQ_API_KEY and token quota")
        # Ensure pr_url is always present
        if isinstance(result, dict) and "pr_url" not in result:
            result["pr_url"] = pr_url
        return result

    def summarize_for_slack(
        self,
        startup_idea: str,
        agent_outputs: Dict[str, Any],
        qa_notes: str,
    ) -> str:
        """CEO uses Groq to write the final pipeline summary for Slack."""
        if not self.enabled or not self._client:
            pr_url = agent_outputs.get("engineer", {}).get("pr_url", "N/A")
            issue_url = agent_outputs.get("engineer", {}).get("issue_url", "N/A")
            tagline = agent_outputs.get("marketing", {}).get("tagline", "N/A")
            return (
                "*InvoiceHound Pipeline Complete*\n"
                f"- QA: {qa_notes[:240]}\n"
                f"- Tagline: {tagline}\n"
                f"- Issue: {issue_url}\n"
                f"- PR: {pr_url}\n"
                "- Deliverables: product spec, landing page HTML, launch copy, QA review comments"
            )
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system",
                     "content": "You write concise Slack summaries for startup agent pipelines. Use *bold* for emphasis."},
                    {"role": "user",
                     "content": (
                         f"Summarise this InvoiceHound agent pipeline run for a Slack post.\n"
                         f"Startup idea: {startup_idea[:200]}\n"
                         f"QA notes: {qa_notes[:300]}\n"
                         f"PR URL: {agent_outputs.get('engineer', {}).get('pr_url', 'N/A')}\n"
                         f"Tagline: {agent_outputs.get('marketing', {}).get('tagline', 'N/A')}\n\n"
                         "Keep it between 120 and 220 words. Mention GitHub issue + PR, QA result, "
                         "what was built, and one next step."
                     )},
                ],
                temperature=0.4,
            )
            return response.choices[0].message.content or ""
        except Exception:
            return f"*InvoiceHound Pipeline Complete* — QA: {qa_notes[:200]}"

    def generate_pr_comment(self, issue: str) -> str:
        """Generate a specific inline PR comment from a QA issue."""
        mock = f"QA: {issue}"
        if not self.enabled or not self._client:
            return mock
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You write concise GitHub PR review comments. One sentence only."},
                {"role": "user",
                 "content": f"Write a GitHub PR inline comment for this QA issue: {issue}"},
            ],
            temperature=0.3,
        )
        return (response.choices[0].message.content or mock).strip()
