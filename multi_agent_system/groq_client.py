"""
Groq LLM client — used by Agent 1 (CEO) for orchestration reasoning
and Agent 5 (QA / Reviewer) for review.  Using Groq for both CEO and
QA demonstrates multi-LLM integration (bonus +2%):

  CEO  → Groq  llama-3.3-70b  (orchestration, decomposition, review)
  QA   → Groq  llama-3.3-70b  (HTML + copy review, PR comments)
  Product / Engineer / Marketing → OpenAI gpt-4o-mini (generation tasks)

Falls back to structured mock outputs when no API key is provided.
"""

from __future__ import annotations

import json
from typing import Any, Dict, Optional

from groq import Groq

from multi_agent_system.models import ReviewDecision
from multi_agent_system.prompts import (
    QA_HTML_ROLE_PROMPT,
    QA_COPY_ROLE_PROMPT,
    CEO_DECOMPOSE_ROLE_PROMPT,
    CEO_REVIEW_ROLE_PROMPT,
    PRODUCT_ROLE_PROMPT,
    ENGINEER_ROLE_PROMPT,
    compose_system_prompt,
)


class GroqClient:
    def __init__(self, api_key: str, model: str = "llama-3.3-70b-versatile") -> None:
        self.model = model
        self.enabled = bool(api_key)
        self._client: Optional[Groq] = Groq(api_key=api_key) if self.enabled else None

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
    ) -> Dict[str, Any]:
        if not self.enabled or not self._client:
            return mock_default

        system_prompt = compose_system_prompt(role_prompt)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = response.choices[0].message.content or ""
        return self._extract_json(text)

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
            1: "polite and friendly",
            7: "firm but professional",
            14: "formal and urgent",
        }
        tone = tone_map.get(days_overdue, "professional")
        mock = {
            "subject": f"Payment reminder: Invoice {invoice_id} — {days_overdue} day(s) overdue",
            "body": f"Dear {client_name}, your invoice {invoice_id} for {project_name} "
                    f"is {days_overdue} day(s) overdue. Amount: {currency} {total_amount:,.2f}.",
        }
        return self._complete_json(
            role_prompt="You write payment reminder messages. Return ONLY valid JSON.",
            user_prompt=(
                f"Write a {tone} payment reminder:\n"
                f"Client: {client_name}\n"
                f"Project: {project_name}\n"
                f"Invoice: {invoice_id}\n"
                f"Amount: {currency} {total_amount:,.2f}\n"
                f"Days overdue: {days_overdue}\n\n"
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
                    "Define InvoiceHound personas, value proposition, ranked core features, and user stories "
                    "focused on payment reminder escalation and team split fairness."
                ),
                "expected_output": [
                    "Value proposition",
                    "Three personas with pain points",
                    "Five ranked core features",
                    "Three user stories",
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
                "For InvoiceHound, enforce explicit reminder escalation (Day 1/7/14) and split-logic requirements."
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
            "acceptable": True, "score": 8,
            "rationale": "Output is complete and addresses all key requirements.",
            "follow_up_instruction": "",
        }
        mock_fail = {
            "acceptable": False, "score": 4,
            "rationale": "Output is too short or missing important sections.",
            "follow_up_instruction": "Expand concrete deliverables and execution details.",
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
            "acceptable": True, "score": 8,
            "rationale": "Product spec covers escalation logic and split fairness.",
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
            "Business requirements you MUST include in features or user stories:\n"
            "- Payment reminder escalation: Day 1 polite Slack reminder, Day 7 firm Slack reminder, Day 14 formal email with full HTML invoice.\n"
            "- Internal payment split logic: split earnings among developers/designers based on logged hours.\n"
        )

        result = self._complete_json(
            role_prompt=PRODUCT_ROLE_PROMPT,
            user_prompt=user_prompt,
            mock_default=mock,
        )

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
          - html: full working index.html (inline CSS)
          - issue_body: string (LLM-generated)
          - pr_title: string (LLM-generated)
          - pr_body: string (LLM-generated, markdown ok)
        """
        mock_html = """<!doctype html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>InvoiceHound</title>
  <style>
    body { font-family: Arial, sans-serif; margin: 0; background: #0b1220; color: #e5e7eb; }
    .wrap { max-width: 980px; margin: 0 auto; padding: 56px 20px; }
    h1 { font-size: 42px; margin-bottom: 8px; }
    .sub { color: #cbd5e1; margin-bottom: 28px; }
    .cta { display: inline-block; padding: 12px 20px; border-radius: 8px; background: #22c55e; color: #07130a; text-decoration: none; font-weight: 700; }
    .grid { display: grid; gap: 14px; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); margin: 28px 0; }
    .card { border: 1px solid #1f2937; padding: 16px; border-radius: 10px; background: #111827; }
    .flow { line-height: 1.9; margin-top: 18px; }
  </style>
</head>
<body>
  <div class="wrap">
    <h1>Your team did the work. We make sure you get paid.</h1>
    <p class="sub">One invoice to the client. Automatic reminders. Fair splits. Zero awkward conversations.</p>
    <a class="cta" href=\"#start\">Start Chasing Invoices Free</a>
    <h2>Features</h2>
    <div class="grid">
      <div class="card">Single invoice generation for the whole team</div>
      <div class="card">Internal hour-based earnings split calculator</div>
      <div class="card">Escalating reminder engine (Day 1 / Day 7 / Day 14)</div>
      <div class="card">AI-written reminder emails (friendly / firm / formal)</div>
      <div class="card">Team payment notification once client pays</div>
    </div>
    <h2>How it works</h2>
    <p class="flow">Log Hours → Send Invoice → Get Paid</p>
  </div>
</body>
</html>"""

        mock = {
            "html": mock_html,
            "issue_body": (
                "Create the initial landing page for InvoiceHound.\n\n"
                "Requirements:\n"
                "- Headline/subheadline + CTA\n"
                "- Features section from product spec\n"
                "- Basic responsive CSS\n"
            ),
            "pr_title": "Initial landing page",
            "pr_body": (
                "## Summary\n"
                "- Adds `index.html` landing page for InvoiceHound.\n\n"
                "## Test plan\n"
                "- Open index.html in browser\n"
            ),
        }

        user_prompt = (
            f"Startup idea:\n{startup_idea}\n\n"
            f"Product spec JSON:\n{json.dumps(product_spec, indent=2)[:6000]}\n\n"
        )
        if revision_instruction:
            user_prompt += f"Revision instruction from CEO:\n{revision_instruction}\n\n"

        user_prompt += (
            "Generate a complete landing page and GitHub text.\n"
            "Return ONLY JSON with keys:\n"
            "- html: full working index.html (inline CSS)\n"
            "- issue_body: issue description text (no title)\n"
            "- pr_title: pull request title\n"
            "- pr_body: pull request body (markdown)\n\n"
            "Landing page requirements:\n"
            "- Include headline, subheadline, CTA button\n"
            "- Include features section that reflects product spec\n"
            "- Mention reminder escalation (Day 1 / Day 7 / Day 14)\n"
            "- Mention internal hour-based split logic\n"
        )

        return self._complete_json(
            role_prompt=ENGINEER_ROLE_PROMPT,
            user_prompt=user_prompt,
            mock_default=mock,
        )

    def summarize_for_slack(
        self,
        startup_idea: str,
        agent_outputs: Dict[str, Any],
        qa_notes: str,
    ) -> str:
        """CEO uses Groq to write the final pipeline summary for Slack."""
        if not self.enabled or not self._client:
            return (
                f"*InvoiceHound Pipeline Complete*\n"
                f"Idea: {startup_idea[:100]}\n"
                f"QA: {qa_notes[:200]}"
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
                         "Keep it under 200 words. Mention GitHub PR, QA result, and what was built."
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
