"""LLM helper for decomposition, generation, and review."""

import json
from typing import Any, Dict, Optional

from openai import OpenAI

from multi_agent_system.models import ReviewDecision


class LLMClient:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self.model = model
        self.enabled = bool(api_key)
        self._client: Optional[OpenAI] = OpenAI(api_key=api_key) if self.enabled else None

    @staticmethod
    def _extract_json(text: str) -> Dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.strip("`")
            if text.startswith("json"):
                text = text[4:]
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1 or end < start:
            raise ValueError("No JSON object found in model response.")
        return json.loads(text[start : end + 1])

    def _complete_json(
        self, system_prompt: str, user_prompt: str, mock_default: Dict[str, Any]
    ) -> Dict[str, Any]:
        if not self.enabled or not self._client:
            return mock_default

        response = self._client.responses.create(
            model=self.model,
            input=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        text = response.output_text
        return self._extract_json(text)

    def decompose_startup_idea(self, startup_idea: str) -> Dict[str, Any]:
        mock = {
            "product_task": {
                "task_brief": (
                    "Define InvoiceHound personas, value proposition, ranked core features, and user stories "
                    "focused on payment reminder escalation and team split fairness."
                ),
                "expected_output": [
                    "Value proposition",
                    "Three personas",
                    "Five ranked core features",
                    "At least three user stories",
                ],
                "constraints": [
                    "Must explicitly cover Day 1/Day 7/Day 14 escalation reminders.",
                    "Must explicitly cover internal hour-based split logic.",
                ],
            },
            "engineer_task": {
                "task_brief": (
                    "Build a complete index.html landing page for InvoiceHound and execute real GitHub workflow: "
                    "create issue, branch, commit, and PR."
                ),
                "expected_output": [
                    "Landing page HTML",
                    "Issue URL",
                    "PR URL",
                ],
                "constraints": [
                    "Include exact headline, subheadline, CTA, and feature cards from product spec.",
                    "Use branch name agent-landing-page.",
                ],
            },
            "marketing_task": {
                "task_brief": (
                    "Generate launch copy for freelancers, send cold outreach email via SendGrid, and post Slack "
                    "Block Kit launch message to #launches with PR link."
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
            system_prompt=(
                "You are a startup CEO assistant. Return only JSON. "
                "Break startup idea into three actionable tasks for Product, Engineer, Marketing."
            ),
            user_prompt=(
                f"Startup idea: {startup_idea}\n\n"
                "Return JSON with keys: product_task, engineer_task, marketing_task.\n"
                "Each must include task_brief, expected_output (array), constraints (array).\n"
                "For InvoiceHound, enforce explicit reminder escalation and split-logic requirements."
            ),
            mock_default=mock,
        )

    def review_output(
        self, startup_idea: str, task_brief: str, agent_name: str, agent_output: Dict[str, Any]
    ) -> ReviewDecision:
        mock_pass = ReviewDecision(
            acceptable=True,
            score=8,
            rationale="Output is reasonably complete for current stage.",
            follow_up_instruction="",
        )
        mock_fail = ReviewDecision(
            acceptable=False,
            score=5,
            rationale="Output is too short or missing important sections.",
            follow_up_instruction="Expand concrete deliverables, assumptions, and execution details.",
        )
        mock_default = (
            {
                "acceptable": mock_pass.acceptable,
                "score": mock_pass.score,
                "rationale": mock_pass.rationale,
                "follow_up_instruction": mock_pass.follow_up_instruction,
            }
            if len(json.dumps(agent_output)) > 180
            else {
                "acceptable": mock_fail.acceptable,
                "score": mock_fail.score,
                "rationale": mock_fail.rationale,
                "follow_up_instruction": mock_fail.follow_up_instruction,
            }
        )

        result = self._complete_json(
            system_prompt=(
                "You are a strict CEO reviewer. Return only JSON. "
                "Judge quality and completeness for startup execution."
            ),
            user_prompt=(
                f"Startup idea: {startup_idea}\n"
                f"Agent: {agent_name}\n"
                f"Task brief: {task_brief}\n"
                f"Agent output JSON: {json.dumps(agent_output)}\n\n"
                "Return JSON with keys:\n"
                "- acceptable (boolean)\n"
                "- score (integer 1-10)\n"
                "- rationale (string)\n"
                "- follow_up_instruction (string, empty if acceptable)"
            ),
            mock_default=mock_default,
        )
        return ReviewDecision(
            acceptable=bool(result.get("acceptable")),
            score=int(result.get("score", 0)),
            rationale=str(result.get("rationale", "")),
            follow_up_instruction=str(result.get("follow_up_instruction", "")),
        )

    def review_product_spec(self, startup_idea: str, product_output: Dict[str, Any]) -> ReviewDecision:
        mock_default = {
            "acceptable": True,
            "score": 8,
            "rationale": "Product spec is specific and includes escalation and split logic.",
            "follow_up_instruction": "",
        }
        result = self._complete_json(
            system_prompt="You are a strict CEO reviewer. Return only JSON.",
            user_prompt=(
                f"Startup idea: {startup_idea}\n"
                f"Product output JSON: {json.dumps(product_output)}\n\n"
                "Answer this exact check: "
                "Is this product spec specific enough for InvoiceHound? "
                "Does it clearly address payment reminder escalation and internal split logic?\n\n"
                "Return JSON keys: acceptable (bool), score (1-10), rationale (string), "
                "follow_up_instruction (string, empty when acceptable)."
            ),
            mock_default=mock_default,
        )
        return ReviewDecision(
            acceptable=bool(result.get("acceptable")),
            score=int(result.get("score", 0)),
            rationale=str(result.get("rationale", "")),
            follow_up_instruction=str(result.get("follow_up_instruction", "")),
        )

    def generate_landing_page_html(self, startup_idea: str, product_spec: Dict[str, Any]) -> str:
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
    <a class="cta" href="#start">Start Chasing Invoices Free</a>
    <h2>Features</h2>
    <div class="grid">
      <div class="card">Single invoice generation for the whole team</div>
      <div class="card">Internal hour-based earnings split calculator</div>
      <div class="card">Escalating reminder engine (Day 1 / Day 7 / Day 14)</div>
      <div class="card">AI-written reminder emails (friendly / firm / formal)</div>
      <div class="card">Team payment notification once client pays</div>
    </div>
    <h2>How it works</h2>
    <p class="flow">Log Hours -> Send Invoice -> Get Paid</p>
  </div>
</body>
</html>"""
        result = self._complete_json(
            system_prompt="You write production-ready landing pages. Return only JSON.",
            user_prompt=(
                f"Startup idea: {startup_idea}\n"
                f"Product spec: {json.dumps(product_spec)}\n\n"
                "Return JSON with key html. Build full index.html with embedded CSS. Required:\n"
                "- Headline: Your team did the work. We make sure you get paid.\n"
                "- Subheadline: One invoice to the client. Automatic reminders. Fair splits. Zero awkward conversations.\n"
                "- Features section with five cards from spec\n"
                "- How it works: Log Hours -> Send Invoice -> Get Paid\n"
                "- CTA button text: Start Chasing Invoices Free"
            ),
            mock_default={"html": mock_html},
        )
        return str(result.get("html", mock_html))

    def generate_marketing_assets(self, startup_idea: str, product_spec: Dict[str, Any], pr_url: str) -> Dict[str, Any]:
        mock = {
            "tagline": "Get paid. Without the awkward follow-up.",
            "landing_description": (
                "InvoiceHound handles everything after project delivery. One invoice goes to the client, "
                "reminders escalate automatically, and team earnings split the moment payment lands."
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
                "twitter": "Tired of saying 'just following up on that invoice'? InvoiceHound has your back. #freelance",
                "linkedin": "Late payments hurt freelance teams. InvoiceHound automates reminders and protects healthy client communication.",
                "instagram": "No more awkward payment chases. Just deliver work and let InvoiceHound follow up. #freelancer",
            },
            "pr_url": pr_url,
        }
        return self._complete_json(
            system_prompt="You are a startup marketing strategist. Return only JSON.",
            user_prompt=(
                f"Startup idea: {startup_idea}\n"
                f"Product spec: {json.dumps(product_spec)}\n"
                f"PR URL: {pr_url}\n\n"
                "Return JSON with keys: tagline, landing_description, cold_email (subject/body), "
                "social_posts (twitter/linkedin/instagram), pr_url.\n"
                "Tagline must be under 10 words and compelling."
            ),
            mock_default=mock,
        )

    def review_qa_html(self, product_spec: Dict[str, Any], html_content: str) -> Dict[str, Any]:
        mock = {
            "verdict": "pass",
            "issues": [],
            "comments": [
                "Great headline-value alignment.",
                "Reminder escalation is clearly stated.",
            ],
        }
        return self._complete_json(
            system_prompt="You are QA reviewer for startup landing pages. Return only JSON.",
            user_prompt=(
                f"Product spec: {json.dumps(product_spec)}\n"
                f"HTML: {html_content}\n\n"
                "Review this landing page against InvoiceHound spec. Check value proposition, five features, "
                "reminder escalation clarity, and CTA strength.\n"
                "Return JSON with verdict (pass/fail), issues (array), comments (array)."
            ),
            mock_default=mock,
        )

    def review_qa_copy(self, marketing_copy: Dict[str, Any]) -> Dict[str, Any]:
        mock = {
            "verdict": "pass",
            "issues": [],
            "comments": [
                "Tagline is concise and clear.",
                "Cold email includes a strong CTA.",
            ],
        }
        return self._complete_json(
            system_prompt="You are QA reviewer for marketing copy. Return only JSON.",
            user_prompt=(
                f"Marketing copy JSON: {json.dumps(marketing_copy)}\n\n"
                "Review for InvoiceHound: tagline under 10 words and compelling, cold email CTA clarity, "
                "and freelancer-appropriate tone. Return JSON with verdict, issues, comments."
            ),
            mock_default=mock,
        )

    def summarize_for_slack(
        self, startup_idea: str, agent_outputs: Dict[str, Dict[str, Any]], qa_notes: str
    ) -> str:
        mock = {
            "slack_message": (
                f"Startup Idea: {startup_idea}\n\n"
                "CEO reviewed Product, Engineering, and Marketing outputs.\n"
                f"QA verdict: {qa_notes}\n"
                "Status: Review complete."
            )
        }
        result = self._complete_json(
            system_prompt="You write crisp team updates. Return only JSON.",
            user_prompt=(
                f"Startup idea: {startup_idea}\n"
                f"Agent outputs: {json.dumps(agent_outputs)}\n"
                f"QA notes: {qa_notes}\n\n"
                "Return JSON with key slack_message as a concise multi-line status update."
            ),
            mock_default=mock,
        )
        return str(result.get("slack_message", mock["slack_message"]))

