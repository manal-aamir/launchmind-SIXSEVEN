"""Product agent (Agent 2) — forces real LLM, clear fallback logging."""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict

from multi_agent_system.models import AgentResult, TaskMessage


class ProductAgent:
    agent_name = "product"

    def __init__(self, groq_client=None) -> None:
        self.groq = groq_client
        self._anthropic_client = None
        self._groq_direct = None
        self._setup_clients()

    def _setup_clients(self) -> None:
        """Set up LLM clients directly — not through groq_client wrapper."""
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        if anthropic_key:
            try:
                import anthropic
                self._anthropic_client = anthropic.Anthropic(api_key=anthropic_key)
                print("[PRODUCT AGENT] Anthropic client ready")
            except Exception as e:
                print(f"[PRODUCT AGENT] Anthropic setup failed: {e}")

        groq_key = os.environ.get("GROQ_API_KEY", "")
        if groq_key and not self._anthropic_client:
            try:
                from groq import Groq
                self._groq_direct = Groq(api_key=groq_key)
                print("[PRODUCT AGENT] Groq direct client ready")
            except Exception as e:
                print(f"[PRODUCT AGENT] Groq direct setup failed: {e}")

    def _call_llm(self, system_prompt: str, user_prompt: str) -> str:
        """Call LLM directly — tries Anthropic first, then Groq."""
        if self._anthropic_client:
            try:
                print("[PRODUCT AGENT] Calling Anthropic Claude...")
                response = self._anthropic_client.messages.create(
                    model="claude-sonnet-4-20250514",
                    max_tokens=2000,
                    system=system_prompt,
                    messages=[{"role": "user", "content": user_prompt}],
                )
                text = response.content[0].text
                print(f"[PRODUCT AGENT] Anthropic responded ({len(text)} chars)")
                return text
            except Exception as e:
                print(f"[PRODUCT AGENT] Anthropic failed: {e}")

        if self._groq_direct:
            try:
                print("[PRODUCT AGENT] Calling Groq directly...")
                response = self._groq_direct.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    temperature=0.3,
                )
                text = response.choices[0].message.content or ""
                print(f"[PRODUCT AGENT] Groq responded ({len(text)} chars)")
                return text
            except Exception as e:
                err = str(e)
                if "429" in err or "rate_limit" in err:
                    print("[PRODUCT AGENT] Groq rate limit hit — waiting 30s then retrying...")
                    time.sleep(30)
                    try:
                        response = self._groq_direct.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[
                                {"role": "system", "content": system_prompt},
                                {"role": "user", "content": user_prompt},
                            ],
                            temperature=0.3,
                        )
                        text = response.choices[0].message.content or ""
                        print(f"[PRODUCT AGENT] Groq retry succeeded ({len(text)} chars)")
                        return text
                    except Exception as e2:
                        print(f"[PRODUCT AGENT] Groq retry also failed: {e2}")
                else:
                    print(f"[PRODUCT AGENT] Groq failed: {e}")

        print("[PRODUCT AGENT] ALL LLM clients failed — using mock")
        return ""

    def _parse_json(self, text: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response."""
        if not text:
            return {}
        clean = text.strip()
        if "```" in clean:
            lines = [l for l in clean.splitlines() if not l.strip().startswith("```")]
            clean = "\n".join(lines)
        start = clean.find("{")
        end = clean.rfind("}")
        if start == -1 or end == -1:
            print(f"[PRODUCT AGENT] No JSON found in response: {clean[:200]}")
            return {}
        try:
            return json.loads(clean[start:end + 1])
        except json.JSONDecodeError as e:
            print(f"[PRODUCT AGENT] JSON parse error: {e}")
            return {}

    def _generate_with_llm(
        self,
        startup_idea: str,
        task_brief: str,
        revision_instruction: str = "",
    ) -> Dict[str, Any]:
        """Generate real product spec using LLM."""
        system_prompt = (
            "You are the Product Manager agent for InvoiceHound.\n\n"
            "InvoiceHound is a tool for freelance teams that:\n"
            "- Generates ONE professional invoice and sends it to the client\n"
            "- Internally splits earnings among team members by hours logged\n"
            "- Auto-sends escalating payment reminders:\n"
            "  Day 1: polite Slack nudge\n"
            "  Day 7: firm Slack follow-up\n"
            "  Day 14: formal email with HTML invoice embedded\n"
            "- Notifies each team member of their cut once client pays\n\n"
            "Return ONLY valid JSON with EXACTLY these keys:\n"
            "- value_proposition: one sentence mentioning freelance teams and payment\n"
            "- personas: array of exactly 3 objects, each: {name, role, pain_point}\n"
            "  Pain points must relate to: chasing payments, calculating splits, awkward follow-ups\n"
            "- features: array of exactly 5 objects: {name, description, priority}\n"
            "  Priority 1=highest. Must include: single invoice, hour-based split,\n"
            "  Day 1/7/14 reminders, AI-written emails, payment notification\n"
            "- user_stories: array of exactly 3 strings:\n"
            "  Format: 'As a [user], I want to [action] so that [benefit]'\n"
            "- confirmation_message: one sentence confirming spec is ready\n\n"
            "No markdown fences. No extra text. Only the JSON object."
        )
        user_prompt = (
            f"Startup idea:\n{startup_idea}\n\n"
            f"Task brief:\n{task_brief}\n\n"
        )
        if revision_instruction:
            user_prompt += f"CEO revision instruction:\n{revision_instruction}\n\n"
        user_prompt += (
            "Generate the InvoiceHound product spec JSON now.\n"
            "Make the personas feel like real distinct people with specific pain points.\n"
            "Make the user stories specific and actionable.\n"
            "Every feature must clearly relate to InvoiceHound's core value."
        )

        text = self._call_llm(system_prompt, user_prompt)
        result = self._parse_json(text)

        if result and "value_proposition" in result and "personas" in result:
            print("[PRODUCT AGENT] Real LLM product spec generated successfully")
            if "features" in result:
                result["core_features_ranked"] = result["features"]
            return result

        print("[PRODUCT AGENT] LLM response missing required fields — using mock")
        return {}

    def mock_output(self, task: TaskMessage) -> Dict[str, Any]:
        """Clearly labeled mock — only used when ALL LLMs fail."""
        print("[PRODUCT AGENT] RETURNING MOCK OUTPUT — check LLM errors above")
        mock_features = [
            {
                "name": "Single invoice per client",
                "description": "One professional invoice to the client, split details stay internal.",
                "priority": 1,
            },
            {
                "name": "Hour-based internal payment split",
                "description": "Each member's share calculated proportionally to hours logged.",
                "priority": 2,
            },
            {
                "name": "Escalating reminder engine (Day 1 / Day 7 / Day 14)",
                "description": "Slack nudge Day 1, firm Slack Day 7, formal email with invoice Day 14.",
                "priority": 3,
            },
            {
                "name": "AI-written reminders with tone control",
                "description": "Polite to firm to formal messages, no hardcoding.",
                "priority": 4,
            },
            {
                "name": "Payment notification + settlement",
                "description": "Team notified on Slack with individual cuts when client pays.",
                "priority": 5,
            },
        ]
        return {
            "value_proposition": (
                "[MOCK] InvoiceHound helps freelance teams get paid on time "
                "with one invoice, auto reminders, and fair hour-based splits."
            ),
            "personas": [
                {
                    "name": "Aisha [MOCK]",
                    "role": "Freelance UI/UX designer",
                    "pain_point": "Always chasing clients for payment, damaging relationships.",
                },
                {
                    "name": "Bilal [MOCK]",
                    "role": "Full-stack developer",
                    "pain_point": "Spends hours calculating splits and explaining them to teammates.",
                },
                {
                    "name": "Sara [MOCK]",
                    "role": "Freelance copywriter",
                    "pain_point": "Sent reminders that felt too harsh and lost a client.",
                },
            ],
            "features": mock_features,
            "core_features_ranked": mock_features,
            "user_stories": [
                "[MOCK] As a freelance team lead, I want one client invoice so internal split details stay private.",
                "[MOCK] As a team member, I want to preview my exact cut so I trust the settlement calculation.",
                "[MOCK] As a freelancer, I want automated escalating reminders so I avoid awkward payment conversations.",
            ],
            "confirmation_message": "[MOCK FALLBACK] Product spec — LLM unavailable, check terminal for errors.",
        }

    def run(self, task: TaskMessage, revision_instruction: str = "") -> AgentResult:
        print("\n" + "=" * 50)
        print("[PRODUCT AGENT] Starting run...")
        print(f"[PRODUCT AGENT] startup_idea: {task.startup_idea[:80]}...")
        print(f"[PRODUCT AGENT] task_brief: {task.task_brief[:80]}...")
        print("=" * 50)

        output = self._generate_with_llm(
            startup_idea=task.startup_idea,
            task_brief=task.task_brief,
            revision_instruction=revision_instruction,
        )

        if not output:
            output = self.mock_output(task)

        source = "MOCK" if "[MOCK]" in str(output.get("value_proposition", "")) else "REAL LLM"
        print(f"[PRODUCT AGENT] Output source: {source}")
        print(f"[PRODUCT AGENT] value_proposition: {output.get('value_proposition', '')[:80]}")

        return AgentResult(
            agent_name=self.agent_name,
            task_id=task.task_id,
            output=output,
            revision_round=1 if revision_instruction else 0,
        )
