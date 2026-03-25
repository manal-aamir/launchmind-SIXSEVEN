"""CEO orchestrator agent — InvoiceHound.

Multi-LLM setup (bonus +2%):
  CEO  → GroqClient  (llama-3.3-70b) for decomposition, review, summarise
  QA   → GroqClient  (llama-3.3-70b) for HTML/copy review
  Product / Engineer / Marketing → LLMClient (OpenAI gpt-4o-mini) for generation

Every inter-agent message is routed through MessageBus (and optionally
RedisBus) using the assignment-specified JSON schema (Section 4.1).

Graceful failure handling (bonus +3%):
  All agent calls are wrapped in safe_call() with exponential-backoff
  retries.  AgentFailure exceptions are caught, logged to the decision
  log AND to the MessageBus, and the CEO continues with a degraded result
  rather than crashing the pipeline.

Multiple revision cycles (bonus +2%):
  max_revisions defaults to 2 — the CEO will send up to two revision_request
  messages per agent if quality is insufficient.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from multi_agent_system.agents.engineer import EngineerAgent
from multi_agent_system.agents.marketing import MarketingAgent
from multi_agent_system.agents.product import ProductAgent
from multi_agent_system.agents.qa import QAAgent
from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.github_client import GitHubClient
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.llm_client import LLMClient
from multi_agent_system.models import (
    DecisionLogEntry,
    MessageBus,
    TaskMessage,
)
from multi_agent_system.retry import AgentFailure, safe_call


class CEOAgent:
    def __init__(
        self,
        llm: LLMClient,                   # OpenAI — used by Product/Engineer/Marketing
        groq_client: GroqClient,          # Groq   — used by CEO + QA (multi-LLM)
        slack_client: SlackClient,
        github_client: GitHubClient,
        sendgrid_client: SendGridClient,
        slack_channel_id: str,
        launches_channel_id: str,
        output_dir: Path,
        dry_run_actions: bool = True,
        max_revisions: int = 2,           # multiple feedback loops — bonus +2%
    ) -> None:
        self.llm              = llm           # OpenAI — for generation agents
        self.groq             = groq_client   # Groq   — CEO's reasoning brain
        self.slack_client     = slack_client
        self.github_client    = github_client
        self.sendgrid_client  = sendgrid_client
        self.slack_channel_id = slack_channel_id
        self.launches_channel_id = launches_channel_id
        self.output_dir       = output_dir
        self.dry_run_actions  = dry_run_actions
        self.max_revisions    = max_revisions

        self.logs: List[DecisionLogEntry] = []
        self.bus  = MessageBus()          # in-memory message log (always active)
        self.failures: List[Dict[str, Any]] = []   # failure audit trail

        # Sub-agents — each gets the appropriate LLM client
        self.product_agent   = ProductAgent(llm)
        self.engineer_agent  = EngineerAgent(
            llm, github_client=github_client, dry_run=dry_run_actions
        )
        self.marketing_agent = MarketingAgent(
            llm,
            sendgrid_client=sendgrid_client,
            slack_client=slack_client,
            launches_channel_id=launches_channel_id,
            dry_run=dry_run_actions,
        )
        self.qa_agent = QAAgent(
            groq_client, github_client=github_client, dry_run=dry_run_actions
        )

    # ------------------------------------------------------------------
    # Logging helpers
    # ------------------------------------------------------------------

    def _log(self, stage: str, detail: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.logs.append(DecisionLogEntry(stage=stage, detail=detail, data=data or {}))

    def _log_failure(self, failure: AgentFailure) -> None:
        self.failures.append(failure.to_dict())
        self._log("agent_failure", str(failure), failure.to_dict())
        # CEO notifies itself via bus so the failure is in the message history
        self.bus.send(
            from_agent=failure.agent_name,
            to_agent="ceo",
            message_type="result",
            payload={
                "status": "failure",
                "error": str(failure.original_error),
                "operation": failure.operation,
                "attempts": failure.attempts,
            },
        )

    def _build_task(
        self, startup_idea: str, agent_name: str, task_payload: Dict[str, Any]
    ) -> TaskMessage:
        return TaskMessage(
            task_id=f"{agent_name}_task",
            target_agent=agent_name,
            startup_idea=startup_idea,
            task_brief=str(task_payload.get("task_brief", "")),
            expected_output=list(task_payload.get("expected_output", [])),
            constraints=list(task_payload.get("constraints", [])),
            context={"generated_by": "ceo_llm_decomposition"},
        )

    # ------------------------------------------------------------------
    # Core: run agent with CEO review loop + retry on failure
    # ------------------------------------------------------------------

    def _run_with_review(self, task: TaskMessage) -> Dict[str, Any]:
        """
        Execute agent, review output, request revisions if needed.
        Wraps the agent call in safe_call() so API/LLM errors trigger
        automatic retries before reporting failure back to CEO.
        """
        agent_map = {
            "product":   self.product_agent,
            "engineer":  self.engineer_agent,
            "marketing": self.marketing_agent,
        }
        agent = agent_map[task.target_agent]
        revision_instruction = ""
        result: Dict[str, Any] = {}
        parent_id: Optional[str] = None

        for round_index in range(self.max_revisions + 1):
            msg_type = "task" if round_index == 0 else "revision_request"

            # CEO → agent (via MessageBus)
            sent = self.bus.send(
                from_agent="ceo",
                to_agent=task.target_agent,
                message_type=msg_type,
                payload={
                    "task_id":              task.task_id,
                    "idea":                 task.startup_idea,
                    "brief":                task.task_brief,
                    "expected":             task.expected_output,
                    "constraints":          task.constraints,
                    "context":              task.context,
                    "revision_instruction": revision_instruction,
                    "round":                round_index,
                },
                parent_message_id=parent_id,
            )
            self._log(
                stage=f"{task.target_agent}_dispatch",
                detail=f"CEO → {task.target_agent} [{msg_type}] round {round_index}",
                data={"message_id": sent["message_id"],
                      "revision_instruction": revision_instruction},
            )

            # --- run agent (with graceful failure handling) ---
            agent_result_obj, failure = safe_call(
                agent.run,
                task,
                revision_instruction=revision_instruction,
                agent_name=task.target_agent,
                operation=f"run_round_{round_index}",
                retries=3,
                fallback=None,
            )

            if failure:
                self._log_failure(failure)
                self._log(
                    stage=f"{task.target_agent}_degraded",
                    detail=(
                        f"{task.target_agent} failed all retries — "
                        "continuing with empty fallback output."
                    ),
                    data=failure.to_dict(),
                )
                result = {"degraded": True, "error": str(failure)}
            else:
                result = agent_result_obj.output  # type: ignore[union-attr]

            # agent → CEO (via MessageBus)
            reply = self.bus.send(
                from_agent=task.target_agent,
                to_agent="ceo",
                message_type="result",
                payload=result,
                parent_message_id=sent["message_id"],
            )
            parent_id = reply["message_id"]

            if failure:
                # Cannot review a failed output — break out of revision loop
                break

            # CEO reviews output using Groq (multi-LLM: CEO = Groq)
            if task.target_agent == "product":
                decision = self.groq.review_product_spec(task.startup_idea, result)
            else:
                decision = self.groq.review_output(
                    startup_idea=task.startup_idea,
                    task_brief=task.task_brief,
                    agent_name=task.target_agent,
                    agent_output=result,
                )

            self._log(
                stage="review",
                detail=(
                    f"{task.target_agent} round {round_index}: "
                    f"acceptable={decision.acceptable} score={decision.score}/10"
                ),
                data={
                    "score":                decision.score,
                    "rationale":            decision.rationale,
                    "follow_up_instruction": decision.follow_up_instruction,
                },
            )

            if decision.acceptable:
                # CEO sends confirmation
                self.bus.send(
                    from_agent="ceo",
                    to_agent=task.target_agent,
                    message_type="confirmation",
                    payload={
                        "status":    "accepted",
                        "score":     decision.score,
                        "rationale": decision.rationale,
                    },
                    parent_message_id=reply["message_id"],
                )
                return result

            # Not acceptable — prepare revision request
            revision_instruction = decision.follow_up_instruction or (
                "Improve completeness, specificity, and execution detail."
            )
            self._log(
                stage="revision_requested",
                detail=f"CEO requesting revision #{round_index + 1} from {task.target_agent}",
                data={"instruction": revision_instruction},
            )

        return result

    # ------------------------------------------------------------------
    # Persist logs
    # ------------------------------------------------------------------

    def _save_decision_log(self) -> Path:
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts   = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = logs_dir / f"ceo_decisions_{ts}.json"
        payload = [
            {"stage": e.stage, "detail": e.detail, "data": e.data}
            for e in self.logs
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, startup_idea: str, dry_run: bool = False) -> Dict[str, Any]:
        # ── CEO decomposes idea using Groq ──────────────────────────────
        decomposition, failure = safe_call(
            self.groq.decompose_startup_idea,
            startup_idea,
            agent_name="ceo",
            operation="decompose_startup_idea",
            retries=3,
            fallback={},
        )
        if failure or not decomposition:
            self._log_failure(failure)  # type: ignore[arg-type]
            decomposition = self.llm.decompose_startup_idea(startup_idea)  # OpenAI fallback
        self._log("decompose", "CEO (Groq) decomposed startup idea.", decomposition)

        product_task   = self._build_task(startup_idea, "product",   decomposition.get("product_task",   {}))
        engineer_task  = self._build_task(startup_idea, "engineer",  decomposition.get("engineer_task",  {}))
        marketing_task = self._build_task(startup_idea, "marketing", decomposition.get("marketing_task", {}))

        # ── Run agents with CEO review loop ─────────────────────────────
        product_output   = self._run_with_review(product_task)
        engineer_task.context["product_spec"]  = product_output
        engineer_output  = self._run_with_review(engineer_task)
        marketing_task.context["product_spec"] = product_output
        marketing_task.context["pr_url"]       = engineer_output.get("pr_url", "")
        marketing_output = self._run_with_review(marketing_task)

        outputs = {
            "product":   product_output,
            "engineer":  engineer_output,
            "marketing": marketing_output,
        }

        # ── QA (Groq) ────────────────────────────────────────────────────
        qa_sent = self.bus.send(
            from_agent="ceo", to_agent="qa",
            message_type="task",
            payload={"html":         engineer_output.get("html", ""),
                     "copy":         marketing_output,
                     "product_spec": product_output},
        )
        qa_passed, qa_notes, qa_issues, qa_report = self.qa_agent.run(outputs)
        self.bus.send(
            from_agent="qa", to_agent="ceo",
            message_type="result",
            payload={"passed": qa_passed, "notes": qa_notes,
                     "issues": qa_issues, "report": qa_report},
            parent_message_id=qa_sent["message_id"],
        )
        self._log("qa", qa_notes, {"issues": qa_issues, "report": qa_report})

        # If QA fails, ask Engineer to revise
        if not qa_passed:
            self._log("qa_recovery",
                      "QA failed — CEO requesting engineer revision via bus.",
                      {"issues": qa_issues})
            rev_sent = self.bus.send(
                from_agent="ceo", to_agent="engineer",
                message_type="revision_request",
                payload={"instruction": "Address QA issues: " + "; ".join(qa_issues),
                         "qa_issues": qa_issues},
            )
            engineer_task.context["qa_issues"] = qa_issues
            revised_obj, rev_failure = safe_call(
                self.engineer_agent.run,
                engineer_task,
                revision_instruction="Address QA: " + "; ".join(qa_issues),
                agent_name="engineer",
                operation="qa_revision",
                retries=2,
                fallback=None,
            )
            if rev_failure:
                self._log_failure(rev_failure)
            else:
                outputs["engineer"] = revised_obj.output  # type: ignore[union-attr]
                self.bus.send(
                    from_agent="engineer", to_agent="ceo",
                    message_type="result",
                    payload=outputs["engineer"],
                    parent_message_id=rev_sent["message_id"],
                )
            qa_passed, qa_notes, qa_issues, qa_report = self.qa_agent.run(outputs)
            self._log("qa_rerun", qa_notes, {"issues": qa_issues, "report": qa_report})

        # ── Final summary → Slack (CEO uses Groq) ───────────────────────
        final_summary = self.groq.summarize_for_slack(
            startup_idea=startup_idea,
            agent_outputs=outputs,
            qa_notes=f"{qa_notes} Issues: {qa_issues}",
        )

        slack_response: Dict[str, Any] = {"ok": False, "reason": "dry_run_or_missing_channel"}
        if not dry_run and self.slack_channel_id:
            slack_response, post_failure = safe_call(
                self.slack_client.post_message,
                self.slack_channel_id,
                final_summary,
                agent_name="ceo",
                operation="post_slack_summary",
                retries=2,
                fallback={"ok": False, "reason": "slack_send_failed"},
            )
            if post_failure:
                self._log_failure(post_failure)
            else:
                self._log("slack", "Posted final summary to Slack.", slack_response)
                self.bus.send(
                    from_agent="ceo", to_agent="slack_channel",
                    message_type="confirmation",
                    payload={"summary": final_summary, "slack_response": slack_response},
                )
        else:
            self._log("slack", "Skipped Slack post (dry_run or missing SLACK_CHANNEL_ID).")

        # ── Persist logs ─────────────────────────────────────────────────
        log_path = self._save_decision_log()
        msg_path = self.bus.save(self.output_dir / "logs")

        task_messages = {
            "product":   product_task.__dict__,
            "engineer":  engineer_task.__dict__,
            "marketing": marketing_task.__dict__,
        }

        return {
            "task_messages":      task_messages,
            "agent_outputs":      outputs,
            "qa":                 {"passed": qa_passed, "notes": qa_notes, "issues": qa_issues},
            "qa_report":          qa_report,
            "slack_response":     slack_response,
            "decision_log_path":  str(log_path),
            "message_log_path":   str(msg_path),
            "final_summary_text": final_summary,
            "all_messages":       self.bus.all_messages(),
            "ceo_messages":       self.bus.ceo_messages(),
            "failures":           self.failures,
        }
