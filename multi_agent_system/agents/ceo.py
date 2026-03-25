"""CEO orchestrator agent.

Every inter-agent message is emitted through MessageBus so the full
send/receive history can be inspected at any time (assignment requirement:
'Show me every message the CEO agent sent and received').
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

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


class CEOAgent:
    def __init__(
        self,
        llm: LLMClient,
        groq_client: GroqClient,
        slack_client: SlackClient,
        github_client: GitHubClient,
        sendgrid_client: SendGridClient,
        slack_channel_id: str,
        launches_channel_id: str,
        output_dir: Path,
        dry_run_actions: bool = True,
        max_revisions: int = 2,
    ) -> None:
        self.llm = llm
        self.groq_client = groq_client
        self.slack_client = slack_client
        self.github_client = github_client
        self.sendgrid_client = sendgrid_client
        self.slack_channel_id = slack_channel_id
        self.launches_channel_id = launches_channel_id
        self.output_dir = output_dir
        self.dry_run_actions = dry_run_actions
        self.max_revisions = max_revisions

        self.logs: List[DecisionLogEntry] = []
        self.bus = MessageBus()   # ← tracks every message CEO sends/receives

        self.product_agent   = ProductAgent(llm)
        self.engineer_agent  = EngineerAgent(llm, github_client=github_client, dry_run=dry_run_actions)
        self.marketing_agent = MarketingAgent(
            llm,
            sendgrid_client=sendgrid_client,
            slack_client=slack_client,
            launches_channel_id=launches_channel_id,
            dry_run=dry_run_actions,
        )
        self.qa_agent = QAAgent(groq_client, github_client=github_client, dry_run=dry_run_actions)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _log(self, stage: str, detail: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.logs.append(DecisionLogEntry(stage=stage, detail=detail, data=data or {}))

    def _build_task(self, startup_idea: str, agent_name: str, task_payload: Dict[str, Any]) -> TaskMessage:
        return TaskMessage(
            task_id=f"{agent_name}_task",
            target_agent=agent_name,
            startup_idea=startup_idea,
            task_brief=str(task_payload.get("task_brief", "")),
            expected_output=list(task_payload.get("expected_output", [])),
            constraints=list(task_payload.get("constraints", [])),
            context={"generated_by": "ceo_llm_decomposition"},
        )

    def _run_with_review(self, task: TaskMessage) -> Dict[str, Any]:
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
            # CEO → agent: task (or revision_request on subsequent rounds)
            msg_type = "task" if round_index == 0 else "revision_request"
            sent = self.bus.send(
                from_agent="ceo",
                to_agent=task.target_agent,
                message_type=msg_type,
                payload={
                    "task_id":   task.task_id,
                    "idea":      task.startup_idea,
                    "brief":     task.task_brief,
                    "expected":  task.expected_output,
                    "constraints": task.constraints,
                    "context":   task.context,
                    "revision_instruction": revision_instruction,
                },
                parent_message_id=parent_id,
            )

            agent_result = agent.run(task, revision_instruction=revision_instruction)
            result = agent_result.output

            # agent → CEO: result
            reply = self.bus.send(
                from_agent=task.target_agent,
                to_agent="ceo",
                message_type="result",
                payload=result,
                parent_message_id=sent["message_id"],
            )
            parent_id = reply["message_id"]

            # CEO reviews
            if task.target_agent == "product":
                decision = self.llm.review_product_spec(task.startup_idea, result)
            else:
                decision = self.llm.review_output(
                    startup_idea=task.startup_idea,
                    task_brief=task.task_brief,
                    agent_name=task.target_agent,
                    agent_output=result,
                )

            self._log(
                stage="review",
                detail=f"{task.target_agent} round {round_index}: acceptable={decision.acceptable}",
                data={
                    "score": decision.score,
                    "rationale": decision.rationale,
                    "follow_up_instruction": decision.follow_up_instruction,
                },
            )

            if decision.acceptable:
                # CEO → agent: confirmation
                self.bus.send(
                    from_agent="ceo",
                    to_agent=task.target_agent,
                    message_type="confirmation",
                    payload={"status": "accepted", "score": decision.score,
                             "rationale": decision.rationale},
                    parent_message_id=reply["message_id"],
                )
                return result

            revision_instruction = decision.follow_up_instruction or (
                "Improve completeness, specificity, and execution detail."
            )

        return result

    def _save_decision_log(self) -> Path:
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = logs_dir / f"ceo_decisions_{ts}.json"
        payload = [
            {"stage": item.stage, "detail": item.detail, "data": item.data}
            for item in self.logs
        ]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    # ------------------------------------------------------------------
    # Main entry point
    # ------------------------------------------------------------------

    def run(self, startup_idea: str, dry_run: bool = False) -> Dict[str, Any]:
        decomposition = self.llm.decompose_startup_idea(startup_idea)
        self._log("decompose", "LLM decomposed startup idea into agent tasks.", decomposition)

        product_task   = self._build_task(startup_idea, "product",   decomposition["product_task"])
        engineer_task  = self._build_task(startup_idea, "engineer",  decomposition["engineer_task"])
        marketing_task = self._build_task(startup_idea, "marketing", decomposition["marketing_task"])

        # ---------- run agents with CEO review loop ----------
        product_output   = self._run_with_review(product_task)
        engineer_task.context["product_spec"] = product_output
        engineer_output  = self._run_with_review(engineer_task)
        marketing_task.context["product_spec"] = product_output
        marketing_task.context["pr_url"]       = engineer_output.get("pr_url", "")
        marketing_output = self._run_with_review(marketing_task)

        outputs = {
            "product":   product_output,
            "engineer":  engineer_output,
            "marketing": marketing_output,
        }

        # ---------- QA ----------
        # CEO → QA: task
        qa_sent = self.bus.send(
            from_agent="ceo",
            to_agent="qa",
            message_type="task",
            payload={"html": engineer_output.get("html", ""),
                     "copy": marketing_output,
                     "product_spec": product_output},
        )
        qa_passed, qa_notes, qa_issues, qa_report = self.qa_agent.run(outputs)
        # QA → CEO: result
        self.bus.send(
            from_agent="qa",
            to_agent="ceo",
            message_type="result",
            payload={"passed": qa_passed, "notes": qa_notes,
                     "issues": qa_issues, "report": qa_report},
            parent_message_id=qa_sent["message_id"],
        )
        self._log("qa", qa_notes, {"issues": qa_issues, "report": qa_report})

        if not qa_passed:
            self._log("qa_recovery",
                      "QA failed — asking engineer for revision.",
                      {"issues": qa_issues})
            # CEO → engineer: revision_request
            rev_sent = self.bus.send(
                from_agent="ceo",
                to_agent="engineer",
                message_type="revision_request",
                payload={"instruction": "Address QA issues: " + "; ".join(qa_issues),
                         "qa_issues": qa_issues},
            )
            engineer_task.context["qa_issues"] = qa_issues
            revised = self.engineer_agent.run(
                engineer_task,
                revision_instruction="Address these QA issues: " + "; ".join(qa_issues),
            )
            outputs["engineer"] = revised.output
            # engineer → CEO: result
            self.bus.send(
                from_agent="engineer",
                to_agent="ceo",
                message_type="result",
                payload=revised.output,
                parent_message_id=rev_sent["message_id"],
            )
            qa_passed, qa_notes, qa_issues, qa_report = self.qa_agent.run(outputs)
            self._log("qa_rerun", qa_notes, {"issues": qa_issues, "report": qa_report})

        # ---------- final summary → Slack ----------
        final_summary = self.llm.summarize_for_slack(
            startup_idea=startup_idea,
            agent_outputs=outputs,
            qa_notes=f"{qa_notes} Issues: {qa_issues}",
        )

        slack_response: Dict[str, Any] = {"ok": False, "reason": "dry_run_or_missing_channel"}
        if not dry_run and self.slack_channel_id:
            slack_response = self.slack_client.post_message(self.slack_channel_id, final_summary)
            self._log("slack", "Posted final summary to Slack.", slack_response)
            self.bus.send(
                from_agent="ceo",
                to_agent="slack_channel",
                message_type="confirmation",
                payload={"summary": final_summary, "slack_response": slack_response},
            )
        else:
            self._log("slack", "Skipped Slack post (dry run or missing SLACK_CHANNEL_ID).")

        # ---------- persist logs ----------
        log_path   = self._save_decision_log()
        msg_path   = self.bus.save(self.output_dir / "logs")

        task_messages = {
            "product":   product_task.__dict__,
            "engineer":  engineer_task.__dict__,
            "marketing": marketing_task.__dict__,
        }

        return {
            "task_messages":       task_messages,
            "agent_outputs":       outputs,
            "qa":                  {"passed": qa_passed, "notes": qa_notes, "issues": qa_issues},
            "qa_report":           qa_report,
            "slack_response":      slack_response,
            "decision_log_path":   str(log_path),
            "message_log_path":    str(msg_path),
            "final_summary_text":  final_summary,
            "all_messages":        self.bus.all_messages(),
            "ceo_messages":        self.bus.ceo_messages(),
        }
