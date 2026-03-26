"""CEO orchestrator agent — InvoiceHound.

This is the canonical CEO agent implementation for the assignment-required
`agents/` folder. Other modules may import via `multi_agent_system.agents`
which re-exports these classes for backward compatibility.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional

from agents.engineer_agent import EngineerAgent
from agents.marketing_agent import MarketingAgent
from agents.product_agent import ProductAgent
from agents.qa_agent import QAAgent
from multi_agent_system.deepseek_client import DeepSeekClient
from multi_agent_system.gemini_client import GeminiClient
from multi_agent_system.groq_client import GroqClient
from multi_agent_system.integrations.github_client import GitHubClient
from multi_agent_system.integrations.sendgrid_client import SendGridClient
from multi_agent_system.integrations.slack_client import SlackClient
from multi_agent_system.models import DecisionLogEntry, MessageBus, TaskMessage
from multi_agent_system.redis_bus import RedisBus
from multi_agent_system.retry import AgentFailure, safe_call


class CEOAgent:
    def __init__(
        self,
        groq_client: GroqClient,
        slack_client: SlackClient,
        github_client: GitHubClient,
        sendgrid_client: SendGridClient,
        slack_channel_id: str,
        launches_channel_id: str,
        output_dir: Path,
        dry_run_actions: bool = True,
        max_revisions: int = 2,
        redis_bus: Optional[RedisBus] = None,
        deepseek_client: Optional[DeepSeekClient] = None,
        gemini_client: Optional[GeminiClient] = None,
    ) -> None:
        self.groq = groq_client
        self.redis_bus = redis_bus
        self.slack_client = slack_client
        self.github_client = github_client
        self.sendgrid_client = sendgrid_client
        self.slack_channel_id = slack_channel_id
        self.launches_channel_id = launches_channel_id
        self.output_dir = output_dir
        self.dry_run_actions = dry_run_actions
        self.max_revisions = max_revisions

        self.logs: List[DecisionLogEntry] = []
        self.bus = MessageBus()
        self.failures: List[Dict[str, Any]] = []
        self._bus_lock = threading.Lock()

        self._redis_stop_event: Optional[threading.Event] = None
        self._redis_threads: List[threading.Thread] = []

        self.product_agent = ProductAgent(
            groq_client,
            deepseek_client=deepseek_client,
            gemini_client=gemini_client,
        )
        self.engineer_agent = EngineerAgent(
            groq_client,
            github_client=github_client,
            dry_run=dry_run_actions,
        )
        self.marketing_agent = MarketingAgent(
            groq_client,
            sendgrid_client=sendgrid_client,
            slack_client=slack_client,
            launches_channel_id=launches_channel_id,
            dry_run=dry_run_actions,
        )
        self.qa_agent = QAAgent(
            groq_client,
            github_client=github_client,
            dry_run=dry_run_actions,
        )

    def _bus_send(self, **kwargs: Any) -> Dict[str, Any]:
        with self._bus_lock:
            msg = self.bus.send(**kwargs)
        self._publish_to_redis(str(msg.get("to_agent", "")), msg)
        return msg

    def _log(self, stage: str, detail: str, data: Optional[Dict[str, Any]] = None) -> None:
        self.logs.append(DecisionLogEntry(stage=stage, detail=detail, data=data or {}))

    def _log_failure(self, failure: AgentFailure) -> None:
        self.failures.append(failure.to_dict())
        self._log("agent_failure", str(failure), failure.to_dict())
        self._bus_send(
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

    def _redis_enabled(self) -> bool:
        return bool(self.redis_bus and getattr(self.redis_bus, "is_redis", False))

    def _publish_to_redis(self, to_agent: str, message: Dict[str, Any]) -> None:
        if not self._redis_enabled():
            return
        self.redis_bus.publish(to_agent=to_agent, message=message)

    def _wait_for_reply(
        self,
        *,
        from_agent: str,
        parent_message_id: str,
        timeout_seconds: float = 120.0,
    ) -> Dict[str, Any]:
        if not self._redis_enabled():
            raise RuntimeError("Redis is not enabled")

        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            for msg in self.redis_bus.listen("ceo", timeout=1.0):
                if (
                    msg.get("from_agent") == from_agent
                    and msg.get("parent_message_id") == parent_message_id
                    and msg.get("message_type") == "result"
                ):
                    return msg
        raise TimeoutError(
            f"Timed out waiting for Redis reply from {from_agent} "
            f"(parent_message_id={parent_message_id})"
        )

    def _redis_agent_worker(self, agent_name: str, agent_obj: Any) -> None:
        assert self._redis_enabled()
        assert self._redis_stop_event is not None

        while not self._redis_stop_event.is_set():
            for msg in self.redis_bus.listen(agent_name, timeout=1.0):
                if self._redis_stop_event.is_set():
                    break

                if msg.get("message_type") not in {"task", "revision_request"}:
                    continue

                payload = msg.get("payload") or {}
                revision_instruction = str(payload.get("revision_instruction", ""))

                if agent_name == "qa":
                    outputs = payload.get("outputs", {}) or {}
                    qa_obj, failure = safe_call(
                        self.qa_agent.run,
                        outputs,
                        agent_name="qa",
                        operation="qa_run",
                        retries=3,
                        fallback=None,
                    )
                    if failure:
                        result_payload = {"degraded": True, "error": str(failure)}
                    else:
                        passed, notes, issues, report = qa_obj
                        result_payload = {"passed": passed, "notes": notes, "issues": issues, "report": report}
                else:
                    task = TaskMessage(
                        task_id=str(payload.get("task_id", f"{agent_name}_task")),
                        target_agent=agent_name,
                        startup_idea=str(payload.get("idea", "")),
                        task_brief=str(payload.get("brief", "")),
                        expected_output=list(payload.get("expected", [])),
                        constraints=list(payload.get("constraints", [])),
                        context=dict(payload.get("context", {})),
                    )

                    agent_result_obj, failure = safe_call(
                        agent_obj.run,
                        task,
                        revision_instruction=revision_instruction,
                        agent_name=agent_name,
                        operation="agent_run",
                        retries=3,
                        fallback=None,
                    )
                    if failure:
                        result_payload = {"degraded": True, "error": str(failure)}
                    else:
                        result_payload = agent_result_obj.output

                self._bus_send(
                    from_agent=agent_name,
                    to_agent="ceo",
                    message_type="result",
                    payload=result_payload,
                    parent_message_id=msg.get("message_id"),
                )

    def _start_redis_workers(self) -> None:
        if not self._redis_enabled():
            return
        if self._redis_stop_event is not None:
            return

        self._redis_stop_event = threading.Event()
        agent_map = {
            "product": self.product_agent,
            "engineer": self.engineer_agent,
            "marketing": self.marketing_agent,
            "qa": self.qa_agent,
        }
        for agent_name, agent_obj in agent_map.items():
            t = threading.Thread(
                target=self._redis_agent_worker,
                args=(agent_name, agent_obj),
                daemon=True,
            )
            t.start()
            self._redis_threads.append(t)

    def _stop_redis_workers(self) -> None:
        if self._redis_stop_event is not None:
            self._redis_stop_event.set()

    def _run_with_review_via_redis(self, task: TaskMessage) -> Dict[str, Any]:
        self._start_redis_workers()

        revision_instruction = ""
        result: Dict[str, Any] = {}
        parent_id: Optional[str] = None

        for round_index in range(self.max_revisions + 1):
            msg_type = "task" if round_index == 0 else "revision_request"

            sent = self._bus_send(
                from_agent="ceo",
                to_agent=task.target_agent,
                message_type=msg_type,
                payload={
                    "task_id": task.task_id,
                    "idea": task.startup_idea,
                    "brief": task.task_brief,
                    "expected": task.expected_output,
                    "constraints": task.constraints,
                    "context": task.context,
                    "revision_instruction": revision_instruction,
                    "round": round_index,
                },
                parent_message_id=parent_id,
            )

            self._log(
                stage=f"{task.target_agent}_dispatch",
                detail=f"CEO → {task.target_agent} [{msg_type}] round {round_index} (Redis)",
                data={"message_id": sent["message_id"], "revision_instruction": revision_instruction},
            )

            reply = self._wait_for_reply(from_agent=task.target_agent, parent_message_id=sent["message_id"])
            result = reply.get("payload") or {}
            parent_id = reply.get("message_id")

            if isinstance(result, dict) and result.get("confirmation_message"):
                self._bus_send(
                    from_agent=task.target_agent,
                    to_agent="ceo",
                    message_type="confirmation",
                    payload={"confirmation_message": result.get("confirmation_message")},
                    parent_message_id=reply.get("message_id"),
                )

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
                detail=f"{task.target_agent} round {round_index}: acceptable={decision.acceptable} score={decision.score}/10",
                data={
                    "score": decision.score,
                    "rationale": decision.rationale,
                    "follow_up_instruction": decision.follow_up_instruction,
                },
            )

            if decision.acceptable:
                self._bus_send(
                    from_agent="ceo",
                    to_agent=task.target_agent,
                    message_type="confirmation",
                    payload={"status": "accepted", "score": decision.score, "rationale": decision.rationale},
                    parent_message_id=reply.get("message_id"),
                )
                return result

            revision_instruction = decision.follow_up_instruction or "Improve completeness, specificity, and execution detail."
            self._log(
                stage="revision_requested",
                detail=f"CEO requesting revision #{round_index + 1} from {task.target_agent} (Redis)",
                data={"instruction": revision_instruction},
            )

        return result

    def _run_with_review(self, task: TaskMessage) -> Dict[str, Any]:
        if self._redis_enabled():
            return self._run_with_review_via_redis(task)

        agent_map = {"product": self.product_agent, "engineer": self.engineer_agent, "marketing": self.marketing_agent}
        agent = agent_map[task.target_agent]
        revision_instruction = ""
        result: Dict[str, Any] = {}
        parent_id: Optional[str] = None

        for round_index in range(self.max_revisions + 1):
            msg_type = "task" if round_index == 0 else "revision_request"

            sent = self.bus.send(
                from_agent="ceo",
                to_agent=task.target_agent,
                message_type=msg_type,
                payload={
                    "task_id": task.task_id,
                    "idea": task.startup_idea,
                    "brief": task.task_brief,
                    "expected": task.expected_output,
                    "constraints": task.constraints,
                    "context": task.context,
                    "revision_instruction": revision_instruction,
                    "round": round_index,
                },
                parent_message_id=parent_id,
            )
            self._log(
                stage=f"{task.target_agent}_dispatch",
                detail=f"CEO → {task.target_agent} [{msg_type}] round {round_index}",
                data={"message_id": sent["message_id"], "revision_instruction": revision_instruction},
            )

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
                    detail=f"{task.target_agent} failed all retries — continuing with empty fallback output.",
                    data=failure.to_dict(),
                )
                result = {"degraded": True, "error": str(failure)}
            else:
                result = agent_result_obj.output  # type: ignore[union-attr]

            reply = self.bus.send(
                from_agent=task.target_agent,
                to_agent="ceo",
                message_type="result",
                payload=result,
                parent_message_id=sent["message_id"],
            )
            parent_id = reply["message_id"]

            if isinstance(result, dict) and result.get("confirmation_message"):
                self.bus.send(
                    from_agent=task.target_agent,
                    to_agent="ceo",
                    message_type="confirmation",
                    payload={"confirmation_message": result.get("confirmation_message")},
                    parent_message_id=reply["message_id"],
                )

            if failure:
                break

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
                detail=f"{task.target_agent} round {round_index}: acceptable={decision.acceptable} score={decision.score}/10",
                data={
                    "score": decision.score,
                    "rationale": decision.rationale,
                    "follow_up_instruction": decision.follow_up_instruction,
                },
            )

            if decision.acceptable:
                self.bus.send(
                    from_agent="ceo",
                    to_agent=task.target_agent,
                    message_type="confirmation",
                    payload={"status": "accepted", "score": decision.score, "rationale": decision.rationale},
                    parent_message_id=reply["message_id"],
                )
                return result

            revision_instruction = decision.follow_up_instruction or "Improve completeness, specificity, and execution detail."
            self._log(
                stage="revision_requested",
                detail=f"CEO requesting revision #{round_index + 1} from {task.target_agent}",
                data={"instruction": revision_instruction},
            )

        return result

    def _save_decision_log(self) -> Path:
        logs_dir = self.output_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = logs_dir / f"ceo_decisions_{ts}.json"
        payload = [{"stage": e.stage, "detail": e.detail, "data": e.data} for e in self.logs]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        return path

    def run(self, startup_idea: str, dry_run: bool = False) -> Dict[str, Any]:
        decomposition, failure = safe_call(
            self.groq.decompose_startup_idea,
            startup_idea,
            agent_name="ceo",
            operation="decompose_startup_idea",
            retries=3,
            fallback={},
        )
        if failure or not decomposition:
            if failure:
                self._log_failure(failure)
            decomposition = self.groq._complete_json(
                role_prompt="",
                user_prompt="",
                mock_default={
                    "product_task": {
                        "task_brief": "Define InvoiceHound personas, value proposition, ranked core features, and user stories.",
                        "expected_output": ["Value proposition", "Three personas", "Five ranked features", "Three user stories"],
                        "constraints": ["Must cover Day 1/7/14 escalation.", "Must cover hour-based split."],
                    },
                    "engineer_task": {
                        "task_brief": "Build index.html landing page and execute GitHub workflow: issue, branch, commit, PR.",
                        "expected_output": ["HTML", "Issue URL", "PR URL"],
                        "constraints": ["Branch: agent-landing-page."],
                    },
                    "marketing_task": {
                        "task_brief": "Generate launch copy, send email via SendGrid, post Slack Block Kit to #launches.",
                        "expected_output": ["Tagline", "Description", "Cold email", "Social posts"],
                        "constraints": ["Tone for freelancers frustrated by chasing clients."],
                    },
                },
            )

        self._log("decompose", "CEO (Groq) decomposed startup idea.", decomposition)

        product_task = self._build_task(startup_idea, "product", decomposition.get("product_task", {}))
        engineer_task = self._build_task(startup_idea, "engineer", decomposition.get("engineer_task", {}))
        marketing_task = self._build_task(startup_idea, "marketing", decomposition.get("marketing_task", {}))

        product_output = self._run_with_review(product_task)
        engineer_task.context["product_spec"] = product_output
        engineer_output = self._run_with_review(engineer_task)
        marketing_task.context["product_spec"] = product_output
        marketing_task.context["pr_url"] = engineer_output.get("pr_url", "")
        marketing_output = self._run_with_review(marketing_task)

        outputs = {"product": product_output, "engineer": engineer_output, "marketing": marketing_output}

        if self._redis_enabled():
            qa_sent = self._bus_send(from_agent="ceo", to_agent="qa", message_type="task", payload={"outputs": outputs})
            qa_reply = self._wait_for_reply(from_agent="qa", parent_message_id=qa_sent["message_id"])
            qa_payload = qa_reply.get("payload") or {}
            qa_passed = bool(qa_payload.get("passed", False))
            qa_notes = str(qa_payload.get("notes", ""))
            qa_issues = list(qa_payload.get("issues", []))
            qa_report = qa_payload.get("report", {})
        else:
            qa_sent = self.bus.send(
                from_agent="ceo",
                to_agent="qa",
                message_type="task",
                payload={
                    "html": str(engineer_output.get("html", "")),
                    "copy": marketing_output,
                    "product_spec": product_output,
                },
            )
            qa_passed, qa_notes, qa_issues, qa_report = self.qa_agent.run(outputs)
            self.bus.send(
                from_agent="qa",
                to_agent="ceo",
                message_type="result",
                payload={"passed": qa_passed, "notes": qa_notes, "issues": qa_issues, "report": qa_report},
                parent_message_id=qa_sent["message_id"],
            )

        self._log("qa", qa_notes, {"issues": qa_issues, "report": qa_report})

        final_summary = self.groq.summarize_for_slack(
            startup_idea=startup_idea,
            agent_outputs=outputs,
            qa_notes=f"{qa_notes} Issues: {qa_issues}",
        )

        self.bus.send(
            from_agent="ceo",
            to_agent="slack_channel",
            message_type="final_summary",
            payload={"summary": final_summary},
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
                    from_agent="ceo",
                    to_agent="slack_channel",
                    message_type="confirmation",
                    payload={"summary": final_summary, "slack_response": slack_response},
                )
        else:
            self._log("slack", "Skipped Slack post (dry_run or missing SLACK_CHANNEL_ID).")

        self._stop_redis_workers()

        log_path = self._save_decision_log()
        msg_path = self.bus.save(self.output_dir / "logs")

        task_messages = {
            "product": product_task.__dict__,
            "engineer": engineer_task.__dict__,
            "marketing": marketing_task.__dict__,
        }

        return {
            "task_messages": task_messages,
            "agent_outputs": outputs,
            "qa": {"passed": qa_passed, "notes": qa_notes, "issues": qa_issues},
            "qa_report": qa_report,
            "slack_response": slack_response,
            "decision_log_path": str(log_path),
            "message_log_path": str(msg_path),
            "final_summary_text": final_summary,
            "all_messages": self.bus.all_messages(),
            "ceo_messages": self.bus.ceo_messages(),
            "failures": self.failures,
        }


__all__ = ["CEOAgent"]

