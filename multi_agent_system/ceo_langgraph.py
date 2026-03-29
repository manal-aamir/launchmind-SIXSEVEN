"""
CEO pipeline orchestration using LangGraph (assignment-recommended framework).

Defines an explicit state machine: decompose → product → engineer → marketing → QA,
with a conditional edge QA fail → engineer revision → QA again → finalize.

All side effects (message bus, Groq, GitHub, etc.) still go through CEOAgent methods.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, List, Literal, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from multi_agent_system.models import TaskMessage
from multi_agent_system.retry import safe_call


class CEOPipelineState(TypedDict, total=False):
    """Mutable pipeline state passed between LangGraph nodes."""

    startup_idea: str
    dry_run: bool
    decomposition: Dict[str, Any]
    product_task: Dict[str, Any]
    engineer_task: Dict[str, Any]
    marketing_task: Dict[str, Any]
    product_output: Dict[str, Any]
    engineer_output: Dict[str, Any]
    marketing_output: Dict[str, Any]
    outputs: Dict[str, Any]
    qa_passed: bool
    qa_notes: str
    qa_issues: List[Any]
    qa_report: Any
    # Set by finalize node
    final_summary_text: str
    slack_response: Dict[str, Any]
    decision_log_path: str
    message_log_path: str
    task_messages: Dict[str, Any]
    result: Dict[str, Any]


def _ceo(config: RunnableConfig) -> Any:
    ceo = (config.get("configurable") or {}).get("ceo")
    if ceo is None:
        raise RuntimeError("LangGraph CEO pipeline requires config['configurable']['ceo']")
    return ceo


def _task_from_dict(d: Dict[str, Any]) -> TaskMessage:
    return TaskMessage(
        task_id=str(d["task_id"]),
        target_agent=str(d["target_agent"]),
        startup_idea=str(d["startup_idea"]),
        task_brief=str(d["task_brief"]),
        expected_output=list(d.get("expected_output", [])),
        constraints=list(d.get("constraints", [])),
        context=dict(d.get("context", {})),
    )


def node_decompose(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    startup_idea = state["startup_idea"]

    decomposition, failure = safe_call(
        ceo.groq.decompose_startup_idea,
        startup_idea,
        agent_name="ceo",
        operation="decompose_startup_idea",
        retries=3,
        fallback={},
    )
    if failure or not decomposition:
        if failure:
            ceo._log_failure(failure)
        decomposition = ceo.groq._complete_json(
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

    ceo._log("decompose", "CEO (Groq) decomposed startup idea.", decomposition)

    product_task = ceo._build_task(startup_idea, "product", decomposition.get("product_task", {}))
    engineer_task = ceo._build_task(startup_idea, "engineer", decomposition.get("engineer_task", {}))
    marketing_task = ceo._build_task(startup_idea, "marketing", decomposition.get("marketing_task", {}))

    return {
        "decomposition": decomposition,
        "product_task": asdict(product_task),
        "engineer_task": asdict(engineer_task),
        "marketing_task": asdict(marketing_task),
    }


def node_product(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    task = _task_from_dict(state["product_task"])
    product_output = ceo._run_with_review(task)
    return {"product_output": product_output, "product_task": asdict(task)}


def node_engineer(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    task = _task_from_dict(state["engineer_task"])
    task.context["product_spec"] = state["product_output"]
    engineer_output = ceo._run_with_review(task)
    return {"engineer_output": engineer_output, "engineer_task": asdict(task)}


def node_marketing(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    task = _task_from_dict(state["marketing_task"])
    task.context["product_spec"] = state["product_output"]
    task.context["pr_url"] = str(state["engineer_output"].get("pr_url", ""))
    marketing_output = ceo._run_with_review(task)
    return {"marketing_output": marketing_output, "marketing_task": asdict(task)}


def node_qa_first(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    outputs = {
        "product": state["product_output"],
        "engineer": state["engineer_output"],
        "marketing": state["marketing_output"],
    }

    if ceo._redis_enabled():
        qa_sent = ceo._bus_send(
            from_agent="ceo",
            to_agent="qa",
            message_type="task",
            payload={"outputs": outputs},
        )
        qa_reply = ceo._wait_for_reply(from_agent="qa", parent_message_id=qa_sent["message_id"])
        qa_payload = qa_reply.get("payload") or {}
        qa_passed = bool(qa_payload.get("passed", False))
        qa_notes = str(qa_payload.get("notes", ""))
        qa_issues = list(qa_payload.get("issues", []))
        qa_report = qa_payload.get("report", {})
    else:
        qa_sent = ceo.bus.send(
            from_agent="ceo",
            to_agent="qa",
            message_type="task",
            payload={
                "html": str(state["engineer_output"].get("html", "")),
                "copy": state["marketing_output"],
                "product_spec": state["product_output"],
            },
        )
        qa_passed, qa_notes, qa_issues, qa_report = ceo.qa_agent.run(outputs)
        ceo.bus.send(
            from_agent="qa",
            to_agent="ceo",
            message_type="result",
            payload={"passed": qa_passed, "notes": qa_notes, "issues": qa_issues, "report": qa_report},
            parent_message_id=qa_sent["message_id"],
        )

    ceo._log("qa", qa_notes, {"issues": qa_issues, "report": qa_report})
    return {
        "outputs": outputs,
        "qa_passed": qa_passed,
        "qa_notes": qa_notes,
        "qa_issues": qa_issues,
        "qa_report": qa_report,
    }


def route_after_qa(state: CEOPipelineState) -> Literal["engineer_fix", "finalize"]:
    return "finalize" if state.get("qa_passed") else "engineer_fix"


def node_engineer_fix(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    qa_issues = list(state.get("qa_issues", []))
    ceo._log("qa_recovery", "QA failed — CEO requesting engineer revision.", {"issues": qa_issues})

    engineer_task = _task_from_dict(state["engineer_task"])
    engineer_task.context["qa_issues"] = qa_issues
    issues_text = "; ".join(str(i) for i in qa_issues) if qa_issues else (
        "Improve the landing page to match the product spec and InvoiceHound requirements."
    )
    revised_instruction = f"Address QA issues: {issues_text}"
    outputs = dict(state["outputs"])

    if ceo._redis_enabled():
        rev_sent = ceo._bus_send(
            from_agent="ceo",
            to_agent="engineer",
            message_type="revision_request",
            payload={
                "task_id": engineer_task.task_id,
                "idea": engineer_task.startup_idea,
                "brief": engineer_task.task_brief,
                "expected": engineer_task.expected_output,
                "constraints": engineer_task.constraints,
                "context": dict(engineer_task.context),
                "revision_instruction": revised_instruction,
                "round": 1,
            },
        )
        eng_reply = ceo._wait_for_reply(
            from_agent="engineer",
            parent_message_id=rev_sent["message_id"],
        )
        outputs["engineer"] = eng_reply.get("payload") or outputs["engineer"]
    else:
        revised_obj, rev_failure = safe_call(
            ceo.engineer_agent.run,
            engineer_task,
            revision_instruction=revised_instruction,
            agent_name="engineer",
            operation="qa_revision",
            retries=2,
            fallback=None,
        )
        if rev_failure:
            ceo._log_failure(rev_failure)
        else:
            outputs["engineer"] = revised_obj.output  # type: ignore[union-attr]

    marketing_task = _task_from_dict(state["marketing_task"])
    marketing_task.context["pr_url"] = str(outputs["engineer"].get("pr_url", ""))

    return {
        "outputs": outputs,
        "engineer_output": outputs["engineer"],
        "engineer_task": asdict(engineer_task),
        "marketing_task": asdict(marketing_task),
    }


def node_qa_second(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    outputs = state["outputs"]

    if ceo._redis_enabled():
        qa_sent2 = ceo._bus_send(
            from_agent="ceo",
            to_agent="qa",
            message_type="task",
            payload={"outputs": outputs},
        )
        qa_reply2 = ceo._wait_for_reply(
            from_agent="qa",
            parent_message_id=qa_sent2["message_id"],
        )
        qa_payload2 = qa_reply2.get("payload") or {}
        qa_passed = bool(qa_payload2.get("passed", False))
        qa_notes = str(qa_payload2.get("notes", ""))
        qa_issues = list(qa_payload2.get("issues", []))
        qa_report = qa_payload2.get("report", {})
    else:
        qa_passed, qa_notes, qa_issues, qa_report = ceo.qa_agent.run(outputs)
        qa_sent2 = ceo.bus.send(
            from_agent="ceo",
            to_agent="qa",
            message_type="task",
            payload={"outputs": outputs},
        )
        ceo.bus.send(
            from_agent="qa",
            to_agent="ceo",
            message_type="result",
            payload={
                "passed": qa_passed,
                "notes": qa_notes,
                "issues": qa_issues,
                "report": qa_report,
            },
            parent_message_id=qa_sent2["message_id"],
        )

    ceo._log("qa_rerun", qa_notes, {"issues": qa_issues, "report": qa_report})
    return {
        "qa_passed": qa_passed,
        "qa_notes": qa_notes,
        "qa_issues": qa_issues,
        "qa_report": qa_report,
    }


def node_finalize(state: CEOPipelineState, config: RunnableConfig) -> Dict[str, Any]:
    ceo = _ceo(config)
    startup_idea = state["startup_idea"]
    dry_run = bool(state.get("dry_run", False))
    outputs = state["outputs"]
    qa_notes = str(state.get("qa_notes", ""))
    qa_issues = list(state.get("qa_issues", []))

    final_summary = ceo.groq.summarize_for_slack(
        startup_idea=startup_idea,
        agent_outputs=outputs,
        qa_notes=f"{qa_notes} Issues: {qa_issues}",
    )

    ceo.bus.send(
        from_agent="ceo",
        to_agent="slack_channel",
        message_type="final_summary",
        payload={"summary": final_summary},
    )

    slack_response: Dict[str, Any] = {"ok": False, "reason": "dry_run_or_missing_channel"}
    if not dry_run and ceo.slack_channel_id:
        slack_response, post_failure = safe_call(
            ceo.slack_client.post_message,
            ceo.slack_channel_id,
            final_summary,
            agent_name="ceo",
            operation="post_slack_summary",
            retries=2,
            fallback={"ok": False, "reason": "slack_send_failed"},
        )
        if post_failure:
            ceo._log_failure(post_failure)
        else:
            ceo._log("slack", "Posted final summary to Slack.", slack_response)
            ceo.bus.send(
                from_agent="ceo",
                to_agent="slack_channel",
                message_type="confirmation",
                payload={"summary": final_summary, "slack_response": slack_response},
            )
    else:
        ceo._log("slack", "Skipped Slack post (dry_run or missing SLACK_CHANNEL_ID).")

    ceo._stop_redis_workers()

    log_path = ceo._save_decision_log()
    msg_path = ceo.bus.save(ceo.output_dir / "logs")

    task_messages = {
        "product": state["product_task"],
        "engineer": state["engineer_task"],
        "marketing": state["marketing_task"],
    }

    result = {
        "task_messages": task_messages,
        "agent_outputs": outputs,
        "qa": {"passed": state.get("qa_passed", False), "notes": qa_notes, "issues": qa_issues},
        "qa_report": state.get("qa_report"),
        "slack_response": slack_response,
        "decision_log_path": str(log_path),
        "message_log_path": str(msg_path),
        "final_summary_text": final_summary,
        "all_messages": ceo.bus.all_messages(),
        "ceo_messages": ceo.bus.ceo_messages(),
        "failures": ceo.failures,
    }

    return {
        "final_summary_text": final_summary,
        "slack_response": slack_response,
        "decision_log_path": str(log_path),
        "message_log_path": str(msg_path),
        "task_messages": task_messages,
        "result": result,
    }


def build_ceo_pipeline_graph() -> Any:
    graph = StateGraph(CEOPipelineState)
    graph.add_node("decompose", node_decompose)
    graph.add_node("product", node_product)
    graph.add_node("engineer", node_engineer)
    graph.add_node("marketing", node_marketing)
    graph.add_node("qa_first", node_qa_first)
    graph.add_node("engineer_fix", node_engineer_fix)
    graph.add_node("qa_second", node_qa_second)
    graph.add_node("finalize", node_finalize)

    graph.set_entry_point("decompose")
    graph.add_edge("decompose", "product")
    graph.add_edge("product", "engineer")
    graph.add_edge("engineer", "marketing")
    graph.add_edge("marketing", "qa_first")
    graph.add_conditional_edges(
        "qa_first",
        route_after_qa,
        {"engineer_fix": "engineer_fix", "finalize": "finalize"},
    )
    graph.add_edge("engineer_fix", "qa_second")
    graph.add_edge("qa_second", "finalize")
    graph.add_edge("finalize", END)

    return graph.compile()


def invoke_ceo_langgraph_pipeline(ceo: Any, startup_idea: str, dry_run: bool = False) -> Dict[str, Any]:
    """Run the CEO pipeline through LangGraph; returns the same dict as legacy CEOAgent.run."""
    graph = build_ceo_pipeline_graph()
    final = graph.invoke(
        {"startup_idea": startup_idea, "dry_run": dry_run},
        config={"configurable": {"ceo": ceo}},
    )
    return final["result"]


__all__ = ["build_ceo_pipeline_graph", "invoke_ceo_langgraph_pipeline", "CEOPipelineState"]
