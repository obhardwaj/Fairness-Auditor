"""
Week 4 deliverable (skeleton for now): LangGraph state graph for the audit pipeline.

State flows: profiler -> metric_runner -> critic -> (mitigation loop) -> report

Each node should call into app.ml.* for actual computation — the LLM's job is
reasoning about *which* metrics/mitigations are appropriate and explaining
results, not computing them. Keep that boundary strict: no metric math inside
a prompt.
"""
from __future__ import annotations

from typing import TypedDict, Optional
from langgraph.graph import StateGraph, END


class AuditState(TypedDict, total=False):
    audit_run_id: str
    data_profile: dict
    protected_attrs: list[str]
    model_metadata: dict
    metric_results: dict
    contradictions: list[str]
    mitigation_results: dict
    report_draft: Optional[str]


def profiler_node(state: AuditState) -> AuditState:
    """
    TODO(week 4): Given state['data_profile'], reason about which fairness
    definitions are appropriate (e.g. demographic parity may be inappropriate
    if base rates genuinely differ for legitimate reasons). Call the
    Groq API via langchain_groq.ChatGroq here.
    """
    return state


def metric_runner_node(state: AuditState) -> AuditState:
    """
    TODO(week 4): Call app.ml.metrics functions as tools (no LLM computation),
    populate state['metric_results'] with values + confidence intervals.
    """
    return state


def critic_node(state: AuditState) -> AuditState:
    """
    TODO(week 4): Given state['metric_results'], flag contradictions or
    impossibility-theorem violations (e.g. equalized odds satisfied but
    calibration violated). Populate state['contradictions'].
    """
    return state


def mitigation_node(state: AuditState) -> AuditState:
    """
    TODO(week 4): Based on critic's flags, choose and run a mitigation method
    from app.ml.mitigation (to be added in Week 3), re-run metrics, populate
    state['mitigation_results'].
    """
    return state


def report_node(state: AuditState) -> AuditState:
    """
    TODO(week 4): Synthesize state into a plain-language report for
    state['report_draft'], to be persisted on the AuditRun row.
    """
    return state


def needs_mitigation(state: AuditState) -> str:
    """
    Conditional edge: route to mitigation if any metric violates a threshold
    (e.g. disparate impact ratio < 0.8), otherwise go straight to report.
    """
    # TODO(week 4): inspect state['metric_results'] for real thresholds
    if state.get("contradictions"):
        return "mitigation"
    return "report"


def build_audit_graph():
    graph = StateGraph(AuditState)

    graph.add_node("profiler", profiler_node)
    graph.add_node("metric_runner", metric_runner_node)
    graph.add_node("critic", critic_node)
    graph.add_node("mitigation", mitigation_node)
    graph.add_node("report", report_node)

    graph.set_entry_point("profiler")
    graph.add_edge("profiler", "metric_runner")
    graph.add_edge("metric_runner", "critic")
    graph.add_conditional_edges(
        "critic", needs_mitigation, {"mitigation": "mitigation", "report": "report"}
    )
    graph.add_edge("mitigation", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_audit_graph(audit_run_id: str) -> dict:
    """Entry point called from app.core.tasks.run_audit_task once Week 4 lands."""
    compiled_graph = build_audit_graph()
    initial_state: AuditState = {"audit_run_id": audit_run_id}
    return compiled_graph.invoke(initial_state)
