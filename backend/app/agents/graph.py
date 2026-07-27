"""
LangGraph state graph for the audit pipeline: profiler -> metric_runner ->
critic -> (conditional mitigation) -> report.

Design boundary (kept strict throughout): LLM nodes reason about *which*
metrics/mitigations are appropriate and *why* — they never compute fairness
math themselves. All numbers come from app.ml.metrics / app.ml.mitigation.
"""
from __future__ import annotations

from typing import TypedDict, Optional, Any
import joblib
import numpy as np
from langgraph.graph import StateGraph, END

from app.agents.llm import get_llm
from app.ml.compas_pipeline import load_and_filter_compas, filter_to_binary_race, build_feature_matrix
from app.ml.metrics import run_full_metric_suite
from app.ml.mitigation import apply_reweighing, apply_exponentiated_gradient, apply_threshold_optimizer

DISPARATE_IMPACT_THRESHOLD = 0.8

MITIGATION_FNS = {
    "reweighing": apply_reweighing,
    "exponentiated_gradient": apply_exponentiated_gradient,
    "threshold_optimizer": apply_threshold_optimizer,
}


class AuditState(TypedDict, total=False):
    audit_run_id: str
    dataset_source_path: str
    model_artifact_path: str
    algorithm: str

    # populated by profiler_node
    X: Any
    y: Any
    y_pred: Any
    y_prob: Any
    protected_attr_values: Any
    base_rate_gap: float
    profiler_notes: str

    # populated by metric_runner_node
    metric_results: dict

    # populated by critic_node
    contradictions: list[str]
    needs_mitigation_flag: bool

    # populated by mitigation_node
    chosen_mitigation: str
    mitigation_reasoning: str
    mitigation_metric_results: dict
    mitigation_accuracy: float

    # populated by report_node
    report_draft: str


def profiler_node(state: AuditState) -> AuditState:
    df = load_and_filter_compas(state["dataset_source_path"])
    df = filter_to_binary_race(df)  # matches the race-binary framing used throughout Weeks 2-3
    X, y, protected_attr_cols, protected_attrs_raw = build_feature_matrix(df)

    clf = joblib.load(state["model_artifact_path"])
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]
    race_values = protected_attrs_raw["race"].to_numpy()

    # Base rate context: if the two groups genuinely reoffend at different
    # rates, demographic parity may be an inappropriate target (a model
    # perfectly calibrated to reality would *not* have equal selection rates)
    # -- this is exactly the kind of judgment call the profiler should make
    # explicit rather than silently assuming one "correct" fairness notion.
    groups = np.unique(race_values)
    base_rates = {g: float(y.to_numpy()[race_values == g].mean()) for g in groups}
    base_rate_gap = max(base_rates.values()) - min(base_rates.values())

    llm = get_llm(temperature=0.1)
    prompt = f"""You are auditing a recidivism-prediction classifier for fairness.

Ground truth base rates of the outcome (two_year_recid) by racial group:
{base_rates}
Gap between groups: {base_rate_gap:.4f}

In 3-4 sentences: explain whether demographic parity (equal selection rates
across groups) is an appropriate fairness target here given this base rate
gap, or whether equalized odds / calibration would be more appropriate given
the actual difference in underlying reoffense rates. Be concrete and note
the tradeoff, not just "it depends"."""

    response = llm.invoke(prompt)

    state["X"] = X
    state["y"] = y
    state["y_pred"] = y_pred
    state["y_prob"] = y_prob
    state["protected_attr_values"] = race_values
    state["base_rate_gap"] = base_rate_gap
    state["profiler_notes"] = response.content
    return state


def metric_runner_node(state: AuditState) -> AuditState:
    suite = run_full_metric_suite(
        y_true=state["y"].to_numpy(),
        y_pred=state["y_pred"],
        y_prob=state["y_prob"],
        protected_attr=state["protected_attr_values"],
        n_bootstrap=500,
    )
    state["metric_results"] = suite
    return state


def critic_node(state: AuditState) -> AuditState:
    suite = state["metric_results"]
    di_point, di_lo, di_hi = suite["disparate_impact_ratio"]
    dp_point, _, _ = suite["demographic_parity_difference"]
    eo_point, _, _ = suite["equalized_odds_difference"]

    metrics_summary = "\n".join(
        f"- {name}: {vals[0]:.4f}  (95% CI [{vals[1]:.4f}, {vals[2]:.4f}])"
        for name, vals in suite.items() if name != "calibration_within_groups"
    )

    llm = get_llm(temperature=0.1)
    prompt = f"""You are the critic agent in a fairness audit pipeline. Given these
computed fairness metrics (with bootstrap 95% confidence intervals):

{metrics_summary}

Profiler's earlier note on which fairness notion is appropriate here:
"{state['profiler_notes']}"

List up to 3 concrete contradictions or notable tensions between these
metrics (e.g. one metric passing while a related one fails, or a result
inconsistent with the profiler's framing). Reference the impossibility
theorem (you cannot generally satisfy demographic parity, equalized odds,
and calibration simultaneously unless base rates are equal) where relevant.
Output as a numbered list, no preamble."""

    response = llm.invoke(prompt)
    contradictions = [line.strip() for line in response.content.split("\n") if line.strip()]

    state["contradictions"] = contradictions
    state["needs_mitigation_flag"] = di_point < DISPARATE_IMPACT_THRESHOLD
    return state


def needs_mitigation(state: AuditState) -> str:
    return "mitigation" if state.get("needs_mitigation_flag") else "report"


def mitigation_node(state: AuditState) -> AuditState:
    llm = get_llm(temperature=0.1)
    prompt = f"""A classifier failed the disparate impact threshold (ratio must be >= 0.8).

Current fairness metrics:
{state['metric_results']['disparate_impact_ratio']}

Three mitigation methods are available:
- reweighing (pre-processing): adjusts training sample weights, cheapest but
  weaker on more flexible model classes
- exponentiated_gradient (in-processing): constrains the training objective
  directly, but has shown inconsistent results depending on base model
- threshold_optimizer (post-processing): adjusts decision thresholds per
  group after training, strongest and most consistent in prior runs on this
  dataset

Pick exactly ONE method by name (respond with only the method's identifier:
reweighing, exponentiated_gradient, or threshold_optimizer) and give a
one-sentence justification on the next line."""

    response = llm.invoke(prompt)
    lines = [l.strip() for l in response.content.strip().split("\n") if l.strip()]
    chosen = lines[0].lower().strip(".:")
    reasoning = lines[1] if len(lines) > 1 else ""

    if chosen not in MITIGATION_FNS:
        chosen = "threshold_optimizer"  # safe fallback given its track record above

    mitigation_fn = MITIGATION_FNS[chosen]

    if chosen == "reweighing":
        results = mitigation_fn(
            X=state["X"], y=state["y"], protected_attr_col_name="race",
            protected_attr_values=state["protected_attr_values"],
            artifact_dir="data/artifacts",
        )
    else:
        results = mitigation_fn(
            state["X"], state["y"], state["protected_attr_values"], "data/artifacts",
        )

    result = results[state["algorithm"]]
    y_test = result["y_test"].to_numpy()

    if chosen == "exponentiated_gradient":
        pmf = result["model"]._pmf_predict(result["X_test"])
        y_prob = pmf[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
    elif chosen == "threshold_optimizer":
        np.random.seed(42)
        y_pred = result["model"].predict(result["X_test"], sensitive_features=result["test_protected_attr"])
        y_prob = y_pred.astype(float)
    else:  # reweighing
        y_pred = result["model"].predict(result["X_test"])
        y_prob = result["model"].predict_proba(result["X_test"])[:, 1]

    mitigation_suite = run_full_metric_suite(
        y_true=y_test, y_pred=y_pred, y_prob=y_prob,
        protected_attr=result["test_protected_attr"], n_bootstrap=500,
    )

    state["chosen_mitigation"] = chosen
    state["mitigation_reasoning"] = reasoning
    state["mitigation_metric_results"] = mitigation_suite
    state["mitigation_accuracy"] = result["accuracy"]
    return state


def report_node(state: AuditState) -> AuditState:
    di_baseline = state["metric_results"]["disparate_impact_ratio"][0]

    mitigation_section = ""
    if state.get("chosen_mitigation"):
        di_mitigated = state["mitigation_metric_results"]["disparate_impact_ratio"][0]
        mitigation_section = f"""
Mitigation applied: {state['chosen_mitigation']} ({state['mitigation_reasoning']})
Disparate impact ratio after mitigation: {di_mitigated:.4f} (was {di_baseline:.4f})
Accuracy after mitigation: {state['mitigation_accuracy']:.4f}
"""

    llm = get_llm(temperature=0.4)
    prompt = f"""Write a plain-language fairness audit report (4-6 short paragraphs,
no headers needed) for a non-technical stakeholder, based on:

Profiler's assessment: {state['profiler_notes']}

Baseline disparate impact ratio: {di_baseline:.4f} (0.8 is the legal 80% rule threshold)
Critic's flagged contradictions:
{chr(10).join(state.get('contradictions', ['None found.']))}
{mitigation_section}

Explain what was found, why it matters, and what was (or wasn't) done about
it, in accessible language. Don't invent numbers not given above."""

    response = llm.invoke(prompt)
    state["report_draft"] = response.content
    return state


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
    graph.add_conditional_edges("critic", needs_mitigation, {"mitigation": "mitigation", "report": "report"})
    graph.add_edge("mitigation", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_audit_graph(audit_run_id: str) -> dict:
    """Entry point: loads model/dataset context from DB, runs the graph, persists results."""
    from app.core.db import SessionLocal
    from app.models.models import AuditRun, MLModel, Dataset, AuditStatus

    db = SessionLocal()
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    model_row = db.query(MLModel).filter(MLModel.id == audit_run.model_id).first()
    dataset_row = db.query(Dataset).filter(Dataset.id == model_row.dataset_id).first()

    initial_state: AuditState = {
        "audit_run_id": audit_run_id,
        "dataset_source_path": dataset_row.source_path,
        "model_artifact_path": model_row.artifact_path,
        "algorithm": model_row.algorithm,
    }

    compiled_graph = build_audit_graph()
    final_state = compiled_graph.invoke(initial_state)

    audit_run.report_text = final_state["report_draft"]
    audit_run.agent_trace = {
        "profiler_notes": final_state.get("profiler_notes"),
        "contradictions": final_state.get("contradictions"),
        "chosen_mitigation": final_state.get("chosen_mitigation"),
        "mitigation_reasoning": final_state.get("mitigation_reasoning"),
    }
    audit_run.status = AuditStatus.completed
    db.commit()
    db.close()

    return {"status": "completed", "audit_run_id": audit_run_id}