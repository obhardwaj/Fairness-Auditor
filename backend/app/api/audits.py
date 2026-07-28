from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.core.db import get_db
from app.models.models import AuditRun, AuditStatus
from app.schemas.schemas import AuditRunCreate, AuditRunOut
from app.core.tasks import run_audit_task
import joblib
import pandas as pd
from app.models.models import MLModel, Dataset, MetricResult, AuditStatus
from app.ml.compas_pipeline import load_and_filter_compas, build_feature_matrix
from app.ml.metrics import run_full_metric_suite

router = APIRouter(prefix="/audit", tags=["audits"])


@router.post("", response_model=AuditRunOut)
def create_audit_run(payload: AuditRunCreate, db: Session = Depends(get_db)):
    audit_run = AuditRun(model_id=payload.model_id, status=AuditStatus.pending)
    db.add(audit_run)
    db.commit()
    db.refresh(audit_run)

    # Kick off async pipeline (Week 1: stub task; Week 4: full LangGraph run)
    run_audit_task.delay(audit_run.id)

    return audit_run


@router.get("/mitigation-comparison")
def get_mitigation_comparison(db: Session = Depends(get_db)):
    """
    Returns all persisted MitigationResult rows joined with algorithm name,
    shaped for the frontend's Pareto frontier chart: one point per
    (algorithm, method) with accuracy (x) and disparate impact ratio (y).
    """
    from app.models.models import MitigationResult

    rows = (
        db.query(MitigationResult, MLModel.algorithm)
        .join(AuditRun, MitigationResult.audit_run_id == AuditRun.id)
        .join(MLModel, AuditRun.model_id == MLModel.id)
        .all()
    )

    results = []
    for mitigation_result, algorithm in rows:
        di = mitigation_result.fairness_metrics.get("disparate_impact_ratio", {})
        dp = mitigation_result.fairness_metrics.get("demographic_parity_difference", {})
        eo = mitigation_result.fairness_metrics.get("equalized_odds_difference", {})

        results.append({
            "algorithm": algorithm,
            "method": mitigation_result.method,
            "stage_type": mitigation_result.stage_type,
            "accuracy": mitigation_result.accuracy,
            "disparate_impact_ratio": di.get("value"),
            "demographic_parity_difference": dp.get("value"),
            "equalized_odds_difference": eo.get("value"),
        })

    return results

@router.get("/{audit_run_id}", response_model=AuditRunOut)
def get_audit_run(audit_run_id: str, db: Session = Depends(get_db)):
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")
    return audit_run


@router.get("/{audit_run_id}/report")
def get_audit_report(audit_run_id: str, db: Session = Depends(get_db)):
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")
    if audit_run.status != AuditStatus.completed:
        return {"status": audit_run.status, "report": None, "agent_trace": None}
    return {
        "status": audit_run.status,
        "report": audit_run.report_text,
        "agent_trace": audit_run.agent_trace,
    }


@router.post("/{audit_run_id}/run-metrics")
def run_metrics_for_audit(audit_run_id: str, db: Session = Depends(get_db)):
    """
    Loads the model + dataset tied to this audit run, recomputes predictions,
    runs the full bootstrap fairness metric suite per protected attribute,
    and persists one MetricResult row per (metric, protected_attribute) pair.

    This is the bridge between Week 2's metrics.py and the AuditRun/MetricResult
    tables — after this, results are queryable instead of console-only.
    """
    audit_run = db.query(AuditRun).filter(AuditRun.id == audit_run_id).first()
    if not audit_run:
        raise HTTPException(status_code=404, detail="Audit run not found")

    model_row = db.query(MLModel).filter(MLModel.id == audit_run.model_id).first()
    dataset_row = db.query(Dataset).filter(Dataset.id == model_row.dataset_id).first()

    audit_run.status = AuditStatus.running_metrics
    db.commit()

    # Rebuild the exact same feature matrix used at training time
    df = load_and_filter_compas(dataset_row.source_path)
    X, y_true, protected_attr_cols, protected_attrs_raw = build_feature_matrix(df)

    clf = joblib.load(model_row.artifact_path)
    y_pred = clf.predict(X)
    y_prob = clf.predict_proba(X)[:, 1]

    for protected_col in protected_attr_cols:
        protected_values = protected_attrs_raw[protected_col].to_numpy()

        suite = run_full_metric_suite(
            y_true=y_true.to_numpy(),
            y_pred=y_pred,
            y_prob=y_prob,
            protected_attr=protected_values,
            n_bootstrap=500,  # lower than 1000 for faster iteration; raise for final results
        )

        for metric_name, result in suite.items():
            if metric_name == "calibration_within_groups":
                continue  # curve data, not a scalar CI — store separately if/when needed
            point, lower, upper = result
            db.add(MetricResult(
                audit_run_id=audit_run.id,
                stage="baseline",
                metric_name=metric_name,
                protected_attribute=protected_col,
                value=point,
                ci_lower=lower,
                ci_upper=upper,
            ))

    audit_run.status = AuditStatus.completed
    db.commit()

    return {"status": "completed", "audit_run_id": audit_run_id}
