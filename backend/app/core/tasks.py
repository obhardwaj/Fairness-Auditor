from app.core.celery_app import celery_app


@celery_app.task(name="run_audit")
def run_audit_task(audit_run_id: str) -> dict:
    """
    Entry point for an async audit job.

    Week 1: stub — just marks the run as received.
    Week 4: replace body with a call into app.agents.graph.run_audit_graph(audit_run_id)
    which executes the full LangGraph pipeline (profiler -> metric-runner -> critic ->
    mitigation -> report) and persists results via app.models.
    """
    # TODO(week 4): from app.agents.graph import run_audit_graph
    # return run_audit_graph(audit_run_id)
    return {"audit_run_id": audit_run_id, "status": "received"}
