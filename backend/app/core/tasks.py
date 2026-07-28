from app.core.celery_app import celery_app


@celery_app.task(name="run_audit")
def run_audit_task(audit_run_id: str) -> dict:
    from app.agents.graph import run_audit_graph
    return run_audit_graph(audit_run_id)