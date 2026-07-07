from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.core.db import get_db
from app.models.models import AuditRun, AuditStatus
from app.schemas.schemas import AuditRunCreate, AuditRunOut
from app.core.tasks import run_audit_task

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
        return {"status": audit_run.status, "report": None}
    return {"status": audit_run.status, "report": audit_run.report_text}
