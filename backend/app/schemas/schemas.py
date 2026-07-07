from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class DatasetOut(BaseModel):
    id: str
    name: str
    protected_attributes: Optional[list[str]] = None
    detected_proxies: Optional[dict] = None
    row_count: Optional[int] = None
    created_at: datetime

    class Config:
        from_attributes = True


class ModelOut(BaseModel):
    id: str
    dataset_id: str
    name: str
    algorithm: str
    baseline_accuracy: Optional[float] = None
    created_at: datetime

    class Config:
        from_attributes = True


class AuditRunCreate(BaseModel):
    model_id: str


class AuditRunOut(BaseModel):
    id: str
    model_id: str
    status: str
    created_at: datetime
    completed_at: Optional[datetime] = None

    class Config:
        from_attributes = True
