import uuid
from datetime import datetime

from sqlalchemy import (
    Column, String, Float, Integer, DateTime, ForeignKey, JSON, Text, Enum
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship
import enum

from app.core.db import Base


def gen_uuid():
    return str(uuid.uuid4())


class AuditStatus(str, enum.Enum):
    pending = "pending"
    profiling = "profiling"
    running_metrics = "running_metrics"
    mitigating = "mitigating"
    generating_report = "generating_report"
    completed = "completed"
    failed = "failed"


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    name = Column(String, nullable=False)
    source_path = Column(String, nullable=False)  # path/URI to raw file
    protected_attributes = Column(JSON, nullable=True)  # e.g. ["race", "sex"]
    detected_proxies = Column(JSON, nullable=True)  # e.g. {"zip_code": 0.71}
    target_column = Column(String, nullable=True)
    row_count = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    models = relationship("MLModel", back_populates="dataset")


class MLModel(Base):
    __tablename__ = "models"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    dataset_id = Column(UUID(as_uuid=False), ForeignKey("datasets.id"), nullable=False)
    name = Column(String, nullable=False)  # e.g. "logistic_regression_v1"
    algorithm = Column(String, nullable=False)  # "logistic_regression" | "gradient_boosting" | ...
    artifact_path = Column(String, nullable=True)  # where the pickled/joblib model lives
    hyperparameters = Column(JSON, nullable=True)
    baseline_accuracy = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    dataset = relationship("Dataset", back_populates="models")
    audit_runs = relationship("AuditRun", back_populates="model")


class AuditRun(Base):
    __tablename__ = "audit_runs"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    model_id = Column(UUID(as_uuid=False), ForeignKey("models.id"), nullable=False)
    status = Column(Enum(AuditStatus), default=AuditStatus.pending, nullable=False)
    agent_trace = Column(JSON, nullable=True)  # log of profiler/critic reasoning steps
    report_text = Column(Text, nullable=True)  # final plain-language report
    created_at = Column(DateTime, default=datetime.utcnow)
    completed_at = Column(DateTime, nullable=True)

    model = relationship("MLModel", back_populates="audit_runs")
    metric_results = relationship("MetricResult", back_populates="audit_run")
    mitigation_results = relationship("MitigationResult", back_populates="audit_run")


class MetricResult(Base):
    __tablename__ = "metric_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    audit_run_id = Column(UUID(as_uuid=False), ForeignKey("audit_runs.id"), nullable=False)
    stage = Column(String, nullable=False)  # "baseline" | "post_mitigation"
    metric_name = Column(String, nullable=False)  # "demographic_parity_diff", etc.
    protected_attribute = Column(String, nullable=False)
    value = Column(Float, nullable=False)
    ci_lower = Column(Float, nullable=True)
    ci_upper = Column(Float, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    audit_run = relationship("AuditRun", back_populates="metric_results")


class MitigationResult(Base):
    __tablename__ = "mitigation_results"

    id = Column(UUID(as_uuid=False), primary_key=True, default=gen_uuid)
    audit_run_id = Column(UUID(as_uuid=False), ForeignKey("audit_runs.id"), nullable=False)
    method = Column(String, nullable=False)  # "reweighing" | "exponentiated_gradient" | "threshold_optimizer"
    stage_type = Column(String, nullable=False)  # "pre" | "in" | "post"
    accuracy = Column(Float, nullable=True)
    fairness_metrics = Column(JSON, nullable=True)  # snapshot of metric suite after mitigation
    created_at = Column(DateTime, default=datetime.utcnow)

    audit_run = relationship("AuditRun", back_populates="mitigation_results")
