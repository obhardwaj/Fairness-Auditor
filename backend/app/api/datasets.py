from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd
from app.core.db import get_db
from app.models.models import Dataset, MLModel
from app.ml.compas_pipeline import run_compas_baseline_pipeline
from app.schemas.schemas import DatasetOut
from app.ml.profiling import profile_dataset

router = APIRouter(prefix="/datasets", tags=["datasets"])


@router.post("/ingest", response_model=DatasetOut)
def ingest_dataset(name: str, source_path: str, db: Session = Depends(get_db)):
    """
    Week 1 stub: given a path to a CSV already on disk (e.g. COMPAS or Adult
    dataset placed in a data/ volume), profile it and store metadata.
    Replace with a proper multipart file upload endpoint once the frontend
    upload flow is built.
    """
    try:
        df = pd.read_csv(source_path)
    except FileNotFoundError:
        raise HTTPException(status_code=400, detail=f"File not found: {source_path}")

    profile = profile_dataset(df)

    dataset = Dataset(
        name=name,
        source_path=source_path,
        protected_attributes=profile["protected_attributes"],
        detected_proxies=profile["detected_proxies"],
        row_count=profile["row_count"],
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)
    return dataset


@router.get("", response_model=list[DatasetOut])
def list_datasets(db: Session = Depends(get_db)):
    return db.query(Dataset).all()

@router.post("/train-compas-baseline", response_model=DatasetOut)
def train_compas_baseline(
    name: str = "compas",
    csv_path: str = "data/compas-scores-two-years.csv",
    artifact_dir: str = "data/artifacts",
    db: Session = Depends(get_db),
):
    """
    Runs the full COMPAS loading/filtering/training pipeline and persists
    a Dataset row plus one MLModel row per trained baseline classifier.

    This is the bridge between app/ml/compas_pipeline.py (which only prints
    to console) and Postgres — after this runs, Week 2's metric suite has
    real stored model rows to audit instead of one-off script output.
    """
    result = run_compas_baseline_pipeline(csv_path=csv_path, artifact_dir=artifact_dir)

    dataset = Dataset(
        name=name,
        source_path=csv_path,
        protected_attributes=result["protected_attributes"],
        row_count=result["row_count_after_filtering"],
    )
    db.add(dataset)
    db.commit()
    db.refresh(dataset)

    for trained in result["trained_models"]:
        model_row = MLModel(
            dataset_id=dataset.id,
            name=trained.name,
            algorithm=trained.algorithm,
            artifact_path=trained.artifact_path,
            baseline_accuracy=trained.accuracy,
        )
        db.add(model_row)

    db.commit()
    db.refresh(dataset)
    return dataset

@router.get("/models")
def list_models(db: Session = Depends(get_db)):
    """Lists all trained models, for populating the 'Run New Audit' picker."""
    from app.models.models import MLModel
    models = db.query(MLModel).order_by(MLModel.created_at.desc()).all()
    return [
        {"id": m.id, "name": m.name, "algorithm": m.algorithm, "baseline_accuracy": m.baseline_accuracy}
        for m in models
    ]