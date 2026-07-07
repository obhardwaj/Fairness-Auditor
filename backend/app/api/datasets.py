from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
import pandas as pd

from app.core.db import get_db
from app.models.models import Dataset
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
