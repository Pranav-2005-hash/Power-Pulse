import os
import shutil
from typing import Optional
from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from pydantic import BaseModel

from app.core.config import DATA_DIR, SAMPLE_DATA_PATH
from app.storage.state_manager import state_manager

router = APIRouter(tags=["Dataset"])


class PreprocessRequest(BaseModel):
    target_column: Optional[str] = None
    datetime_column: Optional[str] = None


@router.post("/upload")
def upload_dataset(file: UploadFile = File(...)):
    """
    Uploads and validates electricity consumption CSV.
    Returns preview of first 10 rows, dimensions, and auto-detected columns.
    """
    if not file.filename.lower().endswith(".csv") and not file.filename.lower().endswith(".txt"):
        raise HTTPException(status_code=400, detail="Only CSV or semicolon-separated text files are supported.")

    raw_dir = DATA_DIR / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    destination_path = raw_dir / file.filename

    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    try:
        metadata = state_manager.save_raw_upload(str(destination_path), file.filename)
        return metadata
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Invalid dataset: {str(e)}")


@router.post("/dataset/load-sample")
@router.post("/upload/sample")
def load_sample_dataset():
    """
    Loads the bundled UCI Individual Household Electric Power sample dataset for quick demo.
    """
    if not os.path.exists(SAMPLE_DATA_PATH):
        raise HTTPException(status_code=404, detail="Default sample dataset not found on disk.")

    try:
        metadata = state_manager.save_raw_upload(str(SAMPLE_DATA_PATH), "household_power_sample.csv")
        return metadata
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading sample dataset: {str(e)}")


@router.get("/dataset/preview")
def get_dataset_preview():
    """
    Returns preview of the first 10 rows, columns, and dimensions.
    """
    if not state_manager.is_dataset_loaded:
        # If sample dataset exists, auto-load it as default preview
        if os.path.exists(SAMPLE_DATA_PATH):
            try:
                state_manager.save_raw_upload(str(SAMPLE_DATA_PATH), "household_power_sample.csv")
            except Exception:
                pass

    if not state_manager.is_dataset_loaded:
        return {
            "is_loaded": False,
            "filename": None,
            "total_rows": 0,
            "total_columns": 0,
            "columns": [],
            "preview": [],
            "detected_target_col": None,
            "is_preprocessed": False
        }

    meta = state_manager.raw_metadata
    return {
        "is_loaded": True,
        "filename": meta.get("filename"),
        "total_rows": meta.get("total_rows", 0),
        "total_columns": meta.get("total_columns", 0),
        "columns": meta.get("columns", []),
        "numeric_columns": meta.get("numeric_columns", []),
        "preview": meta.get("preview", []),
        "detected_target_col": meta.get("detected_target_col"),
        "detected_datetime_col": meta.get("detected_datetime_col"),
        "is_preprocessed": state_manager.is_preprocessed,
        "preprocessing_summary": state_manager.preprocessing_summary if state_manager.is_preprocessed else None
    }


@router.post("/preprocess")
def preprocess_dataset(req: Optional[PreprocessRequest] = None):
    """
    Cleans dataset, resamples to hourly mean, splits chronologically (70/15/15),
    and fits MinMaxScaler on training partition only.
    """
    target_col = req.target_column if req else None
    dt_col = req.datetime_column if req else None

    try:
        summary = state_manager.preprocess_dataset(target_col=target_col, timestamp_col=dt_col)
        return summary
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Preprocessing error: {str(e)}")
