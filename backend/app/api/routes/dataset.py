import os
import shutil
from fastapi import APIRouter, UploadFile, File, Form, HTTPException, Query
from typing import Optional

from app.core.config import SAMPLE_DATA_PATH, DATA_DIR
from app.storage.state_manager import state_manager
from app.schemas.dataset import DatasetStatusResponse, DatasetPreviewResponse, UploadResponse

router = APIRouter(prefix="/dataset", tags=["Dataset"])

@router.get("/status", response_model=DatasetStatusResponse)
def get_dataset_status():
    if not state_manager.is_dataset_loaded or state_manager.hourly_df is None:
        return DatasetStatusResponse(
            is_loaded=False,
            source_type="none",
            models_trained=False
        )

    df = state_manager.hourly_df
    return DatasetStatusResponse(
        is_loaded=True,
        filename=state_manager.dataset_filename,
        source_type=state_manager.source_type,
        total_raw_rows=state_manager.raw_metadata.get("initial_rows", len(df)),
        total_hourly_rows=len(df),
        start_date=str(df['timestamp'].min()),
        end_date=str(df['timestamp'].max()),
        missing_values_handled=state_manager.resample_stats.get("missing_hours_detected", 0),
        sampling_rate="1 hour (mean resampled)",
        target_column="Global_active_power",
        units="kW",
        columns=["timestamp", "Global_active_power"],
        models_trained=state_manager.models_trained,
        last_updated=str(df['timestamp'].max())
    )

@router.post("/load-sample", response_model=DatasetStatusResponse)
def load_sample_dataset():
    if not os.path.exists(SAMPLE_DATA_PATH):
        # Generate sample if missing
        from app.services.sample_generator import generate_sample_dataset
        generate_sample_dataset(str(SAMPLE_DATA_PATH), days=90)

    try:
        state_manager.load_dataset_from_file(
            file_path=str(SAMPLE_DATA_PATH),
            source_type="sample"
        )
        return get_dataset_status()
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load sample dataset: {str(e)}")

@router.post("/upload", response_model=UploadResponse)
async def upload_dataset(
    file: UploadFile = File(...),
    timestamp_column: Optional[str] = Form(None),
    consumption_column: Optional[str] = Form(None)
):
    if not file.filename.endswith(('.csv', '.txt')):
        raise HTTPException(status_code=400, detail="Only .csv and .txt files are supported.")

    raw_dir = DATA_DIR / "raw"
    os.makedirs(raw_dir, exist_ok=True)
    destination_path = raw_dir / file.filename

    # Save uploaded file
    try:
        with open(destination_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving uploaded file: {str(e)}")

    try:
        load_result = state_manager.load_dataset_from_file(
            file_path=str(destination_path),
            source_type="uploaded",
            timestamp_col=timestamp_column,
            consumption_col=consumption_column
        )
        raw_meta = load_result["raw_metadata"]
        res_stats = load_result["resample_stats"]

        return UploadResponse(
            success=True,
            message="Dataset uploaded, validated, cleaned, and resampled to hourly mean successfully.",
            filename=file.filename,
            raw_rows=raw_meta["initial_rows"],
            hourly_rows=res_stats["total_hourly_records"],
            start_date=res_stats["start_timestamp"],
            end_date=res_stats["end_timestamp"],
            duplicate_timestamps=raw_meta["duplicate_timestamps"],
            missing_values_count=raw_meta["missing_consumption_raw"]
        )
    except Exception as e:
        if os.path.exists(destination_path):
            os.remove(destination_path)
        raise HTTPException(status_code=422, detail=f"Validation failed on uploaded file: {str(e)}")

@router.get("/preview", response_model=DatasetPreviewResponse)
def get_dataset_preview(
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=10, le=200)
):
    if not state_manager.is_dataset_loaded or state_manager.hourly_df is None:
        raise HTTPException(status_code=400, detail="No dataset loaded. Please load sample data or upload CSV.")

    df = state_manager.hourly_df
    total = len(df)
    start_idx = (page - 1) * page_size
    end_idx = min(start_idx + page_size, total)

    slice_df = df.iloc[start_idx:end_idx].copy()
    slice_df['timestamp'] = slice_df['timestamp'].astype(str)
    slice_df['Global_active_power'] = slice_df['Global_active_power'].round(3)

    return DatasetPreviewResponse(
        columns=["timestamp", "Global_active_power"],
        rows=slice_df.to_dict(orient="records"),
        total_rows=total,
        page=page,
        page_size=page_size
    )
