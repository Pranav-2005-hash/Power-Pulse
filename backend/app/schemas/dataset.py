from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class DatasetStatusResponse(BaseModel):
    is_loaded: bool
    filename: Optional[str] = None
    source_type: str = "none"  # "sample", "uploaded", "none"
    total_raw_rows: int = 0
    total_hourly_rows: int = 0
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    missing_values_handled: int = 0
    sampling_rate: str = "1 hour (mean resampled)"
    target_column: str = "Global_active_power"
    units: str = "kW"
    columns: List[str] = []
    models_trained: bool = False
    last_updated: Optional[str] = None

class DatasetPreviewResponse(BaseModel):
    columns: List[str]
    rows: List[Dict[str, Any]]
    total_rows: int
    page: int = 1
    page_size: int = 50

class UploadResponse(BaseModel):
    success: bool
    message: str
    filename: str
    raw_rows: int
    hourly_rows: int
    start_date: str
    end_date: str
    duplicate_timestamps: int
    missing_values_count: int
