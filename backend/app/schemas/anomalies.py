from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class AnomalyEvent(BaseModel):
    id: str
    timestamp: str
    actual: float
    predicted: float
    residual: float
    threshold: float
    severity: str  # "Critical", "High", "Moderate"
    percentile: float
    reviewed: bool = False
    review_notes: Optional[str] = None

class AnomaliesResponse(BaseModel):
    total_anomalies: int
    unreviewed_count: int
    reviewed_count: int
    severity_counts: Dict[str, int]
    threshold_percentile: float
    threshold_value_kw: float
    anomalies: List[AnomalyEvent]
    series_sample: List[Dict[str, Any]] = []
    disclaimer: str = (
        "Statistical Note: Flagged anomalies represent time periods where observed power consumption "
        "substantially exceeded model predictions beyond the validation-derived threshold. "
        "These are statistical indicators of unusual demand, not definitive proof of equipment malfunction or waste."
    )

class AnomalyReviewRequest(BaseModel):
    anomaly_id: str
    reviewed: bool
    notes: Optional[str] = None

class RecalculateAnomaliesRequest(BaseModel):
    percentile: float = Field(default=95.0, ge=80.0, le=99.9)
