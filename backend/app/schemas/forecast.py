from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class ForecastRequest(BaseModel):
    model: str = Field(default="best", description="'lstm', 'gru', 'both', or 'best'")
    horizon: int = Field(default=24, description="Forecast horizon in hours (1, 6, 12, 24, 48)")

class HistoricalPoint(BaseModel):
    timestamp: str
    actual: float

class ForecastPoint(BaseModel):
    timestamp: str
    step_ahead: int
    predicted: float
    model: str
    lower_bound: float
    upper_bound: float

class ForecastResponse(BaseModel):
    horizon_hours: int
    models_used: List[str]
    forecast_points: List[ForecastPoint]
    historical_context: List[HistoricalPoint]  # last 24-48 hours
    metrics_reference: Dict[str, Any]
    disclaimer: str = (
        "Predictions are multi-step recursive estimates produced by trained deep learning models "
        "based on historical sequential patterns. Actual future values may differ due to unforeseen "
        "appliance usage, temperature fluctuations, or occupant behavior."
    )
