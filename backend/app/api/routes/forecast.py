import pandas as pd
from fastapi import APIRouter, HTTPException
from typing import List

from app.storage.state_manager import state_manager
from app.schemas.forecast import (
    ForecastRequest, ForecastResponse, ForecastPoint, HistoricalPoint
)
from app.ml.forecasting import recursive_multi_step_forecast

router = APIRouter(prefix="/forecast", tags=["Forecast"])

@router.post("", response_model=ForecastResponse)
def execute_forecast(req: ForecastRequest):
    if not state_manager.is_dataset_loaded or state_manager.hourly_df is None:
        raise HTTPException(status_code=400, detail="Dataset must be loaded before forecasting.")

    if not state_manager.models_trained:
        raise HTTPException(status_code=400, detail="Models are not trained yet. Train models on the Models page first.")

    valid_horizons = [1, 6, 12, 24, 48]
    if req.horizon not in valid_horizons:
        raise HTTPException(status_code=400, detail=f"Horizon must be one of {valid_horizons} hours.")

    hourly = state_manager.hourly_df
    last_raw = hourly['Global_active_power'].values
    last_ts = hourly['timestamp'].iloc[-1]
    scaler = state_manager.scaler

    # Determine which models to run
    models_to_run = []
    req_model_lower = req.model.lower()

    if req_model_lower == "best":
        models_to_run.append(state_manager.best_model_name)
    elif req_model_lower == "lstm":
        models_to_run.append("LSTM")
    elif req_model_lower == "gru":
        models_to_run.append("GRU")
    elif req_model_lower == "both":
        models_to_run.extend(["LSTM", "GRU"])
    else:
        models_to_run.append(state_manager.best_model_name or "LSTM")

    forecast_points: List[ForecastPoint] = []
    metrics_ref = {}

    for m_name in models_to_run:
        model_obj = state_manager.keras_models.get(m_name)
        if not model_obj:
            raise HTTPException(status_code=500, detail=f"Model {m_name} object not loaded in memory.")

        m_metric = state_manager.metrics.get(m_name, {})
        test_rmse = m_metric.get("test_rmse", 0.35)
        metrics_ref[m_name] = {
            "test_mae": m_metric.get("test_mae"),
            "test_rmse": test_rmse,
            "test_r2": m_metric.get("test_r2")
        }

        pts = recursive_multi_step_forecast(
            model=model_obj,
            scaler=scaler,
            last_sequence_raw=last_raw,
            last_timestamp=last_ts,
            horizon_hours=req.horizon,
            test_rmse=test_rmse,
            model_name=m_name
        )

        for p in pts:
            forecast_points.append(ForecastPoint(**p))

    # Grab historical context (last 48 hours for chart context)
    context_df = hourly.iloc[-48:].copy()
    historical_context = [
        HistoricalPoint(
            timestamp=str(row['timestamp']),
            actual=round(float(row['Global_active_power']), 3)
        )
        for _, row in context_df.iterrows()
    ]

    return ForecastResponse(
        horizon_hours=req.horizon,
        models_used=models_to_run,
        forecast_points=forecast_points,
        historical_context=historical_context,
        metrics_reference=metrics_ref
    )
