from fastapi import APIRouter, HTTPException, Query
from typing import Optional

from app.storage.state_manager import state_manager
from app.schemas.anomalies import (
    AnomaliesResponse, AnomalyReviewRequest, RecalculateAnomaliesRequest
)
from app.ml.anomaly import detect_anomalies_from_residuals

router = APIRouter(prefix="/anomalies", tags=["Anomalies"])

@router.get("", response_model=AnomaliesResponse)
def get_anomalies(
    threshold_percentile: float = Query(95.0, description="Percentile threshold (90, 95, 99)"),
    status: Optional[str] = Query("all", description="'all', 'reviewed', or 'unreviewed'")
):
    if not state_manager.models_trained or not state_manager.best_model_name:
        raise HTTPException(
            status_code=400,
            detail="Models must be trained first before anomaly detection can be performed."
        )

    splits = state_manager.splits
    best_m = state_manager.metrics.get(state_manager.best_model_name, {})
    test_actuals = best_m.get("test_actuals", [])
    test_preds = best_m.get("test_predictions", [])
    test_ts = [str(t) for t in splits["ts_test"]]

    anomalies, summary = detect_anomalies_from_residuals(
        timestamps=test_ts,
        actuals=test_actuals,
        predictions=test_preds,
        validation_thresholds=state_manager.validation_thresholds,
        percentile=threshold_percentile,
        reviewed_ids=state_manager.anomaly_reviews
    )

    # Filter by review status if requested
    if status == "reviewed":
        anomalies = [a for a in anomalies if a["reviewed"]]
    elif status == "unreviewed":
        anomalies = [a for a in anomalies if not a["reviewed"]]

    # Build chart series sample (last 120 test points for the Actual vs Predicted chart with anomaly markers)
    series_sample = []
    anomaly_timestamps = {a["timestamp"]: a["severity"] for a in anomalies}
    
    sample_window = min(120, len(test_ts))
    start_idx = len(test_ts) - sample_window

    for i in range(start_idx, len(test_ts)):
        ts = test_ts[i]
        act = round(float(test_actuals[i]), 3)
        pred = round(float(test_preds[i]), 3)
        is_anom = ts in anomaly_timestamps
        series_sample.append({
            "timestamp": ts,
            "actual": act,
            "predicted": pred,
            "residual": round(act - pred, 3),
            "is_anomaly": is_anom,
            "severity": anomaly_timestamps.get(ts)
        })

    return AnomaliesResponse(
        total_anomalies=summary["total_anomalies"],
        unreviewed_count=summary["unreviewed_count"],
        reviewed_count=summary["reviewed_count"],
        severity_counts=summary["severity_counts"],
        threshold_percentile=threshold_percentile,
        threshold_value_kw=summary["threshold_value_kw"],
        anomalies=anomalies,
        series_sample=series_sample
    )

@router.post("/recalculate", response_model=AnomaliesResponse)
def recalculate_anomalies(req: RecalculateAnomaliesRequest):
    return get_anomalies(threshold_percentile=req.percentile, status="all")

@router.post("/review")
def review_anomaly(req: AnomalyReviewRequest):
    state_manager.save_review(req.anomaly_id, req.reviewed)
    return {
        "success": True,
        "anomaly_id": req.anomaly_id,
        "reviewed": req.reviewed,
        "message": f"Anomaly {req.anomaly_id} status updated to {'Reviewed' if req.reviewed else 'Unreviewed'}."
    }
