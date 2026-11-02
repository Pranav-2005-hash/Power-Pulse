import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple

def compute_validation_thresholds(val_residuals: np.ndarray) -> Dict[float, float]:
    """
    Computes anomaly thresholds strictly from VALIDATION positive residuals only.
    Prevents data leakage by ensuring test set is NEVER used to tune thresholds.
    Residual = Actual - Predicted.
    """
    val_res = np.array(val_residuals)
    # Consider only positive residuals (actual consumption exceeding predictions)
    pos_residuals = val_res[val_res > 0]
    
    if len(pos_residuals) == 0:
        # Fallback to standard deviation if no positive residuals exist
        std_val = float(np.std(val_res)) if len(val_res) > 0 else 0.5
        return {90.0: std_val * 1.645, 95.0: std_val * 1.96, 99.0: std_val * 2.576}

    p90 = float(np.percentile(pos_residuals, 90))
    p95 = float(np.percentile(pos_residuals, 95))
    p99 = float(np.percentile(pos_residuals, 99))

    return {
        90.0: round(p90, 4),
        95.0: round(p95, 4),
        99.0: round(p99, 4)
    }

def detect_anomalies_from_residuals(
    timestamps: List[str],
    actuals: List[float],
    predictions: List[float],
    validation_thresholds: Dict[float, float],
    percentile: float = 95.0,
    reviewed_ids: Dict[str, bool] = None
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Flags anomalies where positive residual exceeds validation-derived percentile threshold.
    Classifies into Critical (>= p99), High (>= p95), and Moderate (>= p90).
    """
    reviewed_ids = reviewed_ids or {}
    
    # Selected threshold value
    thresh_val = validation_thresholds.get(float(percentile))
    if thresh_val is None:
        # Closest match or interpolate
        thresh_val = validation_thresholds.get(95.0, 0.5)

    p99_val = validation_thresholds.get(99.0, thresh_val * 1.3)
    p95_val = validation_thresholds.get(95.0, thresh_val)

    anomalies = []
    severity_counts = {"Critical": 0, "High": 0, "Moderate": 0}

    for idx, (ts, act, pred) in enumerate(zip(timestamps, actuals, predictions)):
        residual = act - pred
        if residual > thresh_val:
            # Determine severity
            if residual >= p99_val:
                sev = "Critical"
            elif residual >= p95_val:
                sev = "High"
            else:
                sev = "Moderate"

            severity_counts[sev] += 1

            event_id = f"ANM-{idx}-{str(ts).replace(' ', 'T').replace(':', '')}"
            is_reviewed = reviewed_ids.get(event_id, False)

            anomalies.append({
                "id": event_id,
                "timestamp": str(ts),
                "actual": round(float(act), 3),
                "predicted": round(float(pred), 3),
                "residual": round(float(residual), 3),
                "threshold": round(float(thresh_val), 3),
                "severity": sev,
                "percentile": float(percentile),
                "reviewed": is_reviewed,
                "review_notes": None
            })

    # Sort anomalies descending by timestamp / residual
    anomalies.sort(key=lambda x: x["timestamp"], reverse=True)

    summary = {
        "total_anomalies": len(anomalies),
        "unreviewed_count": sum(1 for a in anomalies if not a["reviewed"]),
        "reviewed_count": sum(1 for a in anomalies if a["reviewed"]),
        "severity_counts": severity_counts,
        "threshold_percentile": float(percentile),
        "threshold_value_kw": round(float(thresh_val), 4)
    }

    return anomalies, summary
