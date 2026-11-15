import pytest
import numpy as np
from app.ml.anomaly import compute_validation_thresholds, detect_anomalies_from_residuals
from app.storage.state_manager import state_manager

def test_validation_threshold_calculation():
    # Residuals: negative, zero, positive
    val_residuals = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 1.2, 2.5])
    thresholds = compute_validation_thresholds(val_residuals)
    
    assert 90.0 in thresholds
    assert 95.0 in thresholds
    assert 99.0 in thresholds
    # Monotonicity: 99th >= 95th >= 90th
    assert thresholds[99.0] >= thresholds[95.0] >= thresholds[90.0]
    # Positives only should be used
    assert thresholds[90.0] > 0.0

def test_anomaly_detection_and_severity():
    timestamps = ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-01 12:00:00"]
    actuals = [1.5, 4.0, 6.5]
    predictions = [1.4, 1.5, 1.5]  # positive residuals: 0.1, 2.5, 5.0
    
    val_thresholds = {
        90.0: 1.0,
        95.0: 2.0,
        99.0: 4.0
    }

    anomalies, summary = detect_anomalies_from_residuals(
        timestamps=timestamps,
        actuals=actuals,
        predictions=predictions,
        validation_thresholds=val_thresholds,
        percentile=95.0
    )

    # Residuals > 2.0 (95th): 2.5 (High) and 5.0 (Critical)
    assert len(anomalies) == 2
    assert summary["severity_counts"]["Critical"] == 1
    assert summary["severity_counts"]["High"] == 1
    assert summary["severity_counts"]["Moderate"] == 0
    assert summary["total_anomalies"] == 2
    assert summary["threshold_value_kw"] == 2.0

def test_academic_anomaly_disclaimer():
    res = state_manager.get_anomalies()
    assert "disclaimer" in res
    assert "not proof of electricity wastage" in res["disclaimer"]
