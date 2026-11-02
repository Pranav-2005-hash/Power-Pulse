import pytest
import numpy as np
from app.ml.anomaly import compute_validation_thresholds, detect_anomalies_from_residuals

def test_validation_threshold_calculation():
    # Residuals: some negative, some positive
    val_residuals = np.array([-0.5, -0.2, 0.0, 0.1, 0.3, 0.6, 1.2, 2.5])
    thresholds = compute_validation_thresholds(val_residuals)
    
    assert 90.0 in thresholds
    assert 95.0 in thresholds
    assert 99.0 in thresholds
    # 99th percentile must be >= 95th >= 90th
    assert thresholds[99.0] >= thresholds[95.0] >= thresholds[90.0]
    # Positives only should be used
    assert thresholds[90.0] > 0.0

def test_anomaly_detection_and_severity():
    timestamps = ["2025-01-01 10:00:00", "2025-01-01 11:00:00", "2025-01-01 12:00:00"]
    actuals = [1.5, 4.0, 6.5]
    predictions = [1.4, 1.5, 1.5]  # residuals: 0.1, 2.5, 5.0
    
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

    # Residuals > 2.0: 2.5 (High) and 5.0 (Critical)
    assert len(anomalies) == 2
    assert summary["severity_counts"]["Critical"] == 1
    assert summary["severity_counts"]["High"] == 1
    assert summary["severity_counts"]["Moderate"] == 0
    assert summary["total_anomalies"] == 2

def test_anomaly_review_persistence():
    from app.storage.state_manager import state_manager
    test_id = "ANM-TEST-123"
    state_manager.save_review(test_id, True)
    assert state_manager.anomaly_reviews.get(test_id) is True
    state_manager.save_review(test_id, False)
    assert state_manager.anomaly_reviews.get(test_id) is False
