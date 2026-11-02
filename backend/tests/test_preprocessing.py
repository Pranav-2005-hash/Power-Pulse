import pytest
import os
import numpy as np
import pandas as pd
from app.ml.preprocessing import (
    parse_uci_or_standard_csv, resample_to_hourly_mean, prepare_dataset_splits
)

def test_timestamp_parsing_and_cleaning(tmp_path):
    # Create test CSV with standard UCI semicolon format and '?' missing values
    csv_file = tmp_path / "test_uci.csv"
    csv_file.write_text(
        "Date;Time;Global_active_power;Voltage\n"
        "16/12/2025;07:00:00;2.345;240.5\n"
        "16/12/2025;07:15:00;?;240.2\n"  # '?' missing value
        "16/12/2025;07:30:00;3.120;239.8\n"
        "16/12/2025;07:45:00;2.890;241.0\n"
        "16/12/2025;07:45:00;2.890;241.0\n"  # Duplicate timestamp
    )

    cleaned, metadata = parse_uci_or_standard_csv(str(csv_file))
    assert metadata["initial_rows"] == 5
    assert metadata["duplicate_timestamps"] == 1
    assert metadata["missing_consumption_raw"] == 1
    assert len(cleaned) == 4  # Dropped duplicate
    assert cleaned['Global_active_power'].iloc[0] == pytest.approx(2.345)

def test_invalid_csv_rejection(tmp_path):
    csv_file = tmp_path / "bad.csv"
    csv_file.write_text("Col1,Col2,Col3\nfoo,bar,baz\n")
    with pytest.raises(ValueError):
        parse_uci_or_standard_csv(str(csv_file))

def test_hourly_resampling_and_gap_handling():
    # 4 readings in 1 hour
    dates = pd.date_range("2025-01-01 00:00:00", periods=4, freq="15min")
    df = pd.DataFrame({
        "timestamp": dates,
        "Global_active_power": [1.0, 2.0, 3.0, 4.0]
    })
    hourly, stats = resample_to_hourly_mean(df)
    assert len(hourly) == 1
    assert hourly["Global_active_power"].iloc[0] == pytest.approx(2.5)
    assert stats["total_hourly_records"] == 1

def test_chronological_split_no_scaler_leakage():
    # Test strict chronological split and scaler fit on train ONLY
    n = 100
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    # Monotonically increasing values: train has min 0, max 69. Test has max 99.
    values = np.linspace(10.0, 100.0, n)
    df = pd.DataFrame({"timestamp": dates, "Global_active_power": values})

    splits, scaler, info = prepare_dataset_splits(df, seq_length=12, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    # Train end is index 70.
    # Scaler min must be train min (10.0), scaler max must be train max, NOT the test max (100.0)!
    train_max = float(values[:70].max())
    train_min = float(values[:70].min())
    assert scaler.data_min_[0] == pytest.approx(train_min)
    assert scaler.data_max_[0] == pytest.approx(train_max)
    assert scaler.data_max_[0] < 100.0  # Confirms NO LEAKAGE from test set!

def test_sequence_generator_shapes():
    n = 60
    dates = pd.date_range("2025-01-01", periods=n, freq="1h")
    values = np.random.uniform(1.0, 4.0, n)
    df = pd.DataFrame({"timestamp": dates, "Global_active_power": values})

    splits, scaler, info = prepare_dataset_splits(df, seq_length=24, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
    
    X_train = splits["X_train"]
    y_train = splits["y_train"]
    # Input shape: (samples, 24, 1)
    assert len(X_train.shape) == 3
    assert X_train.shape[1] == 24
    assert X_train.shape[2] == 1
    assert y_train.shape[1] == 1
