import os
import joblib
import numpy as np
import pandas as pd
from typing import Tuple, Dict, Any, Optional
from sklearn.preprocessing import MinMaxScaler
from app.core.config import (
    SEQUENCE_LENGTH, TRAIN_SPLIT, VAL_SPLIT, TEST_SPLIT,
    SCALERS_DIR, PROCESSED_DATA_PATH
)

def parse_uci_or_standard_csv(
    file_path: str,
    timestamp_col: Optional[str] = None,
    consumption_col: Optional[str] = None,
    sep: Optional[str] = None
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Parses and cleans raw power consumption dataset.
    Detects delimiter (';' or ','), handles '?' as NaN, parses timestamp,
    sorts by timestamp, validates columns, and reports data hygiene stats.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Data file not found at: {file_path}")

    # Auto-detect delimiter if not provided
    if sep is None:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            first_line = f.readline()
            sep = ';' if ';' in first_line else ','

    df_raw = pd.read_csv(
        file_path,
        sep=sep,
        na_values=['?', 'NA', 'null', 'None', ''],
        low_memory=False
    )

    initial_rows = len(df_raw)
    if initial_rows == 0:
        raise ValueError("Uploaded CSV is empty.")

    # Identify timestamp columns
    timestamp_series = None
    if timestamp_col and timestamp_col in df_raw.columns:
        timestamp_series = pd.to_datetime(df_raw[timestamp_col], errors='coerce')
    elif 'Date' in df_raw.columns and 'Time' in df_raw.columns:
        # Standard UCI format: '16/12/2006' and '17:24:00'
        # Parse dayfirst=True for DD/MM/YYYY
        dt_strings = df_raw['Date'].astype(str) + ' ' + df_raw['Time'].astype(str)
        timestamp_series = pd.to_datetime(dt_strings, format="%d/%m/%Y %H:%M:%S", errors='coerce')
        if timestamp_series.isna().sum() > initial_rows * 0.5:
            # Fallback to general parser
            timestamp_series = pd.to_datetime(dt_strings, errors='coerce', dayfirst=True)
    elif 'timestamp' in df_raw.columns:
        timestamp_series = pd.to_datetime(df_raw['timestamp'], errors='coerce')
    elif 'datetime' in df_raw.columns:
        timestamp_series = pd.to_datetime(df_raw['datetime'], errors='coerce')
    else:
        # Attempt to find first column with datetime data
        found = False
        for col in df_raw.columns:
            try:
                sample_parsed = pd.to_datetime(df_raw[col].iloc[:50], errors='coerce')
                if sample_parsed.notna().sum() > 40:
                    timestamp_series = pd.to_datetime(df_raw[col], errors='coerce')
                    found = True
                    break
            except Exception:
                continue
        if not found or timestamp_series is None:
            raise ValueError("Could not automatically identify a timestamp column. Please specify.")

    # Identify consumption column
    target_col = None
    if consumption_col and consumption_col in df_raw.columns:
        target_col = consumption_col
    elif 'Global_active_power' in df_raw.columns:
        target_col = 'Global_active_power'
    elif 'consumption' in df_raw.columns:
        target_col = 'consumption'
    elif 'power' in df_raw.columns:
        target_col = 'power'
    else:
        # Pick the first numeric column
        for col in df_raw.columns:
            if col in ['Date', 'Time', 'timestamp', 'datetime']:
                continue
            numeric_test = pd.to_numeric(df_raw[col], errors='coerce')
            if numeric_test.notna().sum() > initial_rows * 0.5:
                target_col = col
                break
        if not target_col:
            raise ValueError("Could not find a valid numeric power consumption column.")

    consumption_series = pd.to_numeric(df_raw[target_col], errors='coerce')

    # Build clean frame
    cleaned = pd.DataFrame({
        'timestamp': timestamp_series,
        'Global_active_power': consumption_series
    })

    # Data hygiene checks
    invalid_dates_count = int(cleaned['timestamp'].isna().sum())
    cleaned = cleaned.dropna(subset=['timestamp'])
    
    # Sort chronologically
    cleaned = cleaned.sort_values('timestamp')
    
    # Detect duplicate timestamps
    duplicates_count = int(cleaned.duplicated(subset=['timestamp']).sum())
    if duplicates_count > 0:
        cleaned = cleaned.drop_duplicates(subset=['timestamp'], keep='first')

    missing_consumption_count = int(cleaned['Global_active_power'].isna().sum())

    metadata = {
        "initial_rows": initial_rows,
        "valid_rows_after_cleaning": len(cleaned),
        "invalid_timestamps": invalid_dates_count,
        "duplicate_timestamps": duplicates_count,
        "missing_consumption_raw": missing_consumption_count,
        "raw_columns": list(df_raw.columns),
        "target_col": target_col,
        "delimiter": sep
    }

    return cleaned, metadata


def resample_to_hourly_mean(cleaned_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Resamples minute or irregular readings to 1-hour intervals (mean).
    Handles short gaps conservatively by linear interpolation (up to 2 hours),
    flags longer gaps, and drops remaining unfillable NaNs without fabricating data.
    """
    df = cleaned_df.copy().set_index('timestamp')
    
    # Resample to 1H mean
    hourly = df['Global_active_power'].resample('1h').mean()
    
    # Calculate missing hours created by gaps
    missing_hours_before_interp = int(hourly.isna().sum())
    
    # Conservative interpolation: interpolate ONLY short gaps of at most 2 consecutive hours
    # Avoids inventing fictitious data across long power outages / sensor shutdowns
    hourly_filled = hourly.interpolate(method='time', limit=2)
    
    # Drop any remaining unfillable hours (e.g. at the edges or gaps > 2 hours)
    hourly_clean = hourly_filled.dropna().to_frame()
    hourly_clean.reset_index(inplace=True)

    stats = {
        "total_hourly_records": len(hourly_clean),
        "start_timestamp": str(hourly_clean['timestamp'].min()),
        "end_timestamp": str(hourly_clean['timestamp'].max()),
        "missing_hours_detected": missing_hours_before_interp,
        "interpolated_short_gaps": missing_hours_before_interp - int(hourly_clean['Global_active_power'].isna().sum()),
        "mean_kw": float(hourly_clean['Global_active_power'].mean()),
        "max_kw": float(hourly_clean['Global_active_power'].max()),
        "min_kw": float(hourly_clean['Global_active_power'].min()),
    }
    
    return hourly_clean, stats


def prepare_dataset_splits(
    hourly_df: pd.DataFrame,
    seq_length: int = SEQUENCE_LENGTH,
    train_ratio: float = TRAIN_SPLIT,
    val_ratio: float = VAL_SPLIT,
    test_ratio: float = TEST_SPLIT
) -> Tuple[Dict[str, np.ndarray], MinMaxScaler, Dict[str, Any]]:
    """
    Performs strict chronological train/validation/test split.
    FITS MINMAXSCALER ON TRAINING DATA ONLY to prevent data leakage!
    Generates 24-hour input sequences to predict next hour: (samples, 24, 1) and (samples, 1).
    """
    n = len(hourly_df)
    if n < (seq_length + 10):
        raise ValueError(f"Insufficient hourly data points ({n}). Need at least {seq_length + 10} records.")

    values = hourly_df['Global_active_power'].values.reshape(-1, 1)
    timestamps = hourly_df['timestamp'].values

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_vals = values[:train_end]
    val_vals = values[train_end:val_end]
    test_vals = values[val_end:]

    # STRICT: Fit scaler ONLY on train_vals
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train_vals)

    # Save fitted scaler
    os.makedirs(SCALERS_DIR, exist_ok=True)
    scaler_path = os.path.join(SCALERS_DIR, "minmax_scaler.joblib")
    joblib.dump(scaler, scaler_path)

    # Transform all splits using the train-fitted scaler
    scaled_values = scaler.transform(values)

    def build_windowed_sequences(scaled_arr, start_idx, end_idx):
        X, y, ts_y = [], [], []
        # We need sequence from i - seq_length to i to predict at i
        # To ensure continuous boundary across splits, we can use preceding history
        for i in range(max(seq_length, start_idx), end_idx):
            X.append(scaled_arr[i - seq_length:i])
            y.append(scaled_arr[i, 0])
            ts_y.append(timestamps[i])
        return np.array(X), np.array(y).reshape(-1, 1), np.array(ts_y)

    X_train, y_train, ts_train = build_windowed_sequences(scaled_values, 0, train_end)
    X_val, y_val, ts_val = build_windowed_sequences(scaled_values, train_end, val_end)
    X_test, y_test, ts_test = build_windowed_sequences(scaled_values, val_end, n)

    splits = {
        "X_train": X_train,
        "y_train": y_train,
        "ts_train": ts_train,
        "X_val": X_val,
        "y_val": y_val,
        "ts_val": ts_val,
        "X_test": X_test,
        "y_test": y_test,
        "ts_test": ts_test,
        "raw_train": train_vals,
        "raw_val": val_vals,
        "raw_test": test_vals
    }

    split_info = {
        "train_samples": len(X_train),
        "val_samples": len(X_val),
        "test_samples": len(X_test),
        "input_shape": list(X_train.shape),
        "scaler_data_min": float(scaler.data_min_[0]),
        "scaler_data_max": float(scaler.data_max_[0]),
        "train_start_date": str(timestamps[0]),
        "train_end_date": str(timestamps[train_end - 1]),
        "val_start_date": str(timestamps[train_end]),
        "val_end_date": str(timestamps[val_end - 1]),
        "test_start_date": str(timestamps[val_end]),
        "test_end_date": str(timestamps[-1])
    }

    return splits, scaler, split_info
