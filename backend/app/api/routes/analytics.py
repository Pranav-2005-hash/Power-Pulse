import numpy as np
import pandas as pd
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List

from app.storage.state_manager import state_manager
from app.schemas.analytics import (
    AnalyticsSummaryResponse, DescriptiveStats, ConsumptionPoint,
    PatternsResponse, HourlyPattern, DayOfWeekPattern
)

router = APIRouter(prefix="/analytics", tags=["Analytics"])

@router.get("/summary", response_model=AnalyticsSummaryResponse)
def get_analytics_summary():
    if not state_manager.is_dataset_loaded or state_manager.hourly_df is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")

    df = state_manager.hourly_df
    series = df['Global_active_power']

    q25 = float(series.quantile(0.25))
    median = float(series.median())
    q75 = float(series.quantile(0.75))
    iqr = float(q75 - q25)
    mean_val = float(series.mean())
    std_val = float(series.std())
    skew_val = float(series.skew())

    max_idx = series.idxmax()
    min_idx = series.idxmin()

    peak_kw = float(series.loc[max_idx])
    peak_ts = str(df.loc[max_idx, 'timestamp'])
    lowest_kw = float(series.loc[min_idx])
    lowest_ts = str(df.loc[min_idx, 'timestamp'])

    # Hourly kW summed = total kWh
    total_kwh = float(series.sum())
    total_days = max(1.0, len(series) / 24.0)
    avg_daily_kwh = float(total_kwh / total_days)

    recent_mean = float(series.iloc[-24:].mean()) if len(series) >= 24 else mean_val
    overall_mean = mean_val
    diff_pct = ((recent_mean - overall_mean) / overall_mean) * 100.0 if overall_mean > 0 else 0.0

    if diff_pct > 5.0:
        trend = f"Elevated (+{diff_pct:.1f}% vs baseline)"
    elif diff_pct < -5.0:
        trend = f"Conserving ({diff_pct:.1f}% vs baseline)"
    else:
        trend = "Stable (within ±5% baseline)"

    return AnalyticsSummaryResponse(
        stats=DescriptiveStats(
            count=len(series),
            mean=round(mean_val, 3),
            std=round(std_val, 3),
            min=round(float(series.min()), 3),
            p25=round(q25, 3),
            median=round(median, 3),
            p75=round(q75, 3),
            max=round(float(series.max()), 3),
            skewness=round(skew_val, 3),
            iqr=round(iqr, 3)
        ),
        peak_consumption=round(peak_kw, 3),
        peak_timestamp=peak_ts,
        lowest_consumption=round(lowest_kw, 3),
        lowest_timestamp=lowest_ts,
        total_energy_kwh=round(total_kwh, 2),
        avg_daily_kwh=round(avg_daily_kwh, 2),
        current_trend=trend
    )

@router.get("/consumption", response_model=List[ConsumptionPoint])
def get_consumption_series(
    start_date: Optional[str] = Query(None),
    end_date: Optional[str] = Query(None),
    limit: Optional[int] = Query(1000, ge=10, le=10000)
):
    if not state_manager.is_dataset_loaded or state_manager.hourly_df is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")

    df = state_manager.hourly_df.copy()
    
    if start_date:
        df = df[df['timestamp'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['timestamp'] <= pd.to_datetime(end_date)]

    if len(df) == 0:
        return []

    # Compute rolling averages
    df['rolling_6h'] = df['Global_active_power'].rolling(window=6, min_periods=1).mean()
    df['rolling_24h'] = df['Global_active_power'].rolling(window=24, min_periods=1).mean()

    # If exceeding limit, subsample smoothly
    if len(df) > limit:
        step = int(np.ceil(len(df) / limit))
        df = df.iloc[::step]

    results = []
    for _, row in df.iterrows():
        results.append(ConsumptionPoint(
            timestamp=str(row['timestamp']),
            actual=round(float(row['Global_active_power']), 3),
            rolling_6h=round(float(row['rolling_6h']), 3) if pd.notna(row['rolling_6h']) else None,
            rolling_24h=round(float(row['rolling_24h']), 3) if pd.notna(row['rolling_24h']) else None
        ))

    return results

@router.get("/patterns", response_model=PatternsResponse)
def get_consumption_patterns():
    if not state_manager.is_dataset_loaded or state_manager.hourly_df is None:
        raise HTTPException(status_code=400, detail="Dataset not loaded.")

    df = state_manager.hourly_df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    day_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    # Hourly aggregation
    hourly_group = df.groupby('hour')['Global_active_power']
    hourly_patterns = []
    for h in range(24):
        if h in hourly_group.groups:
            vals = hourly_group.get_group(h)
            hourly_patterns.append(HourlyPattern(
                hour=h,
                mean_kw=round(float(vals.mean()), 3),
                std_kw=round(float(vals.std()), 3),
                min_kw=round(float(vals.min()), 3),
                max_kw=round(float(vals.max()), 3)
            ))

    # Daily aggregation
    daily_group = df.groupby('day_of_week')['Global_active_power']
    daily_patterns = []
    for d in range(7):
        if d in daily_group.groups:
            d_vals = daily_group.get_group(d)
            # Find peak hour for this day
            d_df = df[df['day_of_week'] == d]
            peak_h = int(d_df.groupby('hour')['Global_active_power'].mean().idxmax())
            daily_patterns.append(DayOfWeekPattern(
                day_index=d,
                day_name=day_names[d],
                mean_kw=round(float(d_vals.mean()), 3),
                std_kw=round(float(d_vals.std()), 3),
                peak_hour=peak_h
            ))

    peak_h_overall = max(hourly_patterns, key=lambda x: x.mean_kw).hour if hourly_patterns else 20
    peak_d_overall = max(daily_patterns, key=lambda x: x.mean_kw).day_name if daily_patterns else "Sunday"

    return PatternsResponse(
        hourly=hourly_patterns,
        daily=daily_patterns,
        peak_hour_overall=peak_h_overall,
        peak_day_overall=peak_d_overall
    )
