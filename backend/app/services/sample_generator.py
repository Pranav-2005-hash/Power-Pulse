"""
Sample Dataset Generator for PowerPulse
Generates realistic Household Electric Power Consumption data matching the UCI dataset schema:
Columns: Date;Time;Global_active_power;Global_reactive_power;Voltage;Global_intensity;Sub_metering_1;Sub_metering_2;Sub_metering_3
Features:
- Realistic diurnal cycles (morning and evening peaks)
- Weekday vs weekend patterns
- Realistic voltage (235-245V) and reactive power
- Controlled anomaly spikes for testing anomaly detection
- Rare '?' markers to validate preprocessing parser
"""

import os
import math
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

def generate_sample_dataset(output_path: str, days: int = 90, freq_minutes: int = 15):
    """
    Generates realistic multi-month household power consumption data.
    Uses 15-minute intervals (or 1-minute) to balance realism with fast loading.
    Default 90 days at 15-min intervals gives 8,640 records (resamples cleanly to 2,160 hourly records).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    np.random.seed(42)
    start_date = datetime(2025, 1, 1, 0, 0, 0)
    total_steps = days * 24 * (60 // freq_minutes)
    
    dates = []
    times = []
    active_powers = []
    reactive_powers = []
    voltages = []
    intensities = []
    sub1 = []
    sub2 = []
    sub3 = []
    
    current_dt = start_date
    delta = timedelta(minutes=freq_minutes)
    
    for i in range(total_steps):
        hour = current_dt.hour + current_dt.minute / 60.0
        is_weekend = current_dt.weekday() >= 5
        day_of_year = current_dt.timetuple().tm_yday
        
        # Diurnal pattern
        # Morning peak: 7 - 9 AM
        morning_peak = 1.8 * math.exp(-((hour - 8.0) ** 2) / 2.5)
        # Evening peak: 18 - 22 PM
        evening_peak = 2.4 * math.exp(-((hour - 20.0) ** 2) / 3.5)
        # Base overnight / standby
        base = 0.35 + 0.1 * math.sin(2 * math.pi * day_of_year / 365.0)
        
        # Weekend adjustment (later morning sleep, higher daytime consumption)
        if is_weekend:
            morning_peak = 1.6 * math.exp(-((hour - 9.5) ** 2) / 3.0)
            daytime = 0.6 * math.exp(-((hour - 14.0) ** 2) / 8.0)
        else:
            daytime = 0.2 * math.exp(-((hour - 13.0) ** 2) / 4.0)
            
        noise = np.random.normal(0, 0.12)
        power = base + morning_peak + evening_peak + daytime + noise
        power = max(0.08, power)
        
        # Inject realistic occasional high-demand surges (anomalies) on specific days/hours
        # E.g., unexpected electric heater / EV charging / cooking spikes
        if (i % 380 == 120): # Spaced out anomaly spikes
            power += np.random.uniform(3.2, 5.0)
        elif (i % 640 == 310):
            power += np.random.uniform(4.0, 6.2)
            
        # Reactive power roughly 10-25% of active power
        reactive = max(0.02, power * np.random.uniform(0.08, 0.22) + np.random.normal(0, 0.02))
        
        # Voltage fluctuates around 240V
        voltage = 240.0 + 3.5 * math.sin(hour / 4.0) + np.random.normal(0, 1.2)
        
        # Current intensity in Amperes: I = P / V (roughly)
        intensity = (power * 1000.0) / voltage
        
        # Sub-meterings (kitchen, laundry, HVAC)
        s1 = 0.0 if hour < 6 or hour > 22 else max(0.0, power * 2.5 * np.random.uniform(0.1, 0.4))
        s2 = 0.0 if not (8 <= hour <= 11 or 17 <= hour <= 20) else max(0.0, power * 2.0 * np.random.uniform(0.1, 0.5))
        s3 = max(0.0, power * 8.0 * np.random.uniform(0.2, 0.6))
        
        # Occasional missing values ('?') - approximately 0.2%
        is_missing = (np.random.random() < 0.002)
        
        dates.append(current_dt.strftime("%d/%m/%Y"))
        times.append(current_dt.strftime("%H:%M:%S"))
        
        if is_missing:
            active_powers.append("?")
            reactive_powers.append("?")
            voltages.append("?")
            intensities.append("?")
            sub1.append("?")
            sub2.append("?")
            sub3.append("?")
        else:
            active_powers.append(f"{power:.3f}")
            reactive_powers.append(f"{reactive:.3f}")
            voltages.append(f"{voltage:.2f}")
            intensities.append(f"{intensity:.2f}")
            sub1.append(f"{s1:.1f}")
            sub2.append(f"{s2:.1f}")
            sub3.append(f"{s3:.1f}")
            
        current_dt += delta

    df = pd.DataFrame({
        "Date": dates,
        "Time": times,
        "Global_active_power": active_powers,
        "Global_reactive_power": reactive_powers,
        "Voltage": voltages,
        "Global_intensity": intensities,
        "Sub_metering_1": sub1,
        "Sub_metering_2": sub2,
        "Sub_metering_3": sub3,
    })
    
    # Save with ';' separator matching UCI Household Electric Power Consumption format
    df.to_csv(output_path, sep=';', index=False)
    print(f"Generated sample dataset: {output_path} with {len(df)} rows.")
    return output_path

if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "sample", "household_power_sample.csv")
    generate_sample_dataset(out, days=120, freq_minutes=15)
