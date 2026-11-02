from pydantic import BaseModel
from typing import List, Dict, Any, Optional

class DescriptiveStats(BaseModel):
    count: int
    mean: float
    std: float
    min: float
    p25: float
    median: float
    p75: float
    max: float
    skewness: float
    iqr: float

class AnalyticsSummaryResponse(BaseModel):
    stats: DescriptiveStats
    peak_consumption: float
    peak_timestamp: str
    lowest_consumption: float
    lowest_timestamp: str
    total_energy_kwh: float
    avg_daily_kwh: float
    current_trend: str

class ConsumptionPoint(BaseModel):
    timestamp: str
    actual: float
    rolling_6h: Optional[float] = None
    rolling_24h: Optional[float] = None

class HourlyPattern(BaseModel):
    hour: int
    mean_kw: float
    std_kw: float
    min_kw: float
    max_kw: float

class DayOfWeekPattern(BaseModel):
    day_index: int
    day_name: str
    mean_kw: float
    std_kw: float
    peak_hour: int

class PatternsResponse(BaseModel):
    hourly: List[HourlyPattern]
    daily: List[DayOfWeekPattern]
    peak_hour_overall: int
    peak_day_overall: str
