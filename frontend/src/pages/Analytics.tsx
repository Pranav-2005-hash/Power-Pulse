import React, { useState, useEffect } from 'react';
import {
  LineChart as LineChartIcon, BarChart3, Download, Filter,
  Calendar, Zap, Clock, Info, RefreshCw
} from 'lucide-react';
import {
  ResponsiveContainer, LineChart, Line, BarChart, Bar,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts';
import { api } from '../services/api';
import { AnalyticsSummary, ConsumptionPoint, PatternsResponse, DatasetStatus } from '../types';
import { EmptyState, LoadingState } from '../components/EmptyState';

interface AnalyticsProps {
  datasetStatus: DatasetStatus | null;
}

export const Analytics: React.FC<AnalyticsProps> = ({ datasetStatus }) => {
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [patterns, setPatterns] = useState<PatternsResponse | null>(null);
  const [series, setSeries] = useState<ConsumptionPoint[]>([]);
  
  // Date and filter controls
  const [preset, setPreset] = useState<'7d' | '30d' | 'all'>('7d');
  const [showRolling6, setShowRolling6] = useState(true);
  const [showRolling24, setShowRolling24] = useState(true);

  useEffect(() => {
    if (datasetStatus?.is_loaded) {
      loadAnalytics();
    }
  }, [datasetStatus?.is_loaded, preset]);

  const loadAnalytics = async () => {
    setLoading(true);
    try {
      const [sumData, patData] = await Promise.all([
        api.getAnalyticsSummary(),
        api.getConsumptionPatterns()
      ]);
      setSummary(sumData);
      setPatterns(patData);

      const limitMap = { '7d': 168, '30d': 720, 'all': 2000 };
      const seriesData = await api.getConsumptionSeries(undefined, undefined, limitMap[preset]);
      setSeries(seriesData);
    } catch (err) {
      console.error('Error loading analytics:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!datasetStatus?.is_loaded) {
    return (
      <EmptyState
        icon={LineChartIcon}
        title="Dataset Not Initialized"
        description="Load or upload a dataset to compute descriptive statistics, diurnal profiles, and time-series patterns."
        actionText="Go to Dataset Page"
        onAction={() => window.location.href = '/dataset'}
      />
    );
  }

  if (loading && !summary) {
    return <LoadingState message="Computing time-series statistics and diurnal distributions..." />;
  }

  // Calculate histogram bins for consumption distribution
  const computeHistogram = () => {
    if (!series.length) return [];
    const bins = [
      { range: '0.0 - 0.5 kW', count: 0 },
      { range: '0.5 - 1.0 kW', count: 0 },
      { range: '1.0 - 1.5 kW', count: 0 },
      { range: '1.5 - 2.5 kW', count: 0 },
      { range: '2.5 - 3.5 kW', count: 0 },
      { range: '3.5 - 5.0 kW', count: 0 },
      { range: '> 5.0 kW', count: 0 }
    ];

    series.forEach(pt => {
      const v = pt.actual;
      if (v < 0.5) bins[0].count++;
      else if (v < 1.0) bins[1].count++;
      else if (v < 1.5) bins[2].count++;
      else if (v < 2.5) bins[3].count++;
      else if (v < 3.5) bins[4].count++;
      else if (v <= 5.0) bins[5].count++;
      else bins[6].count++;
    });

    return bins;
  };

  const histogramData = computeHistogram();

  return (
    <div className="space-y-6">
      {/* Header and Export Controls */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h1 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <LineChartIcon className="w-5 h-5 text-emerald-500" />
            <span>Exploratory Energy Analytics</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Statistical profiles, diurnal load curves, and rolling moving averages
          </p>
        </div>

        <div className="flex items-center gap-3">
          <a
            href={api.getProcessedDataCsvUrl()}
            download="powerpulse_hourly_consumption.csv"
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-sm transition-all"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Export Cleaned CSV</span>
          </a>

          <button
            onClick={loadAnalytics}
            className="p-2 text-slate-600 hover:text-slate-900 bg-slate-100 hover:bg-slate-200 rounded-lg transition-all"
            title="Refresh statistics"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Descriptive Statistics Grid */}
      {summary && (
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-sm font-bold text-navy-950 uppercase tracking-wider text-slate-500">
              Descriptive Statistics &mdash; Global_active_power (kW)
            </h2>
            <span className="text-xs text-slate-400 font-mono">N = {summary.stats.count.toLocaleString()} hourly records</span>
          </div>

          <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[11px] text-slate-500 block">Mean Load</span>
              <span className="text-lg font-bold text-navy-950 font-mono">{summary.stats.mean}</span>
              <span className="text-[10px] text-slate-400 ml-1">kW</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[11px] text-slate-500 block">Std Deviation</span>
              <span className="text-lg font-bold text-navy-950 font-mono">{summary.stats.std}</span>
              <span className="text-[10px] text-slate-400 ml-1">kW</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[11px] text-slate-500 block">Median (p50)</span>
              <span className="text-lg font-bold text-navy-950 font-mono">{summary.stats.median}</span>
              <span className="text-[10px] text-slate-400 ml-1">kW</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[11px] text-slate-500 block">IQR (p75 - p25)</span>
              <span className="text-lg font-bold text-navy-950 font-mono">{summary.stats.iqr}</span>
              <span className="text-[10px] text-slate-400 ml-1">kW</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[11px] text-slate-500 block">Recorded Peak</span>
              <span className="text-lg font-bold text-rose-600 font-mono">{summary.stats.max}</span>
              <span className="text-[10px] text-slate-400 ml-1">kW</span>
            </div>
            <div className="p-3 bg-slate-50 rounded-lg border border-slate-100">
              <span className="text-[11px] text-slate-500 block">Skewness</span>
              <span className="text-lg font-bold text-navy-950 font-mono">{summary.stats.skewness}</span>
              <span className="text-[10px] text-slate-400 ml-1">pos</span>
            </div>
          </div>
        </div>
      )}

      {/* Main Time Series with Rolling Averages */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
              <span>Time-Series Profile with Moving Averages</span>
            </h2>
            <p className="text-xs text-slate-500">
              Compare instantaneous hourly demand against 6-hour and 24-hour smoothed moving windows
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-4">
            {/* Toggles for rolling averages */}
            <div className="flex items-center gap-3 text-xs">
              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showRolling6}
                  onChange={(e) => setShowRolling6(e.target.checked)}
                  className="rounded text-cyan-600 focus:ring-cyan-500"
                />
                <span className="text-slate-700 font-medium">6h Moving Mean</span>
              </label>

              <label className="flex items-center gap-1.5 cursor-pointer">
                <input
                  type="checkbox"
                  checked={showRolling24}
                  onChange={(e) => setShowRolling24(e.target.checked)}
                  className="rounded text-amber-600 focus:ring-amber-500"
                />
                <span className="text-slate-700 font-medium">24h Moving Mean</span>
              </label>
            </div>

            {/* Presets */}
            <div className="flex items-center gap-1 bg-slate-100 p-1 rounded-lg">
              {(['7d', '30d', 'all'] as const).map((p) => (
                <button
                  key={p}
                  onClick={() => setPreset(p)}
                  className={`px-3 py-1 rounded text-xs font-semibold transition-all ${
                    preset === p ? 'bg-white text-navy-950 shadow-sm' : 'text-slate-600 hover:text-slate-900'
                  }`}
                >
                  {p === '7d' ? '7 Days' : p === '30d' ? '30 Days' : 'Full Window'}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="h-80 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={series} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
              <XAxis
                dataKey="timestamp"
                tickLine={false}
                axisLine={{ stroke: '#CBD5E1' }}
                tick={{ fill: '#64748B', fontSize: 11 }}
                tickFormatter={(val) => {
                  const d = new Date(val);
                  return preset === '7d' ? `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}h` : `${d.getMonth() + 1}/${d.getDate()}`;
                }}
              />
              <YAxis
                tickLine={false}
                axisLine={{ stroke: '#CBD5E1' }}
                tick={{ fill: '#64748B', fontSize: 11 }}
                unit=" kW"
              />
              <Tooltip
                content={({ active, payload, label }) => {
                  if (active && payload && payload.length) {
                    return (
                      <div className="bg-navy-950 text-white p-3 rounded-lg shadow-lg border border-navy-800 text-xs">
                        <p className="text-slate-400 mb-1">{label}</p>
                        {payload.map((entry: any) => (
                          <p key={entry.name} style={{ color: entry.color }} className="font-semibold">
                            {entry.name}: {entry.value} kW
                          </p>
                        ))}
                      </div>
                    );
                  }
                  return null;
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />
              <Line
                type="monotone"
                dataKey="actual"
                name="Hourly Consumption"
                stroke="#10B981"
                strokeWidth={1.5}
                dot={false}
              />
              {showRolling6 && (
                <Line
                  type="monotone"
                  dataKey="rolling_6h"
                  name="6-Hour Rolling Mean"
                  stroke="#06B6D4"
                  strokeWidth={2}
                  dot={false}
                />
              )}
              {showRolling24 && (
                <Line
                  type="monotone"
                  dataKey="rolling_24h"
                  name="24-Hour Rolling Mean"
                  stroke="#F59E0B"
                  strokeWidth={2}
                  dot={false}
                />
              )}
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Two-Column Grid: Hourly Diurnal Profile & Day-of-Week Profile */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Hourly Diurnal Curve */}
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
                <Clock className="w-4 h-4 text-cyan-500" />
                <span>Diurnal Hourly Load Profile</span>
              </h2>
              <p className="text-xs text-slate-500">
                Average consumption by hour of the day (00:00 &ndash; 23:00)
              </p>
            </div>
            {patterns && (
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-cyan-50 text-cyan-700 border border-cyan-200">
                Peak: {patterns.peak_hour_overall}:00
              </span>
            )}
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={patterns?.hourly || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="hour"
                  tickLine={false}
                  axisLine={{ stroke: '#CBD5E1' }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickFormatter={(h) => `${h}h`}
                />
                <YAxis
                  tickLine={false}
                  axisLine={{ stroke: '#CBD5E1' }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  unit=" kW"
                />
                <Tooltip
                  formatter={(val: any) => [`${val} kW`, 'Mean Load']}
                  labelFormatter={(h) => `Hour ${h}:00 - ${h}:59`}
                />
                <Bar
                  dataKey="mean_kw"
                  fill="#06B6D4"
                  radius={[4, 4, 0, 0]}
                  name="Mean Consumption (kW)"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Day of Week Profile */}
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
                <Calendar className="w-4 h-4 text-emerald-500" />
                <span>Day-of-Week Load Profile</span>
              </h2>
              <p className="text-xs text-slate-500">
                Average power demand by weekday vs weekend
              </p>
            </div>
            {patterns && (
              <span className="px-2.5 py-1 rounded-full text-xs font-semibold bg-emerald-50 text-emerald-700 border border-emerald-200">
                Peak Day: {patterns.peak_day_overall}
              </span>
            )}
          </div>

          <div className="h-64 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={patterns?.daily || []} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="day_name"
                  tickLine={false}
                  axisLine={{ stroke: '#CBD5E1' }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickFormatter={(d) => d.slice(0, 3)}
                />
                <YAxis
                  tickLine={false}
                  axisLine={{ stroke: '#CBD5E1' }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  unit=" kW"
                />
                <Tooltip
                  formatter={(val: any) => [`${val} kW`, 'Mean Load']}
                />
                <Bar
                  dataKey="mean_kw"
                  fill="#10B981"
                  radius={[4, 4, 0, 0]}
                  name="Mean Load (kW)"
                />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      {/* Consumption Distribution Histogram */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <h2 className="text-base font-bold text-navy-950 flex items-center gap-2 mb-1">
          <BarChart3 className="w-4 h-4 text-slate-700" />
          <span>Consumption Frequency Distribution</span>
        </h2>
        <p className="text-xs text-slate-500 mb-6">
          Frequency count of hourly kilowatt intervals across the active observation window
        </p>

        <div className="h-60 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={histogramData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
              <XAxis
                dataKey="range"
                tickLine={false}
                axisLine={{ stroke: '#CBD5E1' }}
                tick={{ fill: '#64748B', fontSize: 11 }}
              />
              <YAxis
                tickLine={false}
                axisLine={{ stroke: '#CBD5E1' }}
                tick={{ fill: '#64748B', fontSize: 11 }}
              />
              <Tooltip formatter={(val: any) => [`${val} hours`, 'Duration']} />
              <Bar dataKey="count" fill="#3B82F6" radius={[4, 4, 0, 0]} name="Occurrences" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};
