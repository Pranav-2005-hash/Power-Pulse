import React, { useState, useEffect } from 'react';
import {
  TrendingUp, Play, Download, AlertCircle, Info,
  Cpu, CheckCircle2, ChevronRight, ShieldAlert
} from 'lucide-react';
import {
  ResponsiveContainer, ComposedChart, Line, Area,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts';
import { api } from '../services/api';
import { ForecastResponse, DatasetStatus } from '../types';
import { EmptyState, LoadingState } from '../components/EmptyState';

interface ForecastProps {
  datasetStatus: DatasetStatus | null;
}

export const Forecast: React.FC<ForecastProps> = ({ datasetStatus }) => {
  const [modelChoice, setModelChoice] = useState<'best' | 'lstm' | 'gru' | 'both'>('best');
  const [horizon, setHorizon] = useState<number>(24);
  const [loading, setLoading] = useState(false);
  const [forecastData, setForecastData] = useState<ForecastResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    if (datasetStatus?.models_trained) {
      handleRunForecast();
    }
  }, [datasetStatus?.models_trained]);

  const handleRunForecast = async () => {
    if (!datasetStatus?.models_trained) {
      setError("Please train deep learning models on the Models page before running inference.");
      return;
    }

    setLoading(true);
    setError(null);
    try {
      const res = await api.executeForecast(modelChoice, horizon);
      setForecastData(res);
    } catch (err: any) {
      setError(err.response?.data?.detail || "Failed to generate forecast.");
    } finally {
      setLoading(false);
    }
  };

  if (!datasetStatus?.models_trained) {
    return (
      <EmptyState
        icon={TrendingUp}
        title="Trained Deep Learning Models Required"
        description="Multi-step recursive forecasting requires weights from trained LSTM or GRU neural network models. Please run model training first."
        actionText="Go to Models Lab"
        onAction={() => window.location.href = '/models'}
      />
    );
  }

  // Combine historical points and forecast points into a single seamless chart timeline
  const prepareCombinedChartData = () => {
    if (!forecastData) return [];

    const timeline: any[] = [];

    // Historical context (past 48 hours)
    forecastData.historical_context.forEach(h => {
      timeline.push({
        timestamp: h.timestamp,
        actual: h.actual,
        lstm_pred: null,
        gru_pred: null,
        predicted: null,
        lower_bound: null,
        upper_bound: null,
        is_forecast: false
      });
    });

    // Bridge point: replicate the last actual value at the start of forecast to make lines connect smoothly
    if (timeline.length > 0 && forecastData.forecast_points.length > 0) {
      const lastHist = timeline[timeline.length - 1];
      timeline[timeline.length - 1] = {
        ...lastHist,
        predicted: lastHist.actual,
        lstm_pred: lastHist.actual,
        gru_pred: lastHist.actual,
        lower_bound: lastHist.actual,
        upper_bound: lastHist.actual,
      };
    }

    // Append forecast points
    forecastData.forecast_points.forEach(fp => {
      const existing = timeline.find(t => t.timestamp === fp.timestamp);
      if (existing) {
        if (fp.model === 'LSTM') existing.lstm_pred = fp.predicted;
        if (fp.model === 'GRU') existing.gru_pred = fp.predicted;
        existing.predicted = fp.predicted;
      } else {
        timeline.push({
          timestamp: fp.timestamp,
          actual: null,
          lstm_pred: fp.model === 'LSTM' ? fp.predicted : null,
          gru_pred: fp.model === 'GRU' ? fp.predicted : null,
          predicted: fp.predicted,
          lower_bound: fp.lower_bound,
          upper_bound: fp.upper_bound,
          is_forecast: true
        });
      }
    });

    return timeline;
  };

  const chartSeries = prepareCombinedChartData();

  // Peak projected load in horizon
  const projectedValues = forecastData?.forecast_points.map(p => p.predicted) || [];
  const peakProjected = projectedValues.length > 0 ? Math.max(...projectedValues) : null;
  const nextHourProjected = projectedValues.length > 0 ? projectedValues[0] : null;

  return (
    <div className="space-y-6">
      {/* Forecast Controls Bar */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <h1 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <TrendingUp className="w-5 h-5 text-emerald-500" />
            <span>Deep Learning Multi-Step Load Forecasting</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Recursive multi-step forecasting with dynamic uncertainty propagation
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Model Selection */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg text-xs">
            <span className="text-slate-500 px-2 font-medium">Model:</span>
            {(['best', 'lstm', 'gru', 'both'] as const).map(m => (
              <button
                key={m}
                onClick={() => setModelChoice(m)}
                className={`px-3 py-1.5 rounded font-semibold capitalize transition-all ${
                  modelChoice === m 
                    ? 'bg-white text-navy-950 shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {m === 'best' ? 'Best (Auto)' : m.toUpperCase()}
              </button>
            ))}
          </div>

          {/* Horizon Selection */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg text-xs">
            <span className="text-slate-500 px-2 font-medium">Horizon:</span>
            {[1, 6, 12, 24, 48].map(h => (
              <button
                key={h}
                onClick={() => setHorizon(h)}
                className={`px-2.5 py-1.5 rounded font-semibold transition-all ${
                  horizon === h 
                    ? 'bg-navy-900 text-white shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {h}h
              </button>
            ))}
          </div>

          {/* Run Button */}
          <button
            onClick={handleRunForecast}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-xs font-bold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg shadow-sm transition-all disabled:opacity-50"
          >
            {loading ? (
              <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
            ) : (
              <Play className="w-3.5 h-3.5 fill-current" />
            )}
            <span>Generate Forecast</span>
          </button>
        </div>
      </div>

      {error && (
        <div className="bg-rose-50 border border-rose-200 text-rose-800 p-4 rounded-xl text-sm flex items-center gap-2">
          <AlertCircle className="w-5 h-5 shrink-0 text-rose-600" />
          <span>{error}</span>
        </div>
      )}

      {/* Projection Metric Highlights */}
      {forecastData && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Forecast Horizon
            </span>
            <div className="mt-2 flex items-baseline gap-1.5">
              <span className="text-2xl font-extrabold text-navy-950 font-mono">
                +{forecastData.horizon_hours}
              </span>
              <span className="text-xs text-slate-500">hours ahead</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Recursive multi-step rollforward
            </p>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Next Hour (T+1) Projected
            </span>
            <div className="mt-2 flex items-baseline gap-1.5">
              <span className="text-2xl font-extrabold text-emerald-600 font-mono">
                {nextHourProjected}
              </span>
              <span className="text-xs text-slate-500">kW</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Estimated immediate demand
            </p>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Projected Peak Load
            </span>
            <div className="mt-2 flex items-baseline gap-1.5">
              <span className="text-2xl font-extrabold text-navy-950 font-mono">
                {peakProjected}
              </span>
              <span className="text-xs text-slate-500">kW</span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Maximum demand in forecast window
            </p>
          </div>

          <div className="bg-white p-5 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider block">
              Inference Architecture
            </span>
            <div className="mt-2 flex items-baseline gap-1.5">
              <span className="text-2xl font-extrabold text-cyan-600 font-mono">
                {forecastData.models_used.join(' + ')}
              </span>
            </div>
            <p className="text-[11px] text-slate-400 mt-2">
              Trained on 70% chronological split
            </p>
          </div>
        </div>
      )}

      {/* Main Interactive Forecast Chart */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
              <span>Historical Context vs Neural Forecast Timeline</span>
            </h2>
            <p className="text-xs text-slate-500">
              Solid line: Observed Historical readings &bull; Shaded/Dotted: Recursive Forecast with 95% Confidence Bounds
            </p>
          </div>

          <a
            href={api.getForecastCsvUrl()}
            download="powerpulse_forecast.csv"
            className="flex items-center gap-1.5 px-3 py-1.5 text-xs font-semibold text-slate-700 bg-slate-50 hover:bg-slate-100 border border-slate-200 rounded-lg shadow-sm transition-all"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Export Forecast CSV</span>
          </a>
        </div>

        {chartSeries.length > 0 ? (
          <div className="h-80 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart data={chartSeries} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="uncertaintyGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#06B6D4" stopOpacity={0.25} />
                    <stop offset="95%" stopColor="#06B6D4" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="timestamp"
                  tickLine={false}
                  axisLine={{ stroke: '#CBD5E1' }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickFormatter={(val) => {
                    const d = new Date(val);
                    return `${d.getMonth() + 1}/${d.getDate()} ${d.getHours()}h`;
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
                      const dataPoint = payload[0].payload;
                      return (
                        <div className="bg-navy-950 text-white p-3.5 rounded-lg shadow-xl border border-navy-800 text-xs">
                          <p className="text-slate-400 mb-1.5 font-medium">{label}</p>
                          {dataPoint.actual !== null && (
                            <p className="font-semibold text-emerald-400">
                              Observed Actual: {dataPoint.actual} kW
                            </p>
                          )}
                          {dataPoint.predicted !== null && (
                            <p className="font-bold text-cyan-300">
                              Forecast: {dataPoint.predicted} kW
                            </p>
                          )}
                          {dataPoint.lower_bound !== null && dataPoint.upper_bound !== null && (
                            <p className="text-[11px] text-slate-300 mt-1">
                              Confidence: [{dataPoint.lower_bound} - {dataPoint.upper_bound}] kW
                            </p>
                          )}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12, paddingTop: 10 }} />

                {/* Shaded confidence interval band */}
                <Area
                  type="monotone"
                  dataKey="upper_bound"
                  stroke="none"
                  fill="url(#uncertaintyGradient)"
                  name="Uncertainty Range"
                />

                {/* Historical Observed curve */}
                <Line
                  type="monotone"
                  dataKey="actual"
                  name="Observed Actual (kW)"
                  stroke="#10B981"
                  strokeWidth={2}
                  dot={false}
                />

                {/* Primary predicted line */}
                {modelChoice !== 'both' ? (
                  <Line
                    type="monotone"
                    dataKey="predicted"
                    name={`Forecast (${forecastData?.models_used[0] || 'DL'})`}
                    stroke="#06B6D4"
                    strokeWidth={2.5}
                    strokeDasharray="4 4"
                    dot={{ fill: '#06B6D4', r: 3 }}
                  />
                ) : (
                  <>
                    <Line
                      type="monotone"
                      dataKey="lstm_pred"
                      name="LSTM Forecast"
                      stroke="#10B981"
                      strokeWidth={2}
                      strokeDasharray="4 4"
                      dot={false}
                    />
                    <Line
                      type="monotone"
                      dataKey="gru_pred"
                      name="GRU Forecast"
                      stroke="#06B6D4"
                      strokeWidth={2}
                      strokeDasharray="3 3"
                      dot={false}
                    />
                  </>
                )}
              </ComposedChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center text-sm text-slate-500">
            Awaiting forecast generation...
          </div>
        )}
      </div>

      {/* Uncertainty Disclaimer Banner */}
      <div className="bg-amber-50 border border-amber-200/80 rounded-xl p-4 flex items-start gap-3 text-amber-900 text-xs leading-relaxed">
        <ShieldAlert className="w-5 h-5 text-amber-600 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold">Deep Learning Uncertainty Notice: </span>
          {forecastData?.disclaimer || (
            "Multi-step recursive predictions feed previous forecasted outputs back into the input sliding window. Uncertainty compounds with horizon length; actual future readings are never fabricated."
          )}
        </div>
      </div>

      {/* Detailed Forecast Breakdown Table */}
      {forecastData && (
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm overflow-hidden">
          <div className="flex items-center justify-between mb-4">
            <h2 className="text-base font-bold text-navy-950">Forecasted Hourly Values Breakdown</h2>
            <span className="text-xs text-slate-500 font-mono">
              {forecastData.forecast_points.length} Projected Steps
            </span>
          </div>

          <div className="overflow-x-auto max-h-72">
            <table className="w-full text-left text-xs border-collapse">
              <thead className="bg-slate-50 border-b border-slate-200 sticky top-0 z-10">
                <tr>
                  <th className="py-2.5 px-3 font-semibold text-slate-600">Step Ahead</th>
                  <th className="py-2.5 px-3 font-semibold text-slate-600">Timestamp</th>
                  <th className="py-2.5 px-3 font-semibold text-slate-600">Model</th>
                  <th className="py-2.5 px-3 font-semibold text-slate-600">Projected Load</th>
                  <th className="py-2.5 px-3 font-semibold text-slate-600">Lower Bound (95%)</th>
                  <th className="py-2.5 px-3 font-semibold text-slate-600">Upper Bound (95%)</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-100">
                {forecastData.forecast_points.map((pt, i) => (
                  <tr key={`${pt.timestamp}-${pt.model}`} className="hover:bg-slate-50">
                    <td className="py-2 px-3 font-mono font-bold text-slate-700">T+{pt.step_ahead}</td>
                    <td className="py-2 px-3 text-slate-900">{pt.timestamp}</td>
                    <td className="py-2 px-3">
                      <span className="px-2 py-0.5 rounded text-[10px] font-bold bg-slate-100 text-slate-700">
                        {pt.model}
                      </span>
                    </td>
                    <td className="py-2 px-3 font-mono font-bold text-emerald-600">{pt.predicted} kW</td>
                    <td className="py-2 px-3 font-mono text-slate-500">{pt.lower_bound} kW</td>
                    <td className="py-2 px-3 font-mono text-slate-500">{pt.upper_bound} kW</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};
