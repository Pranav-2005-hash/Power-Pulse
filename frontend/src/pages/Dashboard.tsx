import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Zap, TrendingUp, AlertTriangle, Cpu, Database,
  ArrowUpRight, Clock, Award, CheckCircle2, ChevronRight
} from 'lucide-react';
import {
  ResponsiveContainer, AreaChart, Area, XAxis, YAxis, Tooltip, CartesianGrid
} from 'recharts';
import { api } from '../services/api';
import { DatasetStatus, AnalyticsSummary, ModelComparison, AnomaliesResponse } from '../types';
import { MetricCard } from '../components/MetricCard';
import { EmptyState, LoadingState } from '../components/EmptyState';

interface DashboardProps {
  datasetStatus: DatasetStatus | null;
  onRefresh: () => void;
}

export const Dashboard: React.FC<DashboardProps> = ({ datasetStatus, onRefresh }) => {
  const navigate = useNavigate();
  const [loading, setLoading] = useState(true);
  const [summary, setSummary] = useState<AnalyticsSummary | null>(null);
  const [modelComp, setModelComp] = useState<ModelComparison | null>(null);
  const [anomalies, setAnomalies] = useState<AnomaliesResponse | null>(null);
  const [chartData, setChartData] = useState<any[]>([]);
  const [chartRange, setChartRange] = useState<'48h' | '7d' | '30d'>('48h');

  useEffect(() => {
    loadDashboardData();
  }, [datasetStatus?.is_loaded, datasetStatus?.models_trained, chartRange]);

  const loadDashboardData = async () => {
    setLoading(true);
    try {
      if (datasetStatus?.is_loaded) {
        // Fetch summary
        const sumData = await api.getAnalyticsSummary();
        setSummary(sumData);

        // Fetch time-series based on range
        const limitMap = { '48h': 48, '7d': 168, '30d': 720 };
        const seriesData = await api.getConsumptionSeries(undefined, undefined, limitMap[chartRange]);
        setChartData(seriesData);
      }

      if (datasetStatus?.models_trained) {
        const compData = await api.getModelComparison();
        setModelComp(compData);

        const anomData = await api.getAnomalies(95.0, 'all');
        setAnomalies(anomData);
      }
    } catch (err) {
      console.error('Error loading dashboard data:', err);
    } finally {
      setLoading(false);
    }
  };

  if (!datasetStatus?.is_loaded) {
    return (
      <div className="space-y-6">
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-bold text-navy-950">Welcome to PowerPulse</h1>
            <p className="text-sm text-slate-500 mt-1">
              AI-Powered Household Energy Forecasting & Residual-Based Anomaly Detection
            </p>
          </div>
          <button
            onClick={() => api.loadSampleDataset().then(() => onRefresh())}
            className="flex items-center gap-2 px-4 py-2 text-sm font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-lg shadow-sm transition-all"
          >
            <Zap className="w-4 h-4" />
            <span>Load Sample Dataset</span>
          </button>
        </div>

        <EmptyState
          icon={Database}
          title="No Active Dataset Initialized"
          description="PowerPulse requires household electricity consumption time-series data to train LSTM/GRU neural networks and detect anomalies. Load the pre-configured UCI sample or upload custom CSV."
          actionText="Initialize Sample Data"
          onAction={() => api.loadSampleDataset().then(() => onRefresh())}
        />
      </div>
    );
  }

  const bestModelName = modelComp?.best_model;
  const bestModelMetrics = bestModelName && modelComp?.models ? modelComp.models[bestModelName] : null;
  const lstmMetrics = modelComp?.models?.['LSTM'];
  const gruMetrics = modelComp?.models?.['GRU'];

  return (
    <div className="space-y-6">
      {/* Top Banner / Status Overview */}
      <div className="bg-gradient-to-r from-navy-950 via-navy-900 to-navy-850 rounded-2xl p-6 text-white border border-navy-800/80 shadow-lg relative overflow-hidden">
        <div className="absolute right-0 top-0 bottom-0 w-96 bg-gradient-to-l from-emerald-500/10 via-cyan-500/5 to-transparent pointer-events-none" />
        
        <div className="flex flex-col lg:flex-row lg:items-center justify-between gap-6 relative z-10">
          <div>
            <div className="inline-flex items-center gap-2 px-3 py-1 rounded-full bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-xs font-semibold mb-3">
              <Zap className="w-3.5 h-3.5 fill-current" />
              Real Deep Learning Pipeline Active
            </div>
            <h1 className="text-2xl sm:text-3xl font-extrabold tracking-tight text-white">
              PowerPulse Analytics Center
            </h1>
            <p className="text-sm text-slate-300 mt-1.5 max-w-2xl leading-relaxed">
              Monitoring <span className="text-emerald-400 font-semibold">{datasetStatus.total_hourly_rows.toLocaleString()}</span> hourly readings from {datasetStatus.start_date?.slice(0, 10)} to {datasetStatus.end_date?.slice(0, 10)}. Target: <span className="font-mono text-cyan-300">Global_active_power (kW)</span>.
            </p>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {!datasetStatus.models_trained ? (
              <button
                onClick={() => navigate('/models')}
                className="flex items-center gap-2 px-4 py-2.5 text-sm font-semibold text-navy-950 bg-gradient-to-r from-emerald-400 to-cyan-400 hover:from-emerald-300 hover:to-cyan-300 rounded-xl shadow-glow-green transition-all"
              >
                <Cpu className="w-4 h-4" />
                <span>Train LSTM & GRU Models</span>
              </button>
            ) : (
              <button
                onClick={() => navigate('/forecast')}
                className="flex items-center gap-2 px-4 py-2.5 text-sm font-semibold text-white bg-emerald-600 hover:bg-emerald-500 rounded-xl shadow-md transition-all"
              >
                <TrendingUp className="w-4 h-4" />
                <span>Execute Deep Forecast</span>
              </button>
            )}

            <button
              onClick={() => navigate('/analytics')}
              className="flex items-center gap-2 px-4 py-2.5 text-sm font-medium text-slate-200 bg-navy-800/80 hover:bg-navy-700/80 border border-navy-700 rounded-xl transition-all"
            >
              <span>Explore Analytics</span>
              <ChevronRight className="w-4 h-4 text-slate-400" />
            </button>
          </div>
        </div>
      </div>

      {/* Primary KPI Metric Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        <MetricCard
          title="Average Consumption"
          value={summary ? summary.stats.mean : '--'}
          unit="kW"
          icon={Zap}
          badge={summary?.current_trend || 'Baseline'}
          badgeType={summary?.current_trend.includes('Elevated') ? 'amber' : 'green'}
          subtitle={`Daily avg: ${summary ? summary.avg_daily_kwh : '--'} kWh`}
        />

        <MetricCard
          title="Peak Recorded Load"
          value={summary ? summary.peak_consumption : '--'}
          unit="kW"
          icon={ArrowUpRight}
          badge="Max Demand"
          badgeType="rose"
          subtitle={summary?.peak_timestamp ? `At ${summary.peak_timestamp.slice(11, 16)}` : undefined}
        />

        <MetricCard
          title="Residual Anomalies"
          value={anomalies ? anomalies.total_anomalies : (datasetStatus.models_trained ? 0 : '--')}
          unit="events"
          icon={AlertTriangle}
          badge={anomalies?.unreviewed_count ? `${anomalies.unreviewed_count} Unreviewed` : 'Reviewed'}
          badgeType={anomalies?.unreviewed_count ? 'amber' : 'green'}
          subtitle={anomalies ? `95th percentile (> ${anomalies.threshold_value_kw} kW)` : 'Requires trained models'}
        />

        <MetricCard
          title="Selected Best Model"
          value={bestModelName || (datasetStatus.models_trained ? 'Evaluating' : 'Untrained')}
          icon={Award}
          badge={bestModelMetrics ? `Val MAE: ${bestModelMetrics.val_mae} kW` : 'No Models'}
          badgeType="cyan"
          subtitle={bestModelMetrics ? `Test RMSE: ${bestModelMetrics.test_rmse} kW` : 'Click to train'}
          glow={!!bestModelName}
        />
      </div>

      {/* Model Performance Comparison Summary Bar */}
      {datasetStatus.models_trained && (
        <div className="bg-white rounded-xl p-4 border border-slate-200/80 shadow-sm flex flex-col md:flex-row items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-cyan-50 border border-cyan-200 flex items-center justify-center text-cyan-700">
              <Cpu className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-sm font-bold text-navy-950">Deep Learning Recurrent Architecture Metrics</h2>
              <p className="text-xs text-slate-500">Evaluated on held-out 15% test sequences in true kW units</p>
            </div>
          </div>

          <div className="flex items-center gap-6 text-xs">
            <div className="border-l border-slate-200 pl-4">
              <span className="text-slate-500 block">LSTM Model:</span>
              <span className="font-mono font-semibold text-slate-800">
                MAE: {lstmMetrics ? `${lstmMetrics.test_mae} kW` : '--'} &bull; RMSE: {lstmMetrics ? `${lstmMetrics.test_rmse} kW` : '--'}
              </span>
            </div>

            <div className="border-l border-slate-200 pl-4">
              <span className="text-slate-500 block">GRU Model:</span>
              <span className="font-mono font-semibold text-slate-800">
                MAE: {gruMetrics ? `${gruMetrics.test_mae} kW` : '--'} &bull; RMSE: {gruMetrics ? `${gruMetrics.test_rmse} kW` : '--'}
              </span>
            </div>

            <button
              onClick={() => navigate('/models')}
              className="text-xs font-semibold text-emerald-600 hover:text-emerald-700 underline"
            >
              Compare details &rarr;
            </button>
          </div>
        </div>
      )}

      {/* Main Historical Load Time-Series Chart */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-6">
          <div>
            <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
              <span>Historical Consumption Time-Series</span>
              <span className="text-xs font-normal text-slate-500 font-mono">(kW)</span>
            </h2>
            <p className="text-xs text-slate-500 mt-0.5">
              Hourly mean aggregated from {datasetStatus.sampling_rate}
            </p>
          </div>

          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg">
            {(['48h', '7d', '30d'] as const).map((range) => (
              <button
                key={range}
                onClick={() => setChartRange(range)}
                className={`px-3 py-1 rounded-md text-xs font-semibold transition-all ${
                  chartRange === range
                    ? 'bg-white text-navy-950 shadow-sm'
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {range === '48h' ? 'Last 48 Hours' : range === '7d' ? 'Last 7 Days' : 'Last 30 Days'}
              </button>
            ))}
          </div>
        </div>

        {chartData.length > 0 ? (
          <div className="h-72 w-full">
            <ResponsiveContainer width="100%" height="100%">
              <AreaChart data={chartData} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
                <defs>
                  <linearGradient id="consumptionGradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#10B981" stopOpacity={0.4} />
                    <stop offset="95%" stopColor="#10B981" stopOpacity={0.0} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#E2E8F0" />
                <XAxis
                  dataKey="timestamp"
                  tickLine={false}
                  axisLine={{ stroke: '#CBD5E1' }}
                  tick={{ fill: '#64748B', fontSize: 11 }}
                  tickFormatter={(val) => {
                    const dt = new Date(val);
                    return chartRange === '48h' 
                      ? `${dt.getHours()}:00` 
                      : `${dt.getMonth() + 1}/${dt.getDate()}`;
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
                          <p className="font-bold text-emerald-400">
                            Actual: {payload[0].value} kW
                          </p>
                          {payload[1] && payload[1].value && (
                            <p className="text-cyan-300">
                              Rolling 6h: {payload[1].value} kW
                            </p>
                          )}
                        </div>
                      );
                    }
                    return null;
                  }}
                />
                <Area
                  type="monotone"
                  dataKey="actual"
                  stroke="#10B981"
                  strokeWidth={2}
                  fillOpacity={1}
                  fill="url(#consumptionGradient)"
                  name="Consumption (kW)"
                />
                {chartRange !== '48h' && (
                  <Area
                    type="monotone"
                    dataKey="rolling_6h"
                    stroke="#06B6D4"
                    strokeWidth={1.5}
                    strokeDasharray="4 4"
                    fill="none"
                    name="Rolling 6h Mean"
                  />
                )}
              </AreaChart>
            </ResponsiveContainer>
          </div>
        ) : (
          <div className="h-64 flex items-center justify-center text-sm text-slate-500">
            No consumption points found in specified window.
          </div>
        )}
      </div>

      {/* Bottom Grid: Recent Anomaly Feed & Quick Actions */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Recent Anomalies Feed */}
        <div className="lg:col-span-2 bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
          <div className="flex items-center justify-between mb-4">
            <div>
              <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
                <AlertTriangle className="w-4 h-4 text-amber-500" />
                <span>Recent Anomaly Alerts</span>
              </h2>
              <p className="text-xs text-slate-500">
                Flagged when observed demand exceeded positive validation residual threshold
              </p>
            </div>
            <button
              onClick={() => navigate('/anomalies')}
              className="text-xs font-semibold text-emerald-600 hover:text-emerald-700 flex items-center gap-1"
            >
              <span>View All ({anomalies?.total_anomalies || 0})</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
          </div>

          {anomalies && anomalies.anomalies.length > 0 ? (
            <div className="divide-y divide-slate-100">
              {anomalies.anomalies.slice(0, 5).map((anom) => (
                <div key={anom.id} className="py-3 flex items-center justify-between gap-4">
                  <div className="flex items-center gap-3">
                    <span className={`px-2 py-0.5 rounded-full text-[11px] font-semibold border ${
                      anom.severity === 'Critical' 
                        ? 'bg-rose-50 text-rose-700 border-rose-200' 
                        : anom.severity === 'High'
                        ? 'bg-amber-50 text-amber-700 border-amber-200'
                        : 'bg-slate-100 text-slate-700 border-slate-200'
                    }`}>
                      {anom.severity}
                    </span>
                    <div>
                      <p className="text-xs font-bold text-navy-950">
                        {anom.timestamp}
                      </p>
                      <p className="text-[11px] text-slate-500">
                        Actual: <span className="font-semibold text-slate-700">{anom.actual} kW</span> (Expected: {anom.predicted} kW)
                      </p>
                    </div>
                  </div>

                  <div className="text-right flex items-center gap-3">
                    <span className="text-xs font-mono font-bold text-rose-600">
                      +{anom.residual} kW
                    </span>
                    <button
                      onClick={async () => {
                        await api.reviewAnomaly(anom.id, !anom.reviewed);
                        loadDashboardData();
                      }}
                      className={`px-2.5 py-1 rounded text-[11px] font-semibold transition-all ${
                        anom.reviewed 
                          ? 'bg-slate-100 text-slate-600 hover:bg-slate-200' 
                          : 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100'
                      }`}
                    >
                      {anom.reviewed ? 'Reviewed' : 'Mark Done'}
                    </button>
                  </div>
                </div>
              ))}
            </div>
          ) : (
            <div className="py-8 text-center text-xs text-slate-500">
              {datasetStatus.models_trained 
                ? 'No anomaly events detected exceeding current sensitivity threshold.' 
                : 'Train models to enable residual anomaly detection.'}
            </div>
          )}
        </div>

        {/* Quick Launch & Documentation Link */}
        <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col justify-between">
          <div>
            <h2 className="text-base font-bold text-navy-950 flex items-center gap-2 mb-2">
              <Zap className="w-4 h-4 text-emerald-500" />
              <span>Project Workflow</span>
            </h2>
            <p className="text-xs text-slate-500 leading-relaxed mb-4">
              PowerPulse executes an end-to-end college deep learning pipeline following strict chronological validation:
            </p>

            <ol className="space-y-3 text-xs text-slate-600">
              <li className="flex items-start gap-2">
                <span className="w-5 h-5 rounded-full bg-emerald-100 text-emerald-700 font-bold flex items-center justify-center shrink-0 text-[11px]">1</span>
                <span><strong>Hourly Resampling:</strong> UCI 15-min readings aggregated into clean hourly mean kW.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-5 h-5 rounded-full bg-cyan-100 text-cyan-700 font-bold flex items-center justify-center shrink-0 text-[11px]">2</span>
                <span><strong>No Data Leakage:</strong> 70/15/15 chronological split; MinMaxScaler fitted strictly on training split.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-5 h-5 rounded-full bg-navy-100 text-navy-800 font-bold flex items-center justify-center shrink-0 text-[11px]">3</span>
                <span><strong>Dual Architecture:</strong> LSTM(32) and GRU(32) trained with Adam and EarlyStopping.</span>
              </li>
              <li className="flex items-start gap-2">
                <span className="w-5 h-5 rounded-full bg-amber-100 text-amber-800 font-bold flex items-center justify-center shrink-0 text-[11px]">4</span>
                <span><strong>Residual Anomaly Detection:</strong> Statistical threshold calibrated from validation positive residuals.</span>
              </li>
            </ol>
          </div>

          <div className="pt-6 border-t border-slate-100 mt-6 flex justify-between items-center">
            <button
              onClick={() => navigate('/about')}
              className="text-xs font-semibold text-navy-900 hover:text-emerald-600 flex items-center gap-1"
            >
              <span>Read Architecture Specs</span>
              <ChevronRight className="w-3.5 h-3.5" />
            </button>
            <span className="text-[11px] text-slate-400 font-mono">TensorFlow Keras</span>
          </div>
        </div>
      </div>
    </div>
  );
};
