import React, { useState, useEffect } from 'react';
import {
  AlertTriangle, Filter, Download, CheckCircle2, Clock,
  ArrowUpDown, Search, RefreshCw, AlertCircle, Info, ShieldAlert
} from 'lucide-react';
import {
  ResponsiveContainer, ComposedChart, Line, Scatter,
  XAxis, YAxis, Tooltip, CartesianGrid, Legend
} from 'recharts';
import { api } from '../services/api';
import { AnomaliesResponse, AnomalyEvent, DatasetStatus } from '../types';
import { EmptyState, LoadingState } from '../components/EmptyState';

interface AnomaliesProps {
  datasetStatus: DatasetStatus | null;
}

export const Anomalies: React.FC<AnomaliesProps> = ({ datasetStatus }) => {
  const [percentile, setPercentile] = useState<number>(95.0);
  const [statusFilter, setStatusFilter] = useState<'all' | 'unreviewed' | 'reviewed'>('all');
  const [severityFilter, setSeverityFilter] = useState<'all' | 'Critical' | 'High' | 'Moderate'>('all');
  const [searchTerm, setSearchTerm] = useState('');
  const [sortField, setSortField] = useState<'timestamp' | 'residual' | 'actual'>('timestamp');
  const [sortAsc, setSortAsc] = useState(false);

  const [loading, setLoading] = useState(true);
  const [recalculating, setRecalculating] = useState(false);
  const [data, setData] = useState<AnomaliesResponse | null>(null);

  useEffect(() => {
    if (datasetStatus?.models_trained) {
      loadAnomalies();
    }
  }, [datasetStatus?.models_trained, percentile, statusFilter]);

  const loadAnomalies = async () => {
    setLoading(true);
    try {
      const res = await api.getAnomalies(percentile, statusFilter);
      setData(res);
    } catch (err) {
      console.error('Error fetching anomalies:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleRecalculate = async (newPercentile: number) => {
    setPercentile(newPercentile);
    setRecalculating(true);
    try {
      const res = await api.recalculateAnomalies(newPercentile);
      setData(res);
    } catch (err) {
      console.error('Error recalculating anomalies:', err);
    } finally {
      setRecalculating(false);
    }
  };

  const handleToggleReview = async (anomalyId: string, currentStatus: boolean) => {
    try {
      await api.reviewAnomaly(anomalyId, !currentStatus);
      // Update local state optimistically
      setData(prev => {
        if (!prev) return null;
        const updated = prev.anomalies.map(a => 
          a.id === anomalyId ? { ...a, reviewed: !currentStatus } : a
        );
        return {
          ...prev,
          anomalies: updated,
          reviewed_count: currentStatus ? prev.reviewed_count - 1 : prev.reviewed_count + 1,
          unreviewed_count: currentStatus ? prev.unreviewed_count + 1 : prev.unreviewed_count - 1
        };
      });
    } catch (err) {
      console.error('Error updating review:', err);
    }
  };

  if (!datasetStatus?.models_trained) {
    return (
      <EmptyState
        icon={AlertTriangle}
        title="Model Training Required for Anomaly Detection"
        description="Residual anomaly detection evaluates differences between observed actual power consumption and trained neural network predictions. Please train models first."
        actionText="Train Models Now"
        onAction={() => window.location.href = '/models'}
      />
    );
  }

  // Filter and sort anomalies list
  let filteredAnomalies = (data?.anomalies || []).filter(a => {
    if (severityFilter !== 'all' && a.severity !== severityFilter) return false;
    if (searchTerm && !a.timestamp.toLowerCase().includes(searchTerm.toLowerCase())) return false;
    return true;
  });

  filteredAnomalies.sort((a, b) => {
    let comparison = 0;
    if (sortField === 'timestamp') {
      comparison = a.timestamp.localeCompare(b.timestamp);
    } else if (sortField === 'residual') {
      comparison = a.residual - b.residual;
    } else if (sortField === 'actual') {
      comparison = a.actual - b.actual;
    }
    return sortAsc ? comparison : -comparison;
  });

  // Chart data preparation
  const chartSeries = (data?.series_sample || []).map(pt => ({
    ...pt,
    anomaly_actual: pt.is_anomaly ? pt.actual : null,
  }));

  return (
    <div className="space-y-6">
      {/* Top Banner and Sensitivity Controls */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm flex flex-col lg:flex-row lg:items-center justify-between gap-6">
        <div>
          <h1 className="text-xl font-bold text-navy-950 flex items-center gap-2">
            <AlertTriangle className="w-5 h-5 text-amber-500" />
            <span>Residual-Based Anomaly Detection Lab</span>
          </h1>
          <p className="text-xs text-slate-500 mt-1">
            Statistical thresholding derived strictly from positive validation residuals (residual = actual &minus; predicted)
          </p>
        </div>

        <div className="flex flex-wrap items-center gap-3">
          {/* Percentile Threshold Selector */}
          <div className="flex items-center gap-1.5 bg-slate-100 p-1 rounded-lg text-xs">
            <span className="text-slate-500 px-2 font-medium">Threshold:</span>
            {[
              { val: 90.0, label: '90th % (High)' },
              { val: 95.0, label: '95th % (Std)' },
              { val: 99.0, label: '99th % (Spikes)' }
            ].map(p => (
              <button
                key={p.val}
                onClick={() => handleRecalculate(p.val)}
                className={`px-3 py-1.5 rounded font-semibold transition-all ${
                  percentile === p.val 
                    ? 'bg-navy-900 text-white shadow-sm' 
                    : 'text-slate-600 hover:text-slate-900'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>

          <a
            href={api.getAnomaliesCsvUrl()}
            download="powerpulse_anomalies.csv"
            className="flex items-center gap-2 px-3.5 py-2 text-xs font-semibold text-slate-700 bg-white hover:bg-slate-50 border border-slate-200 rounded-lg shadow-sm transition-all"
          >
            <Download className="w-3.5 h-3.5 text-slate-500" />
            <span>Export CSV</span>
          </a>
        </div>
      </div>

      {/* Anomaly Stat Cards */}
      {data && (
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-5 gap-3">
          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
              Total Anomalies
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-extrabold text-navy-950 font-mono">
                {data.total_anomalies}
              </span>
              <span className="text-[11px] text-slate-400">events</span>
            </div>
            <p className="text-[10px] text-slate-400 mt-1 font-mono">
              Cutoff &gt; {data.threshold_value_kw} kW
            </p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
              Critical Surges (p99)
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-extrabold text-rose-600 font-mono">
                {data.severity_counts['Critical'] || 0}
              </span>
            </div>
            <p className="text-[10px] text-rose-500 mt-1">Severe deviation</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
              High Severity (p95)
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-extrabold text-amber-600 font-mono">
                {data.severity_counts['High'] || 0}
              </span>
            </div>
            <p className="text-[10px] text-amber-600 mt-1">Elevated residual</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
              Moderate (p90)
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-extrabold text-slate-700 font-mono">
                {data.severity_counts['Moderate'] || 0}
              </span>
            </div>
            <p className="text-[10px] text-slate-400 mt-1">Minor unexpected load</p>
          </div>

          <div className="bg-white p-4 rounded-xl border border-slate-200/80 shadow-sm">
            <span className="text-[11px] font-semibold text-slate-500 uppercase tracking-wider block">
              Reviewed Status
            </span>
            <div className="mt-1 flex items-baseline gap-1">
              <span className="text-2xl font-extrabold text-emerald-600 font-mono">
                {data.reviewed_count}
              </span>
              <span className="text-xs text-slate-400">/ {data.total_anomalies}</span>
            </div>
            <p className="text-[10px] text-slate-400 mt-1">
              {data.unreviewed_count} pending review
            </p>
          </div>
        </div>
      )}

      {/* Actual vs Predicted Time-Series Chart with Anomaly Markers */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div>
            <h2 className="text-base font-bold text-navy-950 flex items-center gap-2">
              <span>Actual vs Predicted Load &mdash; Flagged Anomalies Timeline</span>
            </h2>
            <p className="text-xs text-slate-500">
              Red markers designate positive residuals exceeding {percentile}th validation percentile threshold ({data?.threshold_value_kw} kW)
            </p>
          </div>
        </div>

        <div className="h-72 w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={chartSeries} margin={{ top: 10, right: 10, left: -20, bottom: 0 }}>
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
                    const d = payload[0].payload;
                    return (
                      <div className="bg-navy-950 text-white p-3 rounded-lg shadow-xl border border-navy-800 text-xs">
                        <p className="text-slate-400 mb-1">{label}</p>
                        <p className="text-emerald-400 font-semibold">Actual: {d.actual} kW</p>
                        <p className="text-slate-300">Predicted: {d.predicted} kW</p>
                        <p className={`font-mono font-bold mt-1 ${d.is_anomaly ? 'text-rose-400' : 'text-slate-400'}`}>
                          Residual: +{d.residual} kW {d.is_anomaly ? `(${d.severity} Anomaly)` : ''}
                        </p>
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
                name="Observed Actual (kW)"
                stroke="#10B981"
                strokeWidth={1.8}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="predicted"
                name="Model Prediction (kW)"
                stroke="#64748B"
                strokeWidth={1.5}
                strokeDasharray="3 3"
                dot={false}
              />
              <Scatter
                dataKey="anomaly_actual"
                name="Flagged Anomaly"
                fill="#EF4444"
                shape="circle"
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Educational Explainer Banner */}
      <div className="bg-slate-900 text-slate-300 rounded-xl p-4 border border-navy-700 flex items-start gap-3 text-xs leading-relaxed">
        <ShieldAlert className="w-5 h-5 text-cyan-400 shrink-0 mt-0.5" />
        <div>
          <span className="font-bold text-white">Academic Deep Learning Notice: </span>
          {data?.disclaimer || (
            "Flagged anomalies indicate time periods where actual consumption significantly exceeded the neural network's expected output based on validation residuals. These statistical indications highlight unusual usage surges, not necessarily equipment failure or hardware defects."
          )}
        </div>
      </div>

      {/* Filterable, Sortable Anomaly Table */}
      <div className="bg-white rounded-xl p-6 border border-slate-200/80 shadow-sm">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <div className="flex items-center gap-3">
            <h2 className="text-base font-bold text-navy-950">Detected Events Register</h2>
            <span className="text-xs text-slate-500 font-mono">
              Showing {filteredAnomalies.length} events
            </span>
          </div>

          <div className="flex flex-wrap items-center gap-3">
            {/* Search */}
            <div className="relative">
              <Search className="w-3.5 h-3.5 text-slate-400 absolute left-2.5 top-1/2 -translate-y-1/2" />
              <input
                type="text"
                placeholder="Search timestamp..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="pl-8 pr-3 py-1.5 text-xs border border-slate-200 rounded-lg focus:outline-none focus:ring-1 focus:ring-emerald-500 w-44"
              />
            </div>

            {/* Severity Filter */}
            <select
              value={severityFilter}
              onChange={(e) => setSeverityFilter(e.target.value as any)}
              className="text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 bg-white text-slate-700 focus:outline-none"
            >
              <option value="all">All Severities</option>
              <option value="Critical">Critical Only</option>
              <option value="High">High Only</option>
              <option value="Moderate">Moderate Only</option>
            </select>

            {/* Status Filter */}
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value as any)}
              className="text-xs border border-slate-200 rounded-lg px-2.5 py-1.5 bg-white text-slate-700 focus:outline-none"
            >
              <option value="all">All Statuses</option>
              <option value="unreviewed">Unreviewed Only</option>
              <option value="reviewed">Reviewed Only</option>
            </select>
          </div>
        </div>

        <div className="overflow-x-auto max-h-96">
          <table className="w-full text-left text-xs border-collapse">
            <thead className="bg-slate-50 border-b border-slate-200 sticky top-0 z-10">
              <tr>
                <th
                  onClick={() => { setSortField('timestamp'); setSortAsc(!sortAsc); }}
                  className="py-2.5 px-3 font-semibold text-slate-600 cursor-pointer hover:bg-slate-100"
                >
                  <div className="flex items-center gap-1">
                    <span>Timestamp</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th
                  onClick={() => { setSortField('actual'); setSortAsc(!sortAsc); }}
                  className="py-2.5 px-3 font-semibold text-slate-600 cursor-pointer hover:bg-slate-100"
                >
                  <div className="flex items-center gap-1">
                    <span>Actual (kW)</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th className="py-2.5 px-3 font-semibold text-slate-600">Predicted (kW)</th>
                <th
                  onClick={() => { setSortField('residual'); setSortAsc(!sortAsc); }}
                  className="py-2.5 px-3 font-semibold text-slate-600 cursor-pointer hover:bg-slate-100"
                >
                  <div className="flex items-center gap-1">
                    <span>Residual (+kW)</span>
                    <ArrowUpDown className="w-3 h-3 text-slate-400" />
                  </div>
                </th>
                <th className="py-2.5 px-3 font-semibold text-slate-600">Threshold</th>
                <th className="py-2.5 px-3 font-semibold text-slate-600">Severity</th>
                <th className="py-2.5 px-3 font-semibold text-slate-600 text-right">Review Action</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-slate-100">
              {filteredAnomalies.length > 0 ? (
                filteredAnomalies.map((anom) => (
                  <tr key={anom.id} className="hover:bg-slate-50">
                    <td className="py-2.5 px-3 font-mono font-medium text-slate-800">{anom.timestamp}</td>
                    <td className="py-2.5 px-3 font-mono font-bold text-navy-950">{anom.actual} kW</td>
                    <td className="py-2.5 px-3 font-mono text-slate-500">{anom.predicted} kW</td>
                    <td className="py-2.5 px-3 font-mono font-bold text-rose-600">+{anom.residual} kW</td>
                    <td className="py-2.5 px-3 font-mono text-slate-500">{anom.threshold} kW</td>
                    <td className="py-2.5 px-3">
                      <span className={`px-2 py-0.5 rounded-full text-[10px] font-bold border ${
                        anom.severity === 'Critical'
                          ? 'bg-rose-50 text-rose-700 border-rose-200'
                          : anom.severity === 'High'
                          ? 'bg-amber-50 text-amber-700 border-amber-200'
                          : 'bg-slate-100 text-slate-700 border-slate-200'
                      }`}>
                        {anom.severity}
                      </span>
                    </td>
                    <td className="py-2.5 px-3 text-right">
                      <button
                        onClick={() => handleToggleReview(anom.id, anom.reviewed)}
                        className={`px-3 py-1 rounded-md text-[11px] font-semibold transition-all ${
                          anom.reviewed
                            ? 'bg-emerald-50 text-emerald-700 border border-emerald-200 hover:bg-emerald-100'
                            : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
                        }`}
                      >
                        {anom.reviewed ? '✓ Reviewed' : 'Mark Reviewed'}
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={7} className="py-8 text-center text-slate-400">
                    No anomalies match the current filter criteria.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
};
