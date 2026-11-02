import React from 'react';
import { LucideIcon } from 'lucide-react';

interface MetricCardProps {
  title: string;
  value: string | number;
  unit?: string;
  icon: LucideIcon;
  badge?: string;
  badgeType?: 'green' | 'cyan' | 'amber' | 'rose' | 'neutral';
  subtitle?: string;
  glow?: boolean;
}

export const MetricCard: React.FC<MetricCardProps> = ({
  title,
  value,
  unit,
  icon: Icon,
  badge,
  badgeType = 'neutral',
  subtitle,
  glow = false
}) => {
  const badgeStyles = {
    green: 'bg-emerald-50 text-emerald-700 border-emerald-200',
    cyan: 'bg-cyan-50 text-cyan-700 border-cyan-200',
    amber: 'bg-amber-50 text-amber-700 border-amber-200',
    rose: 'bg-rose-50 text-rose-700 border-rose-200',
    neutral: 'bg-slate-100 text-slate-700 border-slate-200'
  }[badgeType];

  return (
    <div className={`
      bg-white rounded-xl border border-slate-200/80 p-5 shadow-sm hover:shadow-md transition-all
      ${glow ? 'ring-1 ring-emerald-500/30' : ''}
    `}>
      <div className="flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-500 uppercase tracking-wider">
          {title}
        </span>
        <div className="w-8 h-8 rounded-lg bg-slate-50 border border-slate-200/80 flex items-center justify-center text-slate-700">
          <Icon className="w-4 h-4" />
        </div>
      </div>

      <div className="mt-3 flex items-baseline gap-1.5">
        <span className="text-2xl sm:text-3xl font-extrabold text-navy-950 tracking-tight">
          {value}
        </span>
        {unit && (
          <span className="text-xs font-medium text-slate-500">{unit}</span>
        )}
      </div>

      {(badge || subtitle) && (
        <div className="mt-3 pt-3 border-t border-slate-100 flex items-center justify-between text-xs">
          {badge && (
            <span className={`px-2 py-0.5 rounded-full border text-[11px] font-semibold ${badgeStyles}`}>
              {badge}
            </span>
          )}
          {subtitle && (
            <span className="text-slate-500 truncate text-[11px] ml-auto">{subtitle}</span>
          )}
        </div>
      )}
    </div>
  );
};
