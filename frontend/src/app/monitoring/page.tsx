import { getAnomalies } from '@/lib/api';
import { AnomalyAlert } from '@/lib/types';

const severityStyles: Record<string, string> = {
  HIGH: 'border-red-500 bg-red-950/30',
  MEDIUM: 'border-yellow-500 bg-yellow-950/30',
  LOW: 'border-blue-500 bg-blue-950/30',
};

const severityBadge: Record<string, string> = {
  HIGH: 'bg-red-600 text-white',
  MEDIUM: 'bg-yellow-500 text-gray-900',
  LOW: 'bg-blue-600 text-white',
};

const typeLabel: Record<string, string> = {
  category_spike: 'Category Spike',
  geographic_hotspot: 'Geographic Hotspot',
  mine_type_risk: 'Mine Type Risk',
};

function AlertCard({ alert }: { alert: AnomalyAlert }) {
  return (
    <div className={`rounded-lg border p-5 shadow-lg ${severityStyles[alert.severity] ?? 'border-gray-700 bg-gray-800'}`}>
      <div className="flex items-start justify-between gap-4">
        <div className="flex-1">
          <div className="flex flex-wrap items-center gap-2 mb-2">
            <span className={`rounded px-2 py-0.5 text-xs font-bold uppercase tracking-wide ${severityBadge[alert.severity]}`}>
              {alert.severity}
            </span>
            <span className="rounded bg-gray-700 px-2 py-0.5 text-xs text-gray-300">
              {typeLabel[alert.type] ?? alert.type}
            </span>
          </div>
          <h3 className="text-lg font-semibold text-white">{alert.title}</h3>
          <p className="mt-1 text-sm text-gray-300">{alert.message}</p>
        </div>
      </div>
      <div className="mt-3 rounded bg-gray-900/50 border border-gray-700 p-3">
        <p className="text-xs font-semibold uppercase tracking-wide text-gray-400 mb-1">Recommended Action</p>
        <p className="text-sm text-cyan-300">{alert.recommendation}</p>
      </div>
    </div>
  );
}

export default async function MonitoringPage() {
  let alerts: AnomalyAlert[] = [];
  let error: string | null = null;

  try {
    alerts = await getAnomalies();
  } catch (e) {
    error = e instanceof Error ? e.message : 'An unknown error occurred';
  }

  const high = alerts.filter((a) => a.severity === 'HIGH');
  const medium = alerts.filter((a) => a.severity === 'MEDIUM');
  const low = alerts.filter((a) => a.severity === 'LOW');

  return (
    <div className="flex justify-center">
      <div className="w-[85%] my-6 space-y-6">
        <div>
          <h1 className="text-3xl font-bold text-white">Autonomous Monitoring</h1>
          <p className="mt-2 text-gray-400">
            Real-time anomaly detection and trend analysis from mining accident data. Alerts are generated
            automatically by statistical analysis of incident patterns.
          </p>
        </div>

        {error && (
          <div className="rounded-lg border border-red-700 bg-red-950/30 p-4 text-red-400">
            Error loading alerts: {error}
          </div>
        )}

        {alerts.length === 0 && !error && (
          <p className="text-gray-400">No anomalies detected.</p>
        )}

        {/* Summary bar */}
        {alerts.length > 0 && (
          <div className="grid grid-cols-3 gap-4">
            <div className="rounded-lg border border-red-700 bg-red-950/20 p-4 text-center">
              <p className="text-3xl font-bold text-red-400">{high.length}</p>
              <p className="text-sm text-gray-400 mt-1">High Severity</p>
            </div>
            <div className="rounded-lg border border-yellow-600 bg-yellow-950/20 p-4 text-center">
              <p className="text-3xl font-bold text-yellow-400">{medium.length}</p>
              <p className="text-sm text-gray-400 mt-1">Medium Severity</p>
            </div>
            <div className="rounded-lg border border-blue-700 bg-blue-950/20 p-4 text-center">
              <p className="text-3xl font-bold text-blue-400">{low.length}</p>
              <p className="text-sm text-gray-400 mt-1">Low Severity</p>
            </div>
          </div>
        )}

        {/* Alerts list */}
        <div className="space-y-4">
          {alerts.map((alert, i) => (
            <AlertCard key={i} alert={alert} />
          ))}
        </div>
      </div>
    </div>
  );
}
