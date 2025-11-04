import { getStatistics } from '@/lib/api';
import { Statistics } from '@/lib/types';

function StatCard({ title, value }: { title: string; value: string | number }) {
  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-5 shadow-lg">
      <p className="text-sm font-medium text-gray-400">{title}</p>
      <p className="mt-1 text-3xl font-semibold text-white">{value}</p>
    </div>
  );
}

function ListCard({ title, data }: { title: string; data: Record<string, number> }) {
  const sortedData = Object.entries(data).sort(([, a], [, b]) => b - a);

  return (
    <div className="rounded-lg border border-gray-700 bg-gray-800 p-5 shadow-lg">
      <h3 className="text-lg font-semibold text-white">{title}</h3>
      <ul className="mt-3 space-y-2 overflow-y-auto">
        {sortedData.map(([key, value]) => (
          <li key={key} className="flex justify-between text-sm text-gray-300">
            <span>{key}</span>
            <span className="font-medium text-gray-100">{value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export default async function DashboardPage() {
  let stats: Statistics | null = null;
  let error: string | null = null;

  try {
    stats = await getStatistics();
  } catch (e) {
    error = e instanceof Error ? e.message : 'An unknown error occurred';
  }

  if (error) {
    return <div className="text-center text-red-400">Error loading dashboard: {error}</div>;
  }

  if (!stats) {
    return <div className="text-center text-gray-400">Loading statistics...</div>;
  }

  return (
    <div className='flex justify-center'>
      <div className='w-[85%] my-6'>
        <div className="space-y-6">
          <h1 className="text-3xl font-bold text-white">Mine Safety Dashboard</h1>
          
          {/* Key Metrics */}
          <div className="grid grid-cols-1 gap-5 sm:grid-cols-3">
            <StatCard title="Total Accidents" value={stats.total_accidents} />
            <StatCard title="Total Fatalities" value={stats.total_deaths} />
            <StatCard title="Date Range" value={stats.date_range.split(' ')[0]} />
          </div>

          {/* Detailed Breakdowns */}
          <div className="grid grid-cols-1 gap-5 md:grid-cols-2 lg:grid-cols-3">
            <ListCard title="Fatalities by State" data={stats.by_state} />
            <ListCard title="Fatalities by Category" data={stats.by_category} />
            <ListCard title="Fatalities by Mine Type" data={stats.by_mine_type} />
          </div>
        </div>
      </div>
    </div>
  );
}