'use client';

import { useEffect, useState } from 'react';
import { getStatistics } from '@/lib/api';
import { Statistics } from '@/lib/types';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from 'recharts';

const COLORS = ['#06b6d4', '#0891b2', '#0e7490', '#155e75', '#164e63', '#083344', '#a5f3fc', '#67e8f9'];

function SectionTitle({ children }: { children: React.ReactNode }) {
  return <h2 className="text-xl font-semibold text-white mb-4">{children}</h2>;
}

export default function AnalyticsPage() {
  const [stats, setStats] = useState<Statistics | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStatistics()
      .then(setStats)
      .catch((e) => setError(e instanceof Error ? e.message : 'Failed to load statistics'));
  }, []);

  if (error) {
    return <div className="text-center text-red-400 mt-10">Error: {error}</div>;
  }

  if (!stats) {
    return <div className="text-center text-gray-400 mt-10">Loading analytics...</div>;
  }

  const categoryData = Object.entries(stats.by_category)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 10);

  const stateData = Object.entries(stats.by_state)
    .map(([name, value]) => ({ name, value }))
    .sort((a, b) => b.value - a.value)
    .slice(0, 8);

  const mineTypeData = Object.entries(stats.by_mine_type)
    .filter(([k]) => k && k !== 'nan')
    .map(([name, value]) => ({ name: name.charAt(0).toUpperCase() + name.slice(1), value }));

  const yearData = Object.entries(stats.by_year)
    .map(([year, deaths]) => ({ year: String(year), deaths: Number(deaths) }))
    .sort((a, b) => Number(a.year) - Number(b.year));

  return (
    <div className="flex justify-center">
      <div className="w-[90%] my-6 space-y-10">
        <h1 className="text-3xl font-bold text-white">Interactive Analytics</h1>

        {/* Fatalities by Year */}
        {yearData.length > 0 && (
          <div className="rounded-lg border border-gray-700 bg-gray-800 p-6">
            <SectionTitle>Fatalities by Year</SectionTitle>
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={yearData} margin={{ top: 5, right: 20, left: 0, bottom: 5 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis dataKey="year" stroke="#9ca3af" tick={{ fill: '#d1d5db' }} />
                <YAxis stroke="#9ca3af" tick={{ fill: '#d1d5db' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#f3f4f6' }}
                  itemStyle={{ color: '#06b6d4' }}
                />
                <Bar dataKey="deaths" fill="#06b6d4" radius={[4, 4, 0, 0]} name="Fatalities" />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )}

        {/* Accidents by Category */}
        <div className="rounded-lg border border-gray-700 bg-gray-800 p-6">
          <SectionTitle>Accidents by Category (Top 10)</SectionTitle>
          <ResponsiveContainer width="100%" height={320}>
            <BarChart
              data={categoryData}
              layout="vertical"
              margin={{ top: 5, right: 30, left: 220, bottom: 5 }}
            >
              <CartesianGrid strokeDasharray="3 3" stroke="#374151" horizontal={false} />
              <XAxis type="number" stroke="#9ca3af" tick={{ fill: '#d1d5db' }} />
              <YAxis
                type="category"
                dataKey="name"
                width={215}
                stroke="#9ca3af"
                tick={{ fill: '#d1d5db', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                labelStyle={{ color: '#f3f4f6' }}
                itemStyle={{ color: '#06b6d4' }}
              />
              <Bar dataKey="value" fill="#06b6d4" radius={[0, 4, 4, 0]} name="Incidents" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* State Fatalities + Mine Type Distribution */}
        <div className="grid grid-cols-1 gap-6 lg:grid-cols-2">
          <div className="rounded-lg border border-gray-700 bg-gray-800 p-6">
            <SectionTitle>Fatalities by State (Top 8)</SectionTitle>
            <ResponsiveContainer width="100%" height={280}>
              <BarChart data={stateData} margin={{ top: 5, right: 20, left: 0, bottom: 60 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#374151" />
                <XAxis
                  dataKey="name"
                  stroke="#9ca3af"
                  tick={{ fill: '#d1d5db', fontSize: 11 }}
                  angle={-35}
                  textAnchor="end"
                  interval={0}
                />
                <YAxis stroke="#9ca3af" tick={{ fill: '#d1d5db' }} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  labelStyle={{ color: '#f3f4f6' }}
                  itemStyle={{ color: '#0891b2' }}
                />
                <Bar dataKey="value" fill="#0891b2" radius={[4, 4, 0, 0]} name="Fatalities" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="rounded-lg border border-gray-700 bg-gray-800 p-6">
            <SectionTitle>Distribution by Mine Type</SectionTitle>
            <ResponsiveContainer width="100%" height={280}>
              <PieChart>
                <Pie
                  data={mineTypeData}
                  cx="50%"
                  cy="45%"
                  outerRadius={90}
                  dataKey="value"
                  nameKey="name"
                  label={({ name, percent }) => `${name} ${((percent ?? 0) * 100).toFixed(0)}%`}
                  labelLine={{ stroke: '#6b7280' }}
                >
                  {mineTypeData.map((_, index) => (
                    <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                  ))}
                </Pie>
                <Tooltip
                  contentStyle={{ backgroundColor: '#1f2937', border: '1px solid #374151', borderRadius: '8px' }}
                  itemStyle={{ color: '#d1d5db' }}
                />
                <Legend wrapperStyle={{ color: '#d1d5db', fontSize: 13 }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
}
