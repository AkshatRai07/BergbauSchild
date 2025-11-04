'use client';

import { useState, FormEvent } from 'react';
import { generateReport } from '@/lib/api';

export default function ReportsPage() {
  const [year, setYear] = useState('');
  const [state, setState] = useState('');
  const [report, setReport] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    setIsLoading(true);
    setError(null);
    setReport(null);

    try {
      const res = await generateReport(year, state);
      setReport(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to generate report');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='flex justify-center my-6'>
      <div className='w-[85%]'>
        <div className="space-y-6">
          <h1 className="text-3xl font-bold text-white">Generate Safety Audit Report</h1>
          <p className="text-gray-300">
            Create an automated safety report. Leave fields blank to include all data.
          </p>

          <form onSubmit={handleSubmit} className="flex flex-wrap items-end gap-4">
            <div>
              <label htmlFor="year" className="block text-sm font-medium text-gray-300">
                Year (e.g., 2021)
              </label>
              <input
                type="number"
                id="year"
                value={year}
                onChange={(e) => setYear(e.target.value)}
                placeholder="Optional"
                className="mt-1 block rounded-lg border border-gray-600 bg-gray-800 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
            <div>
              <label htmlFor="state" className="block text-sm font-medium text-gray-300">
                State (e.g., Jharkhand)
              </label>
              <input
                type="text"
                id="state"
                value={state}
                onChange={(e) => setState(e.target.value)}
                placeholder="Optional"
                className="mt-1 block rounded-lg border border-gray-600 bg-gray-800 px-3 py-2 text-white focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
              />
            </div>
            <button
              type="submit"
              className="h-10 rounded-lg bg-cyan-600 px-6 py-2 font-semibold text-white transition-colors hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-gray-600"
              disabled={isLoading}
            >
              {isLoading ? 'Generating...' : 'Generate Report'}
            </button>
          </form>

          {error && <p className="text-red-400">Error: {error}</p>}

          {report && (
            <div>
              <h2 className="text-2xl font-semibold text-white">Generated Report</h2>
              <pre className="mt-4 w-full overflow-auto rounded-lg border border-gray-700 bg-gray-800 p-6 font-mono text-sm text-gray-200">
                {report}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}