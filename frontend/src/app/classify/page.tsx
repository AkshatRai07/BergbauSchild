'use client';

import { useState, FormEvent } from 'react';
import { classifyAccident } from '@/lib/api';
import { ClassificationResult } from '@/lib/types';

export default function ClassifyPage() {
  const [description, setDescription] = useState('');
  const [result, setResult] = useState<ClassificationResult | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!description.trim()) return;

    setIsLoading(true);
    setError(null);
    setResult(null);

    try {
      const res = await classifyAccident(description);
      setResult(res);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to classify');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className='flex justify-center my-6'>
      <div className='w-[85%]'>
        <div className="space-y-6">
          <h1 className="text-3xl font-bold text-white">New Incident Classifier</h1>
          <p className="text-gray-300">
            Enter a description of a new mining accident, and the AI will predict its
            official DGMS accident code based on historical data.
          </p>

          <form onSubmit={handleSubmit} className="space-y-4">
            <textarea
              rows={8}
              value={description}
              onChange={(e) => setDescription(e.target.value)}
              placeholder="e.g., 'A dumper truck overturned while moving material from the quarry face, resulting in one fatality...'"
              className="w-full rounded-lg border border-gray-600 bg-gray-800 p-4 text-white placeholder-gray-400 focus:border-cyan-500 focus:outline-none focus:ring-1 focus:ring-cyan-500"
            />
            <button
              type="submit"
              className="rounded-lg bg-cyan-600 px-6 py-2 font-semibold text-white transition-colors hover:bg-cyan-500 disabled:cursor-not-allowed disabled:bg-gray-600"
              disabled={isLoading || !description.trim()}
            >
              {isLoading ? 'Classifying...' : 'Classify Accident'}
            </button>
          </form>

          {error && <p className="text-red-400">Error: {error}</p>}

          {result && (
            <div className="space-y-4 rounded-lg border border-gray-700 bg-gray-800 p-6">
              <h2 className="text-2xl font-semibold text-white">Classification Result</h2>
              <div>
                <p className="text-sm font-medium text-gray-400">Predicted Code</p>
                <p className="text-xl font-bold text-cyan-400">{result.predicted_code}</p>
              </div>
              <div>
                <p className="text-sm font-medium text-gray-400">Code Description</p>
                <p className="text-lg text-gray-200">{result.code_description}</p>
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}