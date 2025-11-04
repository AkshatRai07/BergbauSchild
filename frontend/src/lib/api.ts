import { Statistics, Update, ClassificationResult } from './types';

// IMPORTANT: Set this to your backend URL
const API_BASE_URL = process.env.NEXT_PUBLIC_BACKEND_URL || 'http://localhost:8000';

export async function getStatistics(): Promise<Statistics> {
  const res = await fetch(`${API_BASE_URL}/statistics`);
  if (!res.ok) {
    throw new Error('Failed to fetch statistics');
  }
  return res.json();
}

export async function getUpdates(): Promise<Update[]> {
  const res = await fetch(`${API_BASE_URL}/updates`);
  if (!res.ok) {
    throw new Error('Failed to fetch updates');
  }
  return res.json();
}

export async function classifyAccident(description: string): Promise<ClassificationResult> {
  const res = await fetch(`${API_BASE_URL}/classify_new_accident`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ description }),
  });
  if (!res.ok) {
    throw new Error('Failed to classify accident');
  }
  return res.json();
}

export async function generateReport(year?: string, state?: string): Promise<string> {
  const params = new URLSearchParams();
  if (year) params.append('year', year);
  if (state) params.append('state', state);

  const res = await fetch(`${API_BASE_URL}/generate_report?${params.toString()}`);
  if (!res.ok) {
    throw new Error('Failed to generate report');
  }
  return res.text();
}

// Note: The chat stream fetch logic will be handled directly in the
// chat component due to its streaming nature.