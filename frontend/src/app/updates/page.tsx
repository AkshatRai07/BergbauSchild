import { getUpdates } from '@/lib/api';
import { Update } from '@/lib/types';

export default async function UpdatesPage() {
  let updates: Update[] = [];
  let error: string | null = null;

  try {
    updates = await getUpdates();
  } catch (e) {
    error = e instanceof Error ? e.message : 'An unknown error occurred';
  }

  return (
    <div className="space-y-6">
      <h1 className="text-3xl font-bold text-white mt-4 ml-35">DGMS & Safety Updates</h1>
      
      {error && <div className="text-center text-red-400">Error loading updates: {error}</div>}

      <div className="space-y-4 flex flex-col justify-center items-center">
        {updates.length === 0 && !error && (
          <p className="text-gray-400">No updates found.</p>
        )}
        {updates.map((update, index) => (
          <a
            key={index}
            target="_blank"
            rel="noopener noreferrer"
            className="block rounded-lg border border-gray-700 bg-gray-800 p-5 shadow-lg transition-all hover:border-cyan-500 hover:bg-gray-700 w-[80%]"
          >
            <p className="text-sm text-gray-400">
              {new Date(update.date).toLocaleDateString()} - {update.source}
            </p>
            <h3 className="mt-1 text-lg font-semibold text-cyan-400">{update.title}</h3>
            <h6 className="mt-1 font-semibold text-cyan-400">{update.summary}</h6>
          </a>
        ))}
      </div>
    </div>
  );
}