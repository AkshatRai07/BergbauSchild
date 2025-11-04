export interface Update {
  date: string;
  title: string;
  url: string;
  summary: string;
  source: string;
}

export interface Statistics {
  total_accidents: number;
  total_deaths: number;
  date_range: string;
  by_category: Record<string, number>;
  by_state: Record<string, number>;
  by_mine_type: Record<string, number>;
  by_year: Record<string, number>;
}

export interface ClassificationResult {
  description: string;
  predicted_code: string;
  code_description: string;
  error?: string;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant';
  content: string;
}