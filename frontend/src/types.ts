export type User = {
  id: number;
  email: string;
  token: string;
  profile?: any;
};

export type Role = "user" | "assistant";

export type Message = {
  id: string;
  role: Role;
  content: string;
  ts: Date;
  requestId?: string;
  symptomAnalysis?: { symptoms?: string[]; possible_tests?: string[]; confidence?: number; event_id?: string | null };
  localRecommendations?: { priority: string; actions: string[]; follow_up: string; rationale: string };
  disclaimer?: string;
  rawKeys?: string[];
  aiSource?: "model" | "fallback" | "skipped";
  rawResponse?: any;
};

export type Conversation = {
  id: string;
  title: string;
  messages: Message[];
};

export type SymptomAnalysisResult = {
  urgency: string;
  summary: string;
  symptoms: { text: string; label: string; score: number; negated?: boolean }[];
};
