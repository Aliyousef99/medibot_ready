import axios from 'axios';
import { useAuthStore } from "../state/authStore";
import { useToastStore } from "../state/toastStore";

export interface ChatResponseCombined {
  conversation_id?: string;
  request_id: string;
  summary?: string;
  user_view?: {
    summary?: string;
    abnormal?: Array<{ name?: string; value?: number | string; unit?: string; status?: string; ref_min?: number; ref_max?: number; reference?: any }>;
    normal?: Array<{ name?: string; value?: number | string; unit?: string; status?: string; ref_min?: number; ref_max?: number; reference?: any }>;
    recommendation?: string;
    confidence?: number;
    explanation?: string;
  };
  symptom_analysis: {
    symptoms?: string[];
    possible_tests?: string[];
    confidence?: number;
    event_id?: string | null;
  };
  local_recommendations: {
    priority: string;
    actions: string[];
    follow_up: string;
    rationale: string;
  };
  ai_explanation: string;
  ai_explanation_source?: 'model' | 'fallback' | 'skipped';
  timed_out?: boolean;
  disclaimer: string;
  pipeline?: any;
  missing_fields?: string[];
  triage?: {
    level: 'ok' | 'watch' | 'urgent' | string;
    reasons?: string[];
  };
}

export interface ConversationSummary {
  id: string;
  title?: string | null;
  created_at: string;
  updated_at: string;
}

export interface ConversationMessageRow {
  id: string;
  role: string;
  content: string;
  created_at: string;
}

export interface SymptomAnalysisResult {
  summary: string;
  symptoms: { text: string; label: string; score: number; negated: boolean }[];
  urgency: 'low' | 'medium' | 'high' | 'urgent';
  engine: string;
}

// Lab and recommendations types used across the app
export interface LabReport {
  id: string;
  user_id: string;
  title?: string | null;
  raw_text: string;
  structured_json?: any;
  summary?: string | null;
  created_at: string;
  updated_at: string;
}

export interface RecommendationSet {
  id?: number;
  risk_tier: 'low' | 'moderate' | 'high' | string;
  actions: string[];
  text?: string;
  llm_used?: boolean;
  created_at?: string;
}

// Resolve API base in priority order:
// 1) window.__CHAT_API_BASE__ (runtime override)
// 2) VITE_API_BASE env (set in Vercel / .env)
// 3) same-origin (relative /api calls when backend is hosted together)
const runtimeBase =
  typeof window !== 'undefined' && (window as any).__CHAT_API_BASE__
    ? String((window as any).__CHAT_API_BASE__)
    : '';
const envBase = (import.meta as any)?.env?.VITE_API_BASE ? String((import.meta as any).env.VITE_API_BASE) : '';
const fallbackBase =
  typeof window !== 'undefined' && window.location ? window.location.origin : '';
const resolvedBase = (runtimeBase || envBase || fallbackBase || '').replace(/\/+$/, '');

const api = axios.create({
  baseURL: resolvedBase,
  withCredentials: true,
});

// Expose the underlying client for tests
export const apiClient = api;
api.interceptors.request.use(config => {
  const authData = localStorage.getItem('auth');
  if (authData) {
    const { token } = JSON.parse(authData);
    if (token) {
      config.headers = config.headers || {};
      (config.headers as any).Authorization = `Bearer ${token}`;
      if (process.env.NODE_ENV !== "production") {
        // eslint-disable-next-line no-console
        console.debug("API request with bearer", { url: config.url });
      }
    } else if (process.env.NODE_ENV !== "production") {
      // eslint-disable-next-line no-console
      console.debug("API request missing token", { url: config.url });
    }
  } else if (process.env.NODE_ENV !== "production") {
    // eslint-disable-next-line no-console
    console.debug("API request without auth data", { url: config.url });
  }
  return config;
});

let isRefreshing = false;
let refreshQueue: Array<() => void> = [];

const flushQueue = () => {
  refreshQueue.forEach(fn => fn());
  refreshQueue = [];
};

const refreshAccessToken = async (): Promise<string> => {
  const stored = localStorage.getItem("auth");
  if (!stored) throw new Error("session expired");
  const parsed = JSON.parse(stored);
  const refresh = parsed.refresh_token;
  if (!refresh) throw new Error("session expired");
  const res = await api.post("/api/auth/refresh", { refresh_token: refresh });
  const newAccess = res.data?.access_token;
  if (!newAccess) throw new Error("invalid refresh");
  const authStore = useAuthStore.getState();
  const nextUser = { ...(authStore.user || {}), token: newAccess, refresh_token: refresh };
  authStore.setUser(nextUser);
  return newAccess;
};

// Test-only hook to exercise refresh logic without firing interceptors
export const __testRefreshAccessToken = refreshAccessToken;

export function handleApiError(error: any) {
  const status = error?.response?.status;
  const toast = useToastStore.getState();
  const msg =
    status === 400
      ? "We could not process that request. Please check your input."
      : status === 401
      ? "Unauthorized. Please log in again."
      : status === 403
      ? "You do not have access to perform this action."
      : status === 429
      ? "You have hit the rate limit. Please wait a moment and try again."
      : status === 500
      ? "Server error. Please try again shortly."
      : error?.message || "Network error. Please try again.";
  toast.add({ type: status === 500 ? "error" : "info", message: msg });
}

api.interceptors.response.use(
  (res) => res,
  async (error) => {
    const status = error?.response?.status;
    const original = error.config;
    if (status === 401 && !original?._retry) {
      original._retry = true;
      const authStore = useAuthStore.getState();
      const toast = useToastStore.getState();

      const doRefresh = async () => {
        try {
          isRefreshing = true;
          authStore.startRefresh();
          await refreshAccessToken();
          return true;
        } finally {
          isRefreshing = false;
          authStore.finishRefresh();
          flushQueue();
        }
      };

      if (!isRefreshing) {
        await doRefresh().catch(() => {
          authStore.logout();
          toast.add({ type: "error", message: "Your session expired. Please log in again." });
        });
      } else {
        await new Promise<void>((resolve) => refreshQueue.push(resolve));
      }
      // After refresh/log out, do not retry; surface 401
    }
    handleApiError(error);
    return Promise.reject(error);
  }
);

export const register = async (email: string, password: string): Promise<any> => {
    const response = await api.post('/api/auth/register', { email, password });
    return response.data;
};

export const login = async (email: string, password: string): Promise<{ access_token: string }> => {
    const response = await api.post('/api/auth/login', { email, password });
    return response.data;
};

export const getProfile = async (): Promise<any> => {
    const response = await api.get('/api/profile/');
    return response.data;
};

export const updateProfile = async (profileData: any): Promise<any> => {
    const response = await api.put('/api/profile/', profileData);
    return response.data;
};

export const getConsent = async (): Promise<{ consent_given: boolean; consent_at?: string | null }> => {
  const res = await api.get("/api/privacy/consent");
  return res.data;
};

export const setConsent = async (consent_given: boolean): Promise<{ consent_given: boolean; consent_at?: string | null }> => {
  const res = await api.post("/api/privacy/consent", { consent_given });
  return res.data;
};

export const deleteUserData = async (): Promise<void> => {
  await api.delete("/api/privacy/delete_data");
};

// Note: Symptom analysis is now fused into /api/chat results; no separate call needed.

export const extractText = async (file: File): Promise<{ text: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    // Let the browser set the correct multipart boundary; avoid overriding Content-Type
    const response = await api.post('/api/extract_text', formData);
    return response.data;
};

export const uploadLabAndSave = async (file: File): Promise<LabReport> => {
  const formData = new FormData();
  formData.append('file', file);
  const res = await api.post('/api/history/labs/upload', formData);
  return res.data as LabReport;
};

export const postChatMessage = async (message: string, conversationId?: string): Promise<ChatResponseCombined> => {
  const payload: any = { message };
  if (conversationId) payload.conversation_id = conversationId;
  const response = await api.post('/api/chat', payload);
  if (process.env.NODE_ENV !== 'production') {
    // DEV: Log RAW payload before any transformation and explicitly surface required keys
    // eslint-disable-next-line no-console
    console.debug('CHAT /api/chat RAW', response.data);
    const {
      request_id,
      summary,
      symptom_analysis,
      local_recommendations,
      ai_explanation,
      disclaimer,
      pipeline,
    } = (response?.data ?? {}) as Partial<ChatResponseCombined> as any;
    // eslint-disable-next-line no-console
    console.debug('CHAT /api/chat KEYS CHECK', {
      request_id,
      summary,
      symptom_analysis,
      local_recommendations,
      ai_explanation,
      disclaimer,
      pipeline,
    });
  }
  return response.data as ChatResponseCombined;
};

export const postChatImage = async (file: File, conversationId?: string): Promise<ChatResponseCombined> => {
  const formData = new FormData();
  formData.append('file', file);
  if (conversationId) formData.append('conversation_id', conversationId);
  const response = await api.post('/api/chat/image', formData);
  return response.data as ChatResponseCombined;
};

// ---------- History (Labs) ----------

export const getHistory = async (): Promise<LabReport[]> => {
  const res = await api.get('/api/history/labs');
  return res.data as LabReport[];
};

export const getHistoryDetail = async (labId: string): Promise<LabReport> => {
  const res = await api.get(`/api/history/labs/${labId}`);
  return res.data as LabReport;
};

export const deleteHistory = async (labId: string): Promise<void> => {
  await api.delete(`/api/history/labs/${labId}`);
};

// ---------- Recommendations ----------

export const getRecommendationsForLab = async (labId: string): Promise<RecommendationSet> => {
  const res = await api.post('/api/recommendations/generate', { lab_id: labId });
  return res.data as RecommendationSet;
};

// ---------- Conversations ----------

export const listConversations = async (): Promise<ConversationSummary[]> => {
  const res = await api.get('/api/chat/conversations');
  return res.data as ConversationSummary[];
};

export const createConversation = async (title?: string): Promise<ConversationSummary> => {
  const res = await api.post('/api/chat/conversations', { title });
  return res.data as ConversationSummary;
};

export const deleteConversationApi = async (conversationId: string): Promise<void> => {
  await api.delete(`/api/chat/conversations/${conversationId}`);
};

export const getConversationMessages = async (conversationId: string, limit = 50): Promise<ConversationMessageRow[]> => {
  const res = await api.get(`/api/chat/conversations/${conversationId}/messages`, { params: { limit } });
  return res.data as ConversationMessageRow[];
};
