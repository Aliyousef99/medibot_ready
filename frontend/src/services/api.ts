import axios from 'axios';

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

const api = axios.create({
    baseURL: 'http://localhost:8000', // Adjust if your backend is on a different port
    withCredentials: true,
});

api.interceptors.request.use(config => {
    const authData = localStorage.getItem('auth');
    if (authData) {
        const { token } = JSON.parse(authData);
        if (token) {
            config.headers.Authorization = `Bearer ${token}`;
        }
    }
    return config;
});

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

export const analyzeSymptoms = async (symptoms: string): Promise<SymptomAnalysisResult> => {
    const response = await api.post('/api/symptoms/analyze', { text: symptoms });
    return response.data;
};

export const extractText = async (file: File): Promise<{ text: string }> => {
    const formData = new FormData();
    formData.append('file', file);
    const response = await api.post('/api/extract_text', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
    });
    return response.data;
};

export const postChatMessage = async (message: string): Promise<{ response: string }> => {
    const response = await api.post('/api/chat', { message });
    return response.data;
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
