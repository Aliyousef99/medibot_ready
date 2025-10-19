import React, { useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { Send, Plus, MessageSquare, Trash2, Moon, Sun, Bot, User, Menu, Loader2, ShieldAlert, Copy } from "lucide-react";
import UserMenu from "./UserMenu";
import {
  register as apiRegister,
  login as apiLogin,
  getProfile as apiGetProfile,
  updateProfile as apiUpdateProfile,
  postChatMessage as apiChat,
  extractText as apiExtract,
  analyzeSymptoms as apiAnalyzeSymptoms,
} from "../services/api"; // <-- relative path is key
import type { ChatResponseCombined } from "../services/api";


// ---------- Types ----------
type User = {
  id: number;
  email: string;
  token: string;
  profile?: any;
};

type Role = "user" | "assistant";

type Message = {
  id: string;
  role: Role;
  content: string;
  ts: Date;
  // DEV-only fields rendered under assistant message
  requestId?: string;
  symptomAnalysis?: { symptoms?: string[]; possible_tests?: string[]; confidence?: number; event_id?: string | null };
  localRecommendations?: { priority: string; actions: string[]; follow_up: string; rationale: string };
  disclaimer?: string;
  // DEV: keys from the raw /api/chat object for quick shape verification
  rawKeys?: string[];
  aiSource?: 'model' | 'fallback' | 'skipped';
};

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
};

function nextActiveIdAfterDelete(
  prev: Conversation[],
  activeId: string | null,
  deletedId: string
): string | null {
  const next = prev.filter((c) => c.id !== deletedId);
  if (!next.length) return null;
  if (activeId === deletedId) return next[0]?.id ?? null;
  return activeId && next.some((c) => c.id === activeId) ? activeId : next[0]?.id ?? null;
}

function safeActiveId(convos: Conversation[], candidate: string | null): string | null {
  if (candidate && convos.some((c) => c.id === candidate)) return candidate;
  return convos[0]?.id ?? null;
}

function seedMessages(): Message[] {
  return [
    {
      id: "m1",
      role: "assistant",
      ts: new Date(Date.now() - 1000 * 60 * 3),
      content: "Hey! I'm your demo assistant. Paste lab text or upload a file.",
    },
  ];
}

function formatTime(d: Date) {
  try {
    return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(d);
  } catch {
    return d.toLocaleTimeString();
  }
}

function prettyRef(ref?: any) {
  if (!ref) return "";
  const u = ref.unit ? ` ${ref.unit}` : "";
  switch (ref.kind) {
    case "lte": return `ref â‰¤${ref.v}${u}`;
    case "lt":  return `ref <${ref.v}${u}`;
    case "gte": return `ref â‰¥${ref.v}${u}`;
    case "gt":  return `ref >${ref.v}${u}`;
    case "between": return `ref ${ref.lo}â€“${ref.hi}${u}`;
    default: return "";
  }
}

function AuthScreen({ onAuth }: { onAuth: (user: User) => void }) {
  const [mode, setMode] = useState<"login" | "register">("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);

  async function submit() {
    setLoading(true);
    setError("");
    try {
      if (mode === "register") {
        await apiRegister(email, password);
      }
      const loginData = await apiLogin(email, password);
      const token = loginData.access_token;
      
      // Store minimal auth data first to enable subsequent authenticated requests
      const userScaffold = { id: 0, email, token }; // Assuming id is not immediately available
      localStorage.setItem("auth", JSON.stringify(userScaffold));

      const profile = await apiGetProfile().catch(() => null);
      
      const user = { ...userScaffold, profile };
      localStorage.setItem("auth", JSON.stringify(user));
      
      onAuth(user);
    } catch (e: any) {
      setError(e.message || "An unknown authentication error occurred.");
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="h-screen flex flex-col items-center justify-center bg-zinc-100 dark:bg-zinc-900">
      <div className="w-full max-w-sm rounded-xl bg-white dark:bg-zinc-950 p-6 shadow">
        <h2 className="text-lg font-semibold mb-4">
          {mode === "login" ? "Login" : "Register"}
        </h2>
        <input
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          placeholder="Email"
          className="mb-2 w-full rounded-md border px-3 py-2"
          disabled={loading}
        />
        <input
          type="password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          placeholder="Password"
          className="mb-4 w-full rounded-md border px-3 py-2"
          disabled={loading}
        />
        {error && <div className="text-red-500 text-sm mb-2 bg-red-100 dark:bg-red-900/20 p-2 rounded-md">{error}</div>}
        <button
          onClick={submit}
          disabled={loading}
          className="w-full rounded-md bg-emerald-600 text-white py-2 flex items-center justify-center gap-2 disabled:opacity-60"
        >
          {loading && <Loader2 className="w-4 h-4 animate-spin" />}
          {loading ? "Please wait..." : (mode === "login" ? "Login" : "Register")}
        </button>
        <button
          onClick={() => setMode(mode === "login" ? "register" : "login")}
          className="mt-3 text-sm text-blue-500 underline disabled:opacity-60"
          disabled={loading}
        >
          {mode === "login" ? "Create an account" : "Have an account? Login"}
        </button>
      </div>
    </div>
  );
}


// ---------- Main Component ----------
function ProfileModal({
  open,
  onClose,
  user,
  onUpdate,
}: {
  open: boolean;
  onClose: () => void;
  user: User;
  onUpdate: (profile: any) => void;
}) {
  const [age, setAge] = useState<number | "">("");
  const [sex, setSex] = useState("");
  const [conditions, setConditions] = useState("");
  const [medications, setMedications] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);

  // prefill when opening
  useEffect(() => {
    if (!open) return;
    const p = user.profile || {};
    setAge(p.age ?? "");
    setSex(p.sex ?? "");
    setConditions(Array.isArray(p.conditions) ? p.conditions.join(", ") : "");
    setMedications(Array.isArray(p.medications) ? p.medications.join(", ") : "");
    setMsg(null);
  }, [open, user.profile]);

  async function save() {
    setSaving(true);
    setMsg(null);
    try {
      const updated = await apiUpdateProfile({
        age: age === "" ? null : Number(age),
        sex: sex || null,
        conditions: conditions
          ? conditions.split(",").map((s) => s.trim()).filter(Boolean)
          : [],
        medications: medications
          ? medications.split(",").map((s) => s.trim()).filter(Boolean)
          : [],
      });
      onUpdate(updated);
      setMsg("Profile saved successfully!");
      setTimeout(() => onClose(), 1000); // Close after a short delay
    } catch (e: any) {
      setMsg(`Save failed: ${e?.message || "Unknown error"}`);
    } finally {
      setSaving(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      onMouseDown={(e) => {
        if (e.currentTarget === e.target) onClose();
      }}
    >
      <div className="w-full max-w-lg rounded-2xl bg-white dark:bg-zinc-950 border border-zinc-200 dark:border-zinc-800 shadow-xl p-5">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold">Profile Settings</h2>
          <button
            onClick={onClose}
            className="rounded-lg px-2 py-1 hover:bg-zinc-100 dark:hover:bg-zinc-900"
            aria-label="Close"
          >
            âœ•
          </button>
        </div>

        <div className="space-y-3">
          <input
            type="number"
            value={age}
            onChange={(e) => setAge(e.target.value ? parseInt(e.target.value) : "")}
            placeholder="Age"
            className="w-full rounded-md border px-3 py-2 bg-white dark:bg-zinc-950"
          />
          <select
            value={sex}
            onChange={(e) => setSex(e.target.value)}
            className="w-full rounded-md border px-3 py-2 bg-white dark:bg-zinc-950"
          >
            <option value="">Sex</option>
            <option value="male">male</option>
            <option value="female">female</option>
            <option value="other">other</option>
          </select>
          <input
            value={conditions}
            onChange={(e) => setConditions(e.target.value)}
            placeholder="Conditions (comma-separated)"
            className="w-full rounded-md border px-3 py-2 bg-white dark:bg-zinc-950"
          />
          <input
            value={medications}
            onChange={(e) => setMedications(e.target.value)}
            placeholder="Medications (comma-separated)"
            className="w-full rounded-md border px-3 py-2 bg-white dark:bg-zinc-950"
          />

          {msg && <div className={`text-sm ${msg.startsWith('Save failed') ? 'text-red-500' : 'text-green-500'}`}>{msg}</div>}

          <div className="pt-2 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl px-3 py-2 border hover:bg-zinc-100 dark:hover:bg-zinc-900"
            >
              Cancel
            </button>
            <button
              onClick={save}
              disabled={saving}
              className="rounded-xl px-3 py-2 bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-60 flex items-center gap-2"
            >
              {saving && <Loader2 className="w-4 h-4 animate-spin"/>}
              {saving ? "Savingâ€¦" : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ClassicChatbotUI() {
  const [user, setUser] = useState<User | null>(null);
  

  useEffect(() => {
    const saved = localStorage.getItem("auth");
    if (saved) {
      try { setUser(JSON.parse(saved)); } catch {}
    }
  }, []);

  const initialConversations: Conversation[] = [
    { id: "c1", title: "Welcome", messages: seedMessages() },
  ];

  // Per-conversation analysis state
  const [structuredById, setStructuredById] = useState<Record<string, any>>({});
  const [explanationById, setExplanationById] = useState<Record<string, string>>({});
  const [explanationSourceById, setExplanationSourceById] = useState<Record<string, 'model'|'fallback'|'skipped'|undefined>>({});
  const [dark, setDark] = useState(true);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [profileOpen, setProfileOpen] = useState(false);
  const [conversations, setConversations] = useState<Conversation[]>(initialConversations);
  const [activeId, setActiveId] = useState<string | null>(() => initialConversations[0]?.id ?? null);
  const [input, setInput] = useState("");
  const [missingFieldsById, setMissingFieldsById] = useState<Record<string, string[]>>({});
  const [hideProfileBanner, setHideProfileBanner] = useState<boolean>(() => {
    try { return sessionStorage.getItem('hideProfileBanner') === '1'; } catch { return false; }
  });
  const structured = activeId ? structuredById[activeId] ?? null : null;
  const explanation = activeId ? explanationById[activeId] ?? "" : "";
  // Precompute sanitized explanation at top-level so hooks order remains stable
  const sanitizedExplanation = useMemo(() => {
    const text = explanation && explanation.trim().length > 0
      ? explanation
      : "AI explanation unavailable; showing structured recommendations.";
    return DOMPurify.sanitize(text, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
  }, [explanation]);
  const [busy, setBusy] = useState(false);
  const [fileBusy, setFileBusy] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const uploadInputRef = useRef<HTMLInputElement>(null);
  type SymptomAnalysisResult = {
    urgency: string;
    summary: string;
    symptoms: { text: string; label: string; score: number; negated?: boolean }[];
  };
  const [symptomAnalysisResult, setSymptomAnalysisResult] = useState<SymptomAnalysisResult | null>(null);

  const active = useMemo(() => {
    return conversations.find((c) => c.id === activeId) ?? null;
  }, [conversations, activeId]);

  useEffect(() => {
    if (dark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  }, [dark]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight, behavior: "smooth" });
  }, [active?.messages?.length]);

  function newConversation() {
    const id = `c_${Date.now()}`;
    const convo: Conversation = { id, title: "New chat", messages: [] };
    setConversations((prev) => [convo, ...prev]);
    setActiveId(id);
    // init per-conversation analysis slots without wiping others
    setStructuredById((s) => ({ ...s, [id]: null }));
    setExplanationById((s) => ({ ...s, [id]: "" }));
    setExplanationSourceById((s) => ({ ...s, [id]: undefined }));
    setSymptomAnalysisResult(null);
  }

  function deleteConversation(id: string) {
    setConversations((prev) => {
      const nextActive = nextActiveIdAfterDelete(prev, activeId, id);
      setActiveId(nextActive);
      return prev.filter((c) => c.id !== id);
    });
    // tidy analysis maps
    setStructuredById((s) => {
      const copy = { ...s };
      delete copy[id];
      return copy;
    });
    setExplanationById((s) => {
      const copy = { ...s };
      delete copy[id];
      return copy;
    });
    if (activeId === id) {
      setSymptomAnalysisResult(null);
    }
  }

  function applyToActive(mutator: (msgs: Message[]) => Message[]) {
    if (!activeId) return;
    setConversations((prev) =>
      prev.map((c) => (c.id === activeId ? { ...c, messages: mutator(c.messages) } : c))
    );
  }

  async function handleAnalyzeSymptoms() {
    const text = input.trim();
    if (!text || !user) return;

    setBusy(true);
    try {
      const result = await apiAnalyzeSymptoms(text);
      setSymptomAnalysisResult(result);
      const botMessage: Message = {
        id: `m_${Date.now() + 1}`,
        role: "assistant",
        content: result.summary || "Symptom analysis complete.",
        ts: new Date(),
      };
      applyToActive((msgs) => [...msgs, botMessage]);
    } catch (err: any) {
      // Handle error in UI if needed
    } finally {
      setBusy(false);
    }
  }
  async function handleSend() {
    const text = input.trim();
    if (!text || !user) return;

    setBusy(true);
    const newUserMessage: Message = { id: `m_${Date.now()}`, role: "user", content: text, ts: new Date() };
    applyToActive((msgs) => [...msgs, newUserMessage]);
    setInput("");

    try {
      const chatResponse: ChatResponseCombined = await apiChat(text);
      // Update side panels with pipeline + AI explanation
      if (activeId) {
        setStructuredById((s) => ({ ...s, [activeId]: chatResponse.pipeline || null }));
        setExplanationById((s) => ({ ...s, [activeId]: chatResponse.ai_explanation || '' }));
        setExplanationSourceById((s) => ({ ...s, [activeId]: (chatResponse.ai_explanation_source as any) }));
        if (Array.isArray(chatResponse.missing_fields)) {
          setMissingFieldsById((m) => ({ ...m, [activeId]: chatResponse.missing_fields as any }));
          setHideProfileBanner((prev) => prev || (chatResponse.missing_fields || []).length === 0);
        }
      }
      const assistantText = chatResponse.ai_explanation || chatResponse.summary || "";
      const safe = DOMPurify.sanitize(assistantText, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
      const botMessage: Message = {
        id: `m_${Date.now() + 1}`,
        role: "assistant",
        content: safe,
        ts: new Date(),
        requestId: chatResponse.request_id,
        symptomAnalysis: chatResponse.symptom_analysis,
        localRecommendations: chatResponse.local_recommendations,
        disclaimer: chatResponse.disclaimer,
        rawKeys: Object.keys(chatResponse || {}),
        aiSource: (chatResponse as any).ai_explanation_source,
      };
      applyToActive((msgs) => [...msgs, botMessage]);
    } catch (err: any) {
      const botMessage: Message = {
        id: `m_${Date.now() + 1}`,
        role: "assistant",
        content: DOMPurify.sanitize(`Error: ${err?.message || "Something went wrong."}`, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] }),
        ts: new Date(),
      };
      applyToActive((msgs) => [...msgs, botMessage]);
    } finally {
      setBusy(false);
    }
  }

  async function onFilePicked(file: File) {
    setFileBusy(true);
    try {
      const { text } = await apiExtract(file);
      setInput(text);
      // After upload, re-run the last user message to reflect improved context
      rerunLastMessage(/*reason*/'lab_upload');
    } catch (e: any) {
      const botMsg: Message = {
        id: `m_${Date.now() + 2}`,
        role: "assistant",
        content: DOMPurify.sanitize(`File extraction failed: ${e?.message || "An unknown error occurred."} Please try again.`, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] }),
        ts: new Date(),
      };
      applyToActive((msgs) => [...msgs, botMsg]);
    } finally {
      setFileBusy(false);
    }
  }

  function rerunLastMessage(reason?: string) {
    if (!active) return;
    const lastUser = [...(active.messages || [])].reverse().find((m) => m.role === 'user');
    if (!lastUser) return;
    // dev hint: show previous req id for correlation
    if (process.env.NODE_ENV !== 'production') {
      const lastAssistant = [...(active.messages || [])].reverse().find((m) => m.role === 'assistant' && m.requestId);
      // eslint-disable-next-line no-console
      console.debug('Re-running last message', { reason, prev_request_id: lastAssistant?.requestId, prompt: lastUser.content });
    }
    (async () => {
      try {
        setBusy(true);
        const resp: ChatResponseCombined = await apiChat(lastUser.content);
        if (activeId) {
          setStructuredById((s) => ({ ...s, [activeId]: resp.pipeline || null }));
          setExplanationById((s) => ({ ...s, [activeId]: resp.ai_explanation || '' }));
          setExplanationSourceById((s) => ({ ...s, [activeId]: (resp.ai_explanation_source as any) }));
          if (Array.isArray(resp.missing_fields)) {
            setMissingFieldsById((m) => ({ ...m, [activeId]: resp.missing_fields as any }));
          }
        }
        const assistantText = resp.ai_explanation || resp.summary || "";
        const safeText = DOMPurify.sanitize(assistantText, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
        const msg: Message = {
          id: `m_${Date.now() + 3}`,
          role: 'assistant',
          content: safeText,
          ts: new Date(),
          requestId: resp.request_id,
          symptomAnalysis: resp.symptom_analysis,
          localRecommendations: resp.local_recommendations,
          disclaimer: resp.disclaimer,
          rawKeys: Object.keys(resp || {}),
          aiSource: (resp as any).ai_explanation_source,
        };
        applyToActive((msgs) => [...msgs, msg]);
        try { sessionStorage.setItem('hideProfileBanner', '1'); } catch {}
        setHideProfileBanner(true);
      } finally {
        setBusy(false);
      }
    })();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  useEffect(() => {
    setActiveId((curr) => safeActiveId(conversations, curr));
  }, [conversations.length]);

  if (!user) {
    return <AuthScreen onAuth={setUser} />;
  }

  function setShowProfile(value: boolean) {
    setProfileOpen(value);
  }
  return (
    <div className="h-screen w-full overflow-hidden bg-zinc-50 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-50">
      {/* Top bar (mobile) */}
      <div className="lg:hidden flex items-center justify-between p-3 border-b border-zinc-200 dark:border-zinc-800">
        <button className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800" onClick={() => setSidebarOpen((s) => !s)}>
          <Menu className="w-5 h-5" />
        </button>
        <div className="font-semibold">Classic Chatbot</div>
        <button className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800" onClick={() => setDark((d) => !d)}>
          {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </div>

      <div className="h-[calc(100vh-0px)] grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_360px]">
        {/* Sidebar */}
          <aside
            className={`min-h-0 border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 ${
              sidebarOpen ? "block" : "hidden"
            } lg:block`}
          >
            {/* Column layout so only the middle scrolls */}
            <div className="h-full flex flex-col min-h-0">
              {/* TOP: actions bar (stays put) */}
              <div className="shrink-0 px-3 py-2 border-b border-zinc-200 dark:border-zinc-800
                              bg-white/80 dark:bg-zinc-950/80 backdrop-blur">
                <div className="flex items-center gap-2">
                  <button
                    onClick={newConversation}
                    className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium
                              bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90"
                  >
                    <Plus className="w-4 h-4" /> New chat
                  </button>
                  <div className="ml-auto" />
                  <button
                    onClick={() => setDark((d) => !d)}
                    className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800"
                    aria-label="Toggle theme"
                  >
                    {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
                  </button>
                </div>
              </div>

              {/* MIDDLE: scrollable conversations */}
              <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain p-2 space-y-2">
                {conversations.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => setActiveId(c.id)}
                    className={`group w-full flex items-center gap-2 rounded-xl px-3 py-2 text-left
                              hover:bg-zinc-100 dark:hover:bg-zinc-800
                              ${activeId === c.id ? "bg-zinc-100 dark:bg-zinc-800" : ""}`}
                  >
                    <MessageSquare className="w-4 h-4 shrink-0" />
                    <span className="flex-1 truncate">{c.title}</span>
                    <span className="text-xs text-zinc-400">{c.messages.length}</span>
                    <span
                      onClick={(e) => {
                        e.stopPropagation();
                        deleteConversation(c.id);
                      }}
                      className="opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700"
                    >
                      <Trash2 className="w-4 h-4" />
                    </span>
                  </button>
                ))}
                {!conversations.length && (
                  <div className="text-xs text-zinc-500 px-3 py-2">No conversations yet.</div>
                )}
              </div>

              {/* BOTTOM: user menu (stays put) */}
              <div className="shrink-0 border-t border-zinc-200 dark:border-zinc-800 p-3">
                <UserMenu
                  user={{ email: user?.email || "" }}
                  onProfile={() => setProfileOpen(true)}
                  onLogout={() => {
                    setUser(null);
                    localStorage.removeItem("auth");
                  }}
                />
              </div>
            </div>
          </aside>
            {user && (
              <ProfileModal
                open={profileOpen}
                onClose={() => setProfileOpen(false)}
                user={user}
                onUpdate={(p) => {
                  const updated = { ...user, profile: p };
                  setUser(updated);
                  localStorage.setItem("auth", JSON.stringify(updated));
                  // Re-run last message to reflect improved profile context
                  rerunLastMessage('profile_update');
                  try { sessionStorage.setItem('hideProfileBanner','1'); } catch {}
                  setHideProfileBanner(true);
                }}
              />
            )}

        {/* Main chat */}
        <main className="relative flex flex-col min-h-0">
          <div className="hidden lg:flex items-center justify-between px-5 py-3 border-b border-zinc-200 dark:border-zinc-800">
            <div className="font-semibold tracking-tight">{active?.title || "Chat"}</div>
          </div>

          {!hideProfileBanner && Array.isArray(missingFieldsById[activeId || '']) && (missingFieldsById[activeId || '']?.length || 0) > 0 && (
            <div className="mx-4 mt-3 rounded-xl border border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-200 p-3 flex items-center gap-3">
              <ShieldAlert className="w-5 h-5" />
              <div className="flex-1 text-sm">
                <div className="font-medium mb-0.5">To personalize guidance, add: {missingFieldsById[activeId || ''].join(', ')}.</div>
                <div className="text-[12px] opacity-80">This banner appears once per session.</div>
              </div>
              <div className="flex items-center gap-2">
                <button
                  className="text-xs px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  onClick={() => setProfileOpen(true)}
                >
                  Open Profile
                </button>
                <button
                  className="text-xs px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  onClick={() => uploadInputRef.current?.click()}
                >
                  Upload Lab
                </button>
                <button
                  className="text-xs px-2 py-1 rounded-lg text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                  onClick={() => { try { sessionStorage.setItem('hideProfileBanner','1'); } catch {}; setHideProfileBanner(true); }}
                >
                  Don’t show again
                </button>
              </div>
            </div>
          )}
          <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto p-4 lg:p-6 space-y-4">
            {active?.messages?.length ? (
              active.messages.map((m) => <ChatMessage key={m.id} msg={m} />)
            ) : (
              <EmptyState />
            )}
          </div>

          {/* Horizontal Composer with "+" upload */}
          <div className="px-4 lg:px-6 py-3 bg-gradient-to-t from-zinc-50 via-zinc-50/90 to-transparent dark:from-zinc-900 dark:via-zinc-900/90">
            <div className="mx-auto max-w-3xl">
              <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-sm p-2">
                <div className="flex items-end gap-2">
                  {/* + (Upload) */}
                  <label
                    className={`shrink-0 inline-flex items-center justify-center rounded-xl p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 cursor-pointer ${fileBusy ? 'animate-pulse' : ''}`}
                    title="Upload (PDF/Image/Text)"
                  >
                    <input
                      type="file"
                      className="hidden"
                      ref={uploadInputRef}
                      onChange={(e) => e.target.files && onFilePicked(e.target.files[0])}
                      disabled={fileBusy}
                    />
                    {fileBusy ? <Loader2 className="w-5 h-5 animate-spin" /> : <Plus className="w-5 h-5" />}
                  </label>

                  {/* Textarea grows horizontally */}
                  <textarea
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={onKeyDown}
                    rows={1}
                    placeholder={fileBusy ? "Extracting text from file..." : "Paste lab text or type your question..."}
                    className="flex-1 max-h-40 h-12 resize-none bg-transparent px-2 py-2 outline-none placeholder:text-zinc-400 text-sm"
                    disabled={fileBusy}
                  />

                  {/* Send */}
                  <button
                    onClick={handleSend}
                    className="shrink-0 inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90 disabled:opacity-40"
                    disabled={!input.trim() || busy || fileBusy}
                    aria-label="Send"
                  >
                    {busy ? <><Loader2 className="w-4 h-4 animate-spin" /> Working...</> : <><Send className="w-4 h-4" /> Send</>}
                  </button>
                  {/* Analyze Symptoms Button */}
                  <button
                    onClick={handleAnalyzeSymptoms}
                    className="shrink-0 inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium bg-amber-600 text-white hover:bg-amber-500 disabled:opacity-40"
                    disabled={!input.trim() || busy || fileBusy}
                    title="Analyze for Symptoms"
                  >
                    {busy ? <><Loader2 className="w-4 h-4 animate-spin" /> Working...</> : <><Send className="w-4 h-4" /> Send</>}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </main>

        {/* Right rail: Structured JSON & Explanation */}
        <section className="hidden lg:flex flex-col overflow-hidden border-l border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
          <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 text-sm font-semibold">Analysis</div>
          <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
            <div className="text-xs uppercase tracking-wide text-zinc-400">Structured (BioBERT-style)</div>
            <pre className="text-xs whitespace-pre-wrap break-words rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900">
              {structured ? JSON.stringify(structured, null, 2) : "â€”"}
            </pre>

            <div className="text-xs uppercase tracking-wide text-zinc-400">Parsed values</div>
            <div className="text-[11px] text-zinc-400 mb-1">Language: {structured?.language || "unknown"}</div>
            <ul className="text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 space-y-2">
              {(structured?.tests || []).map((t: any, i: number) => (
                <li key={i} className="flex items-center justify-between gap-2">
                  <div>
                    <span className="font-medium">{t.name}</span>: {t.value} {t.unit}
                    {t.reference ? ` (${prettyRef(t.reference)})` : ""}
                  </div>
                  <span
                    className={`text-[11px] px-2 py-0.5 rounded-full border ${
                      t.status === "high"
                        ? "border-red-400 text-red-600"
                        : t.status === "low"
                        ? "border-blue-400 text-blue-600"
                        : t.status === "normal"
                        ? "border-emerald-400 text-emerald-600"
                        : "border-zinc-300 text-zinc-500"
                    }`}
                  >
                    {t.status}
                  </span>
                </li>
              ))}
            </ul>

            <div className="text-xs uppercase tracking-wide text-zinc-400">Detected terms</div>
            <ul className="text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 space-y-2">
              {structured?.entities?.length ? (
                structured.entities.map((ent: any, i: number) => (
                  <li key={i}>
                    <div className="flex flex-col gap-1">
                      <div className="flex items-center justify-between">
                        <span className="font-medium">{ent.term || ent.text}</span>
                        <span className="text-[11px] text-zinc-400">{ent.label || ent.entity}{ent.source ? ` | ${ent.source === "gemini" ? "Gemini" : "Glossary"}` : ""}</span>
                      </div>
                      {ent.explanation ? (
                        <p className="text-[13px] text-zinc-500 dark:text-zinc-400">{ent.explanation}</p>
                      ) : (
                        <p className="text-[13px] text-zinc-500 dark:text-zinc-400 italic">No glossary description available.</p>
                      )}
                    </div>
                  </li>
                ))
              ) : (
                <li className="text-[13px] text-zinc-500 dark:text-zinc-400 italic">No medical terms detected.</li>
              )}
            </ul>

            <div className="flex items-center justify-between">
              <div className="text-xs uppercase tracking-wide text-zinc-400">AI Explanation</div>
              <div className="flex items-center gap-2">
                {process.env.NODE_ENV !== 'production' && (
                  <span className="text-[10px] text-zinc-400">source: {explanationSourceById[activeId || ''] || 'unknown'}</span>
                )}
                <button
                  className="text-[11px] px-2 py-0.5 rounded border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 flex items-center gap-1"
                  onClick={() => {
                    try {
                      navigator.clipboard.writeText(explanation || "");
                    } catch {}
                  }}
                  title="Copy AI explanation"
                >
                  <Copy className="w-3 h-3" /> Copy
                </button>
              </div>
            </div>
            <div className="text-xs uppercase tracking-wide text-zinc-400">User Profile</div>
            <pre className="text-xs whitespace-pre-wrap break-words rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900">
              {user?.profile ? JSON.stringify(user.profile, null, 2) : "â€”"}
            </pre>
            <div className="text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 min-h-[120px]">
              {sanitizedExplanation}
            </div>
          </div>
          {symptomAnalysisResult && (
            <div className="p-3 space-y-3">
              <div className="text-xs uppercase tracking-wide text-zinc-400">Symptom Analysis</div>
              <div className={`text-sm rounded-xl border p-3 bg-zinc-50 dark:bg-zinc-900
                ${symptomAnalysisResult.urgency === 'urgent' ? 'border-red-500' : 'border-zinc-200 dark:border-zinc-800'}`}>
                <div className="flex items-center gap-2 mb-2">
                  <ShieldAlert className={`w-5 h-5 ${symptomAnalysisResult.urgency === 'urgent' ? 'text-red-500' : 'text-amber-500'}`} />
                  <span className="font-semibold">Urgency: {symptomAnalysisResult.urgency}</span>
                </div>
                <p className="text-xs text-zinc-500 dark:text-zinc-400 mb-3">{symptomAnalysisResult.summary}</p>
                <ul className="space-y-1 text-xs">
                  {symptomAnalysisResult.symptoms.map((s, i) => (
                    <li key={i} className={`flex justify-between ${s.negated ? 'line-through text-zinc-400' : ''}`}>
                      <span>{s.text}</span>
                      <span className="font-mono text-zinc-500">{s.label} ({(s.score * 100).toFixed(0)}%)</span>
                    </li>
                  ))}
                  {symptomAnalysisResult.symptoms.length === 0 && (
                    <li className="italic text-zinc-400">No specific symptoms detected.</li>
                  )}
                </ul>
              </div>
            </div>
          )}
        </section>
      </div>
    </div>
  );
  
}

function ChatMessage({ msg }: { msg: Message }) {
  const isUser = msg.role === "user";
  return (
    <div className={`flex items-start gap-3 ${isUser ? "flex-row-reverse" : ""}`}>
      <div className={`rounded-full p-2 ${isUser ? "bg-blue-600 text-white" : "bg-emerald-600 text-white"}`}>
        {isUser ? <User className="w-4 h-4" /> : <Bot className="w-4 h-4" />}
      </div>
      <div
        className={`max-w-[78%] lg:max-w-[70%] rounded-2xl px-4 py-3 border text-sm shadow-sm leading-relaxed whitespace-pre-wrap ${
          isUser
            ? "bg-blue-50 dark:bg-blue-950/30 border-blue-200 dark:border-blue-900"
            : "bg-white dark:bg-zinc-950 border-zinc-200 dark:border-zinc-800"
        }`}
      >
        <div>{useMemo(() => DOMPurify.sanitize(msg.content, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] }), [msg.content])}</div>
        {/* Render structured extras for assistant messages */}
        {!isUser && msg.localRecommendations && (
          <div className="mt-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50 dark:bg-zinc-900 p-3">
            <div className="text-xs uppercase tracking-wide text-zinc-500 mb-1">Recommendations</div>
            <div className="text-[13px] mb-2">Priority: {msg.localRecommendations.priority || 'low'}</div>
            {/* DEV/Info: show event_id and confidence from symptom_analysis to ensure we read them */}
            {(msg.symptomAnalysis?.event_id || typeof msg.symptomAnalysis?.confidence === 'number') && (
              <div className="text-[11px] text-zinc-500 mb-2">
                {msg.symptomAnalysis?.event_id ? <span>Event: {msg.symptomAnalysis.event_id}</span> : null}
                {msg.symptomAnalysis?.event_id && typeof msg.symptomAnalysis?.confidence === 'number' ? <span> • </span> : null}
                {typeof msg.symptomAnalysis?.confidence === 'number' ? (
                  <span>Confidence: {Math.round((msg.symptomAnalysis.confidence || 0) * 100)}%</span>
                ) : null}
              </div>
            )}
            <ul className="list-disc pl-5 space-y-1 text-[13px]">
              {msg.localRecommendations.actions?.map((a, i) => (
                <li key={i}>{a}</li>
              ))}
            </ul>
            {msg.symptomAnalysis?.symptoms?.length ? (
              <div className="mt-2 flex flex-wrap gap-1">
                {msg.symptomAnalysis.symptoms.slice(0, 8).map((s, i) => (
                  <span key={i} className="px-2 py-0.5 rounded-full bg-zinc-200 dark:bg-zinc-800 text-[11px]">{s}</span>
                ))}
              </div>
            ) : null}
            {msg.disclaimer && (
              <div className="mt-3 text-[11px] text-zinc-500">{msg.disclaimer}</div>
            )}
          </div>
        )}
        {/* DEV-only keys list to confirm payload shape */}
        {!isUser && process.env.NODE_ENV !== 'production' && msg.rawKeys?.length ? (
          <div className="mt-1 text-[10px] text-zinc-400">keys: [{msg.rawKeys.join(', ')}]</div>
        ) : null}
        <div className="mt-1 text-[10px] text-zinc-400 flex items-center justify-between">
          <span>{formatTime(msg.ts)}</span>
          {/* DEV-only footer showing request_id and source */}
          {!isUser && process.env.NODE_ENV !== 'production' ? (
            <span className="ml-2">
              {msg.requestId ? `req: ${msg.requestId}` : ''}
              {msg.requestId && msg.aiSource ? ' | ' : ''}
              {msg.aiSource ? `source: ${msg.aiSource}` : ''}
            </span>
          ) : null}
        </div>
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="mx-auto mb-4 w-12 h-12 rounded-2xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center">
          <Bot className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-semibold">Ask anything</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Upload a file (PDF/Image) or paste lab text, then hit Send. We'll structure entities with heuristics/BioBERT and explain with Gemini.
        </p>
      </div>
    </div>
  );
}

// ---------- Minimal runtime tests ----------
(function runtimeTests() {
  try {
    const A: Conversation = { id: "A", title: "A", messages: [] };
    const B: Conversation = { id: "B", title: "B", messages: [] };
    console.assert(nextActiveIdAfterDelete([A, B], "A", "A") === "B", "Test#1 failed");
    console.assert(nextActiveIdAfterDelete([A], "A", "A") === null, "Test#2 failed");
    console.assert(nextActiveIdAfterDelete([A, B], "B", "A") === "B", "Test#3 failed");
    console.assert(safeActiveId([A, B], "B") === "B", "Test#4 failed");
    console.assert(safeActiveId([A, B], "Z") === "A", "Test#5 failed");
    console.assert(safeActiveId([], "Z") === null, "Test#6 failed");
    console.assert(nextActiveIdAfterDelete([], null, "X") === null, "Test#7 failed");
    if (!(window as any).__chat_ui_tests__) {
      (window as any).__chat_ui_tests__ = true;
      console.log("ClassicChatbotUI runtime tests passed âœ…");
    }
  } catch {}
})();
