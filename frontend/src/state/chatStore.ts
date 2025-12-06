import { create } from "zustand";
import type { Conversation, Message, SymptomAnalysisResult } from "../types";
// Local helpers for conversation state transitions
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
      content: "",
    },
  ];
}

const CHAT_STORAGE_PREFIX = "medibot.chat";
const DEFAULT_SCOPE = "guest";

function deriveScopeFromUser(user: { id?: string | number; email?: string } | null | undefined): string {
  if (user && user.id !== undefined && user.id !== null) return String(user.id);
  if (user && user.email) return String(user.email);
  return DEFAULT_SCOPE;
}

function deriveScopeFromLocalAuth(): string {
  try {
    const saved = localStorage.getItem("auth");
    if (saved) {
      const parsed = JSON.parse(saved);
      return deriveScopeFromUser(parsed);
    }
  } catch {
    // ignore malformed auth payloads
  }
  return DEFAULT_SCOPE;
}

function chatStorageKey(scope: string): string {
  return `${CHAT_STORAGE_PREFIX}.${scope || DEFAULT_SCOPE}`;
}

export function chatScopeForUser(user: { id?: string | number; email?: string } | null | undefined): string {
  return deriveScopeFromUser(user);
}

type ChatState = {
  storageScope: string;
  conversations: Conversation[];
  activeId: string | null;
  structuredById: Record<string, any>;
  explanationById: Record<string, string>;
  explanationSourceById: Record<string, "model" | "fallback" | "skipped" | undefined>;
  missingFieldsById: Record<string, string[]>;
  hideProfileBanner: boolean;
  triageById: Record<string, { level: string; reasons?: string[] }>;
  urgentAckById: Record<string, boolean>;
  devMode: boolean;
  dark: boolean;
  sidebarOpen: boolean;
  busy: boolean;
  fileBusy: boolean;
  symptomAnalysisResult: SymptomAnalysisResult | null;
  profileOpen: boolean;
  input: string;
};

type ChatActions = {
  setDark: (v: boolean) => void;
  setSidebarOpen: (v: boolean) => void;
  setDevMode: (v: boolean) => void;
  setProfileOpen: (v: boolean) => void;
  setInput: (v: string) => void;
  setBusy: (v: boolean) => void;
  setFileBusy: (v: boolean) => void;
  setHideProfileBanner: (v: boolean) => void;
  setSymptomAnalysisResult: (v: SymptomAnalysisResult | null) => void;
  setActiveId: (v: string | null) => void;
  setStructuredFor: (id: string, data: any) => void;
  setExplanationFor: (id: string, text: string) => void;
  setExplanationSourceFor: (id: string, source: "model" | "fallback" | "skipped" | undefined) => void;
  setMissingFieldsFor: (id: string, fields: string[]) => void;
  setTriageFor: (id: string, triage: { level: string; reasons?: string[] } | null) => void;
  setUrgentAckFor: (id: string, ack: boolean) => void;
  applyToActive: (mutator: (msgs: Message[]) => Message[]) => void;
  addConversation: (title?: string) => string;
  deleteConversation: (id: string) => void;
  setConversations: (c: Conversation[]) => void;
  resetChat: (scope?: string | null) => void;
};

export type ChatStore = ChatState & ChatActions;

function loadInitialState(scope?: string | null): ChatState {
  const storageScope = scope || deriveScopeFromLocalAuth();
  const storageKey = chatStorageKey(storageScope);
  let devMode = false;
  let dark = true;
  let sidebarOpen = true;
  let hideProfileBanner = false;
  try {
    const d = localStorage.getItem("medibot.devMode");
    devMode = d === "1" || d === "true";
    const dk = localStorage.getItem("medibot.dark");
    if (dk !== null) dark = dk === "1" || dk === "true";
    const sb = localStorage.getItem("medibot.sidebarOpen");
    if (sb !== null) sidebarOpen = sb === "1" || sb === "true";
    hideProfileBanner = sessionStorage.getItem("hideProfileBanner") === "1";

    const saved = localStorage.getItem(storageKey);
    if (saved) {
      const parsed = JSON.parse(saved);
      const reviveMsgs = (msgs: any[]) =>
        (msgs || []).map((m) => ({ ...m, ts: m?.ts ? new Date(m.ts) : new Date() }));
      const revivedConvos: Conversation[] = (parsed.conversations || []).map((c: any) => ({
        ...c,
        messages: reviveMsgs(c.messages || []),
      }));
      return {
        storageScope,
        conversations: revivedConvos,
        activeId: parsed.activeId ?? revivedConvos[0]?.id ?? null,
        structuredById: parsed.structuredById || {},
        explanationById: parsed.explanationById || {},
        explanationSourceById: parsed.explanationSourceById || {},
        missingFieldsById: parsed.missingFieldsById || {},
        hideProfileBanner,
        triageById: parsed.triageById || {},
        urgentAckById: parsed.urgentAckById || {},
        devMode,
        dark,
        busy: false,
        fileBusy: false,
        symptomAnalysisResult: parsed.symptomAnalysisResult || null,
        sidebarOpen,
        profileOpen: false,
        input: parsed.input || "",
      };
    }
  } catch {
    // ignore
  }
  const initialConversations: Conversation[] = [{ id: "c1", title: "Welcome", messages: seedMessages() }];
  return {
    storageScope,
    conversations: initialConversations,
    activeId: initialConversations[0]?.id ?? null,
    structuredById: {},
    explanationById: {},
    explanationSourceById: {},
    missingFieldsById: {},
    hideProfileBanner,
    triageById: {},
    urgentAckById: {},
    devMode,
    dark,
    busy: false,
    fileBusy: false,
    symptomAnalysisResult: null,
    sidebarOpen,
    profileOpen: false,
    input: "",
  };
}

export const useChatStore = create<ChatStore>((set, get) => {
  const persistDevMode = (v: boolean) => {
    try {
      localStorage.setItem("medibot.devMode", v ? "1" : "0");
    } catch {
      // ignore
    }
  };
  const persistDark = (v: boolean) => {
    try {
      localStorage.setItem("medibot.dark", v ? "1" : "0");
    } catch {
      // ignore
    }
  };
  const persistSidebar = (v: boolean) => {
    try {
      localStorage.setItem("medibot.sidebarOpen", v ? "1" : "0");
    } catch {
      // ignore
    }
  };
  const persistHideProfileBanner = (v: boolean) => {
    try {
      sessionStorage.setItem("hideProfileBanner", v ? "1" : "0");
    } catch {
      // ignore
    }
  };

  const initial = loadInitialState();

  return {
    ...initial,
    setDark: (v) => set((s) => (persistDark(v), { dark: v })),
    setSidebarOpen: (v) => set((s) => (persistSidebar(v), { sidebarOpen: v })),
    setDevMode: (v) => set((s) => (persistDevMode(v), { devMode: v })),
    setProfileOpen: (v) => set({ profileOpen: v }),
    setInput: (v) => set({ input: v }),
    setBusy: (v) => set({ busy: v }),
    setFileBusy: (v) => set({ fileBusy: v }),
    setHideProfileBanner: (v) => set((s) => (persistHideProfileBanner(v), { hideProfileBanner: v })),
    setSymptomAnalysisResult: (v) => set({ symptomAnalysisResult: v }),
    setActiveId: (v) => set({ activeId: v }),
    setStructuredFor: (id, data) => set((s) => ({ structuredById: { ...s.structuredById, [id]: data } })),
    setExplanationFor: (id, text) => set((s) => ({ explanationById: { ...s.explanationById, [id]: text } })),
    setExplanationSourceFor: (id, source) =>
      set((s) => ({ explanationSourceById: { ...s.explanationSourceById, [id]: source } })),
    setMissingFieldsFor: (id, fields) => set((s) => ({ missingFieldsById: { ...s.missingFieldsById, [id]: fields } })),
    setTriageFor: (id, triage) =>
      set((s) => ({
        triageById: triage ? { ...s.triageById, [id]: triage } : s.triageById,
      })),
    setUrgentAckFor: (id, ack) => set((s) => ({ urgentAckById: { ...s.urgentAckById, [id]: ack } })),
    applyToActive: (mutator) => {
      const activeId = get().activeId;
      if (!activeId) return;
      set((s) => ({
        conversations: s.conversations.map((c) => (c.id === activeId ? { ...c, messages: mutator(c.messages) } : c)),
      }));
    },
    addConversation: (title = "New chat") => {
      const id = `c_${Date.now()}`;
      const convo: Conversation = { id, title, messages: [] };
      set((s) => ({
        conversations: [convo, ...s.conversations],
        activeId: id,
        structuredById: { ...s.structuredById, [id]: null },
        explanationById: { ...s.explanationById, [id]: "" },
        explanationSourceById: { ...s.explanationSourceById, [id]: undefined },
        symptomAnalysisResult: null,
      }));
      return id;
    },
    deleteConversation: (id: string) => {
      set((s) => {
        const nextActive = nextActiveIdAfterDelete(s.conversations, s.activeId, id);
        const { [id]: _, ...structured } = s.structuredById;
        const { [id]: __, ...expl } = s.explanationById;
        const { [id]: ___, ...explSrc } = s.explanationSourceById;
        const { [id]: ____, ...miss } = s.missingFieldsById;
        const { [id]: _____, ...triage } = s.triageById;
        const { [id]: ______, ...urgent } = s.urgentAckById;
        const conversations = s.conversations.filter((c) => c.id !== id);
        return {
          conversations,
          activeId: nextActive,
          structuredById: structured,
          explanationById: expl,
          explanationSourceById: explSrc,
          missingFieldsById: miss,
          triageById: triage,
          urgentAckById: urgent,
          symptomAnalysisResult: nextActive === null ? null : s.symptomAnalysisResult,
        };
      });
    },
    setConversations: (convos) =>
      set((s) => ({
        conversations: convos,
        activeId: safeActiveId(convos, s.activeId),
      })),
    resetChat: (scope) => set(() => loadInitialState(scope)),
  };
});

// Persist a slice of chat state (conversations, analysis, input) to localStorage
useChatStore.subscribe((state) => {
  try {
    const payload = {
      conversations: state.conversations.map((c) => ({
        ...c,
        messages: (c.messages || []).map((m) => ({
          ...m,
          ts: m.ts instanceof Date ? m.ts.toISOString() : m.ts,
        })),
      })),
      activeId: state.activeId,
      structuredById: state.structuredById,
      explanationById: state.explanationById,
      explanationSourceById: state.explanationSourceById,
      missingFieldsById: state.missingFieldsById,
      triageById: state.triageById,
      urgentAckById: state.urgentAckById,
      symptomAnalysisResult: state.symptomAnalysisResult,
      input: state.input,
    };
    const storageKey = chatStorageKey(state.storageScope || deriveScopeFromLocalAuth());
    localStorage.setItem(storageKey, JSON.stringify(payload));
  } catch {
    // best-effort persistence
  }
});
