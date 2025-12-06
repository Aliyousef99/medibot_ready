import React, { useEffect, useMemo, useRef } from "react";
import DOMPurify from "dompurify";
import { Moon, Sun, Menu } from "lucide-react";
import AuthScreen from "./AuthScreen";
import ProfileModal from "./ProfileModal";
import ConversationList from "./ConversationList";
import ChatWindow from "./ChatWindow";
import AnalysisPanel from "./AnalysisPanel";
import { postChatMessage as apiChat, extractText as apiExtract } from "../services/api";
import type { ChatResponseCombined } from "../services/api";
import { useChatStore } from "../state/chatStore";
import { useAuth } from "../hooks/useAuth";
import { useToastStore } from "../state/toastStore";
import type { Message } from "../types";

function prettyRef(ref?: any) {
  if (!ref) return "";
  const u = ref.unit ? ` ${ref.unit}` : "";
  switch (ref.kind) {
    case "lte":
      return `ref <=${ref.v}${u}`;
    case "lt":
      return `ref <${ref.v}${u}`;
    case "gte":
      return `ref >=${ref.v}${u}`;
    case "gt":
      return `ref >${ref.v}${u}`;
    case "between":
      return `ref ${ref.lo}-${ref.hi}${u}`;
    default:
      return "";
  }
}

function ChatView() {
  const { user, setUser, logout } = useAuth();
  const { add: addToast } = useToastStore();
  const lastUserRef = useRef<string | null>(null);
  const conversations = useChatStore((s) => s.conversations);
  const activeId = useChatStore((s) => s.activeId);
  const structuredById = useChatStore((s) => s.structuredById);
  const explanationById = useChatStore((s) => s.explanationById);
  const explanationSourceById = useChatStore((s) => s.explanationSourceById);
  const missingFieldsById = useChatStore((s) => s.missingFieldsById);
  const hideProfileBanner = useChatStore((s) => s.hideProfileBanner);
  const triageById = useChatStore((s) => s.triageById);
  const urgentAckById = useChatStore((s) => s.urgentAckById);
  const devMode = useChatStore((s) => s.devMode);
  const dark = useChatStore((s) => s.dark);
  const busy = useChatStore((s) => s.busy);
  const fileBusy = useChatStore((s) => s.fileBusy);
  const symptomAnalysisResult = useChatStore((s) => s.symptomAnalysisResult);
  const sidebarOpen = useChatStore((s) => s.sidebarOpen);
  const profileOpen = useChatStore((s) => s.profileOpen);
  const input = useChatStore((s) => s.input);
  const actions = useChatStore();

  const active = useMemo(
    () => conversations.find((c) => c.id === activeId) ?? null,
    [conversations, activeId]
  );

  const structured = activeId ? structuredById[activeId] ?? null : null;
  const explanation = activeId ? explanationById[activeId] ?? "" : "";
  const sanitizedExplanation = useMemo(() => {
    const text =
      explanation && explanation.trim().length > 0
        ? explanation
        : "AI explanation unavailable; showing structured recommendations.";
    return DOMPurify.sanitize(text, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
  }, [explanation]);

  const uploadInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.ctrlKey || e.metaKey) && (e.key === "d" || e.key === "D")) {
        e.preventDefault();
        actions.setDevMode(!devMode);
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [devMode, actions]);

  // Reset chat state when switching users to avoid showing previous user's history
  useEffect(() => {
    const current = user?.email || null;
    if (current && lastUserRef.current && current !== lastUserRef.current) {
      actions.resetChat();
    }
    lastUserRef.current = current;
  }, [user, actions]);

  useEffect(() => {
    if (dark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  }, [dark]);

  // auth bootstrap handled by auth store; no local restore

  function newConversation() {
    const id = actions.addConversation("New chat");
    actions.setStructuredFor(id, null);
    actions.setExplanationFor(id, "");
    actions.setExplanationSourceFor(id, undefined);
    actions.setSymptomAnalysisResult(null);
  }

  function deleteConversation(id: string) {
    const wasActive = activeId === id;
    actions.deleteConversation(id);
    if (wasActive) {
      actions.setSymptomAnalysisResult(null);
    }
  }

  async function handleSend() {
    const text = input.trim();
    if (!text || !user) return;

    actions.setBusy(true);
    const newUserMessage: Message = { id: `m_${Date.now()}`, role: "user", content: text, ts: new Date() };
    actions.applyToActive((msgs) => [...msgs, newUserMessage]);
    actions.setInput("");

    try {
      const chatResponse: ChatResponseCombined = await apiChat(text);
      if (activeId) {
        actions.setStructuredFor(activeId, chatResponse.pipeline || null);
        actions.setExplanationFor(activeId, chatResponse.ai_explanation || "");
        actions.setExplanationSourceFor(activeId, chatResponse.ai_explanation_source as any);
        if (Array.isArray(chatResponse.missing_fields)) {
          actions.setMissingFieldsFor(activeId, chatResponse.missing_fields as any);
          actions.setHideProfileBanner(
            hideProfileBanner || (chatResponse.missing_fields || []).length === 0
          );
        }
        if (chatResponse.triage) {
          actions.setTriageFor(activeId, chatResponse.triage as any);
        }
        try {
          const sa = chatResponse.symptom_analysis as any;
          if (sa && (Array.isArray(sa.symptoms) || Array.isArray(sa.possible_tests))) {
            const n = (sa.symptoms || []).length;
            const conf = typeof sa.confidence === "number" ? Math.round(sa.confidence * 100) : undefined;
            const tri = chatResponse.triage as any;
            const urgency = tri && tri.level ? String(tri.level) : "low";
            const summary = `Detected ${n} symptom(s)${typeof conf === "number" ? ` (confidence ${conf}%)` : ""}.`;
            const items = (sa.symptoms || []).map((txt: string) => ({
              text: String(txt),
              label: "symptom",
              score: typeof sa.confidence === "number" ? sa.confidence : 0.5,
              negated: false,
            }));
            actions.setSymptomAnalysisResult({ urgency, summary, symptoms: items });
          }
        } catch {
          // best-effort mapping only
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
        rawResponse: chatResponse,
      };
      actions.applyToActive((msgs) => [...msgs, botMessage]);
    } catch (err: any) {
      addToast({ type: "error", message: err?.message || "Chat request failed. Please try again." });
    } finally {
      actions.setBusy(false);
    }
  }

  async function onFilePicked(file: File) {
    actions.setFileBusy(true);
    try {
      const { text } = await apiExtract(file);
      actions.setInput(text);
      rerunLastMessage("lab_upload");
    } catch (e: any) {
      addToast({ type: "error", message: e?.message || "File extraction failed. Please try again." });
    } finally {
      actions.setFileBusy(false);
    }
  }

  function rerunLastMessage(reason?: string) {
    if (!active) return;
    const lastUser = [...(active.messages || [])].reverse().find((m) => m.role === "user");
    if (!lastUser) return;
    if (process.env.NODE_ENV !== "production") {
      const lastAssistant = [...(active.messages || [])].reverse().find((m) => m.role === "assistant" && m.requestId);
      // eslint-disable-next-line no-console
      console.debug("Re-running last message", {
        reason,
        prev_request_id: lastAssistant?.requestId,
        prompt: lastUser.content,
      });
    }
    (async () => {
      try {
        actions.setBusy(true);
        const resp: ChatResponseCombined = await apiChat(lastUser.content);
        if (activeId) {
          actions.setStructuredFor(activeId, resp.pipeline || null);
          actions.setExplanationFor(activeId, resp.ai_explanation || "");
          actions.setExplanationSourceFor(activeId, resp.ai_explanation_source as any);
          if (Array.isArray(resp.missing_fields)) {
            actions.setMissingFieldsFor(activeId, resp.missing_fields as any);
          }
          if (resp.triage) actions.setTriageFor(activeId, resp.triage as any);
        }
        const assistantText = resp.ai_explanation || resp.summary || "";
        const safeText = DOMPurify.sanitize(assistantText, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] });
        const msg: Message = {
          id: `m_${Date.now() + 3}`,
          role: "assistant",
          content: safeText,
          ts: new Date(),
          requestId: resp.request_id,
          symptomAnalysis: resp.symptom_analysis,
          localRecommendations: resp.local_recommendations,
          disclaimer: resp.disclaimer,
          rawKeys: Object.keys(resp || {}),
          aiSource: (resp as any).ai_explanation_source,
          rawResponse: resp,
        };
        actions.applyToActive((msgs) => [...msgs, msg]);
        actions.setHideProfileBanner(true);
      } finally {
        actions.setBusy(false);
      }
    })();
  }

  function onKeyDown(e: React.KeyboardEvent<HTMLTextAreaElement>) {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  }

  if (!user) {
    return <AuthScreen onAuth={(u) => setUser(u)} />;
  }

  return (
    <div className="h-screen w-full overflow-hidden bg-zinc-50 text-zinc-900 dark:bg-zinc-900 dark:text-zinc-50">
      <div className="lg:hidden flex items-center justify-between p-3 border-b border-zinc-200 dark:border-zinc-800">
        <button className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800" onClick={() => actions.setSidebarOpen(!sidebarOpen)}>
          <Menu className="w-5 h-5" />
        </button>
        <div className="font-semibold">Classic Chatbot</div>
        <button className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800" onClick={() => actions.setDark(!dark)}>
          {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
        </button>
      </div>

      <div className="h-[calc(100vh-0px)] grid grid-cols-1 lg:grid-cols-[300px_minmax(0,1fr)_360px]">
        <div
          className={`min-h-0 h-full ${sidebarOpen ? "block" : "hidden"} lg:block`}
        >
          <ConversationList
            conversations={conversations}
            activeId={activeId}
            onSelect={(id) => actions.setActiveId(id)}
            onNew={newConversation}
            onDelete={deleteConversation}
            dark={dark}
            onToggleDark={() => actions.setDark(!dark)}
            user={user}
            onProfile={() => actions.setProfileOpen(true)}
            onLogout={() => logout()}
          />
        </div>
        {user && (
          <ProfileModal
            open={profileOpen}
            onClose={() => actions.setProfileOpen(false)}
            user={user}
            onUpdate={(p) => {
              const updatedUser = { ...user, profile: p };
              setUser(updatedUser);
              rerunLastMessage("profile_update");
              actions.setHideProfileBanner(true);
            }}
          />
        )}

        <ChatWindow
          active={active}
          devMode={devMode}
          triage={triageById[activeId || ""]}
          urgentAck={urgentAckById[activeId || ""]}
          onAckUrgent={() => actions.setUrgentAckFor(activeId || "", true)}
          missingFields={missingFieldsById[activeId || ""] || []}
          hideProfileBanner={hideProfileBanner}
          onOpenProfile={() => actions.setProfileOpen(true)}
          onHideProfileBanner={() => actions.setHideProfileBanner(true)}
          uploadInputRef={uploadInputRef}
          onUploadClick={() => uploadInputRef.current?.click()}
          onFilePicked={onFilePicked}
          messages={active?.messages || []}
          onSend={handleSend}
          onKeyDown={onKeyDown}
          input={input}
          setInput={actions.setInput}
          busy={busy}
          fileBusy={fileBusy}
        />

        <AnalysisPanel
          devMode={devMode}
          onToggleDevMode={(v) => actions.setDevMode(v)}
          structured={structured}
          explanation={explanation}
          explanationSource={explanationSourceById[activeId || ""]}
          user={user}
          sanitizedExplanation={sanitizedExplanation}
          prettyRef={prettyRef}
          activeId={activeId}
          symptomAnalysisResult={symptomAnalysisResult}
        />
      </div>
    </div>
  );
}

export default function ClassicChatbotUI() {
  return <ChatView />;
}
