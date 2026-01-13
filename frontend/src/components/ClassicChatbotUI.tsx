import React, { useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import { Moon, Sun, Menu } from "lucide-react";
import AuthScreen from "./AuthScreen";
import ProfileModal from "./ProfileModal";
import ConversationList from "./ConversationList";
import ChatWindow from "./ChatWindow";
import AnalysisPanel from "./AnalysisPanel";
import { postChatMessage as apiChat, postChatImage as apiChatImage, uploadLabAndSave, listConversations, createConversation, deleteConversationApi, getConversationMessages, setConsent, getProfile as apiGetProfile } from "../services/api";
import type { ChatResponseCombined, ConversationSummary, ConversationMessageRow } from "../services/api";
import { chatScopeForUser, useChatStore } from "../state/chatStore";
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

function isServerConversationId(id?: string | null): boolean {
  if (!id) return false;
  return /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(id);
}

function ChatView() {
  const { user, setUser, logout } = useAuth();
  const { add: addToast } = useToastStore();
  const lastScopeRef = useRef<string | null>(null);
  const profileLoadedRef = useRef<string | null>(null);
  const [analysisOpen, setAnalysisOpen] = useState(false);
  const [uploadStatus, setUploadStatus] = useState<"idle" | "loading" | "success" | "error">("idle");
  const [uploadError, setUploadError] = useState<string | null>(null);
  const [analysisState, setAnalysisState] = useState<"idle" | "loading" | "error">("idle");
  const [analysisError, setAnalysisError] = useState<string | null>(null);
  const [consentBusy, setConsentBusy] = useState(false);
  const consentWarnedRef = useRef(false);
  const conversations = useChatStore((s) => s.conversations);
  const activeId = useChatStore((s) => s.activeId);
  const structuredById = useChatStore((s) => s.structuredById);
  const explanationById = useChatStore((s) => s.explanationById);
  const explanationSourceById = useChatStore((s) => s.explanationSourceById);
  const missingFieldsById = useChatStore((s) => s.missingFieldsById);
  const hideProfileBanner = useChatStore((s) => s.hideProfileBanner);
  const triageById = useChatStore((s) => s.triageById);
  const urgentAckById = useChatStore((s) => s.urgentAckById);
  const recommendationsById = useChatStore((s) => (s as any).recommendationsById || {});
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

  const triage = triageById[activeId || ""];
  const recommendations = recommendationsById[activeId || ""];

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
    const scope = chatScopeForUser(user);
    if (lastScopeRef.current === null || lastScopeRef.current !== scope) {
      actions.resetChat(scope);
    }
    lastScopeRef.current = scope;
  }, [user, actions]);

  useEffect(() => {
    if (!user) return;
    if (user.profile?.name) {
      profileLoadedRef.current = String(user.id || user.email || "");
      return;
    }
    const key = String(user.id || user.email || "");
    if (!key || profileLoadedRef.current === key) return;
    profileLoadedRef.current = key;
    (async () => {
      try {
        const profile = await apiGetProfile();
        if (!profile) return;
        setUser({ ...user, profile });
      } catch {
        // best-effort profile hydration
      }
    })();
  }, [user, setUser]);

  useEffect(() => {
    if (dark) document.documentElement.classList.add("dark");
    else document.documentElement.classList.remove("dark");
  }, [dark]);
  // Seed theme preference on first load
  useEffect(() => {
    const stored = localStorage.getItem("medibot.dark");
    if (stored === null) {
      const prefersDark = window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
      actions.setDark(prefersDark);
    }
  }, [actions]);

  // Encourage profile completion on first load based on stored profile data
  useEffect(() => {
    if (!user) return;
    if (hideProfileBanner) return;
    const p: any = user.profile || {};
    const missing: string[] = [];
    if (p.age === undefined || p.age === null || p.age === "") missing.push("age");
    if (!p.sex) missing.push("sex");
    if (!Array.isArray(p.conditions) || p.conditions.length === 0) missing.push("conditions");
    if (!Array.isArray(p.medications) || p.medications.length === 0) missing.push("medications");
    if (!missing.length) return;
    const targetId = activeId || conversations[0]?.id;
    if (!targetId) return;
    if ((missingFieldsById[targetId] || []).length === 0) {
      actions.setMissingFieldsFor(targetId, missing);
      actions.setHideProfileBanner(false);
    }
  }, [user, activeId, conversations, missingFieldsById, actions]);

  // Guided first-run: if no conversations or no messages, show a nudge in sidebar
  const firstRun = !active || (active.messages || []).length === 0;

  // Load server-side conversations when the user is present (once per user)
  const conversationsLoadedRef = useRef<string | null>(null);
  useEffect(() => {
    const userKey = user ? String(user.email || user.id || "anon") : null;
    if (!userKey) return;
    if (conversationsLoadedRef.current === userKey) return;
    conversationsLoadedRef.current = userKey;
    (async () => {
      try {
        const serverConvos: ConversationSummary[] = await listConversations();
        const mapped = serverConvos.map((c) => ({
          id: c.id,
          title: c.title || "Chat",
          messages: [],
        }));
        if (mapped.length === 0) {
          const created = await createConversation("New chat");
          actions.setConversations([{ id: created.id, title: created.title || "New chat", messages: [] }]);
          actions.setActiveId(created.id);
        } else {
          actions.setConversations(mapped);
          actions.setActiveId(mapped[0].id);
          try {
            const rows: ConversationMessageRow[] = await getConversationMessages(mapped[0].id, 100);
            const msgs: Message[] = rows.map((r) => ({
              id: r.id,
              role: r.role as any,
              content: r.content,
              ts: new Date(r.created_at),
            }));
            actions.setMessagesFor(mapped[0].id, msgs);
          } catch (e: any) {
            addToast({ type: "error", message: e?.message || "Could not load conversation history." });
          }
        }
      } catch (e: any) {
        addToast({ type: "error", message: e?.message || "Could not load conversations." });
      }
    })();
  }, [user, addToast, actions]);

  // auth bootstrap handled by auth store; no local restore

  async function newConversation() {
    try {
      const created = await createConversation("New chat");
      const conv = { id: created.id, title: created.title || "New chat", messages: [] };
      const next = [conv, ...conversations.filter((c) => c.id !== conv.id)];
      actions.setConversations(next);
      actions.setActiveId(conv.id);
      actions.setStructuredFor(conv.id, null);
      actions.setExplanationFor(conv.id, "");
      actions.setExplanationSourceFor(conv.id, undefined);
      actions.setSymptomAnalysisResult(null);
    } catch (e: any) {
      addToast({ type: "error", message: e?.message || "Unable to start a new chat." });
    }
  }

  async function deleteConversation(id: string) {
    try {
      if (isServerConversationId(id)) {
        await deleteConversationApi(id);
      }
      const wasActive = activeId === id;
      actions.deleteConversation(id);
      if (wasActive) {
        actions.setSymptomAnalysisResult(null);
      }
    } catch (e: any) {
      addToast({ type: "error", message: e?.message || "Could not delete conversation." });
    }
  }

  async function ensureActiveConversation(): Promise<string> {
    if (activeId && isServerConversationId(activeId)) return activeId;
    const created = await createConversation("New chat");
    const conv = { id: created.id, title: created.title || "New chat", messages: [] };
    if (activeId && !isServerConversationId(activeId)) {
      actions.replaceConversationId(activeId, conv.id);
    } else {
      actions.setConversations([conv, ...conversations.filter((c) => c.id !== conv.id)]);
      actions.setActiveId(conv.id);
    }
    return conv.id;
  }

  async function selectConversation(id: string) {
    actions.setActiveId(id);
    if (!isServerConversationId(id)) return;
    const existing = conversations.find((c) => c.id === id);
    if (existing && existing.messages && existing.messages.length > 0) return;
    try {
      const rows: ConversationMessageRow[] = await getConversationMessages(id, 100);
      const mapped: Message[] = rows.map((r) => ({
        id: r.id,
        role: r.role as any,
        content: r.content,
        ts: new Date(r.created_at),
      }));
      actions.setMessagesFor(id, mapped);
    } catch (e: any) {
      addToast({ type: "error", message: e?.message || "Could not load conversation history." });
    }
  }

  async function runChatRequest(
    userText: string,
    requestFn: (conversationId: string) => Promise<ChatResponseCombined>,
    opts: { clearInput?: boolean; imageUrl?: string } = {}
  ) {
    const text = userText.trim();
    if (!text || !user) return;
    if (!user.profile?.consent_given && !consentWarnedRef.current) {
      addToast({ type: "info", message: "For best results, please consent to data processing (see banner above)." });
      consentWarnedRef.current = true;
    }

    const conversationId = activeId || (await ensureActiveConversation());
    actions.setBusy(true);
    setAnalysisState("loading");
    setAnalysisError(null);
    const newUserMessage: Message = {
      id: `m_${Date.now()}`,
      role: "user",
      content: text,
      ts: new Date(),
      imageUrl: opts.imageUrl,
    };
    actions.applyToActive((msgs) => [...msgs, newUserMessage]);
    if (opts.clearInput !== false) {
      actions.setInput("");
    }

    try {
      const chatResponse: ChatResponseCombined = await requestFn(conversationId);
      const targetConversationId = chatResponse.conversation_id || conversationId || activeId || "";
      const replaced =
        Boolean(targetConversationId) && Boolean(activeId) && targetConversationId !== activeId;
      if (targetConversationId && targetConversationId !== activeId) {
        if (activeId) {
          actions.replaceConversationId(activeId, targetConversationId);
        } else {
          actions.setActiveId(targetConversationId);
        }
      }
      if (!replaced && targetConversationId && !conversations.some((c) => c.id === targetConversationId)) {
        actions.setConversations([{ id: targetConversationId, title: "Chat", messages: [] }, ...conversations]);
      }
      if (chatResponse.timed_out) {
        addToast({ type: "info", message: "AI timed out; showing local recommendations instead." });
      } else if (chatResponse.ai_explanation_source === "fallback") {
        addToast({ type: "info", message: "AI explanation unavailable; showing structured recommendations." });
      }
      if (targetConversationId) {
        actions.setStructuredFor(targetConversationId, chatResponse.pipeline || null);
        actions.setExplanationFor(targetConversationId, chatResponse.ai_explanation || "");
        actions.setExplanationSourceFor(targetConversationId, chatResponse.ai_explanation_source as any);
        if ((chatResponse as any).local_recommendations) {
          actions.setRecommendationsFor(targetConversationId, (chatResponse as any).local_recommendations);
        }
        if (Array.isArray(chatResponse.missing_fields)) {
          actions.setMissingFieldsFor(targetConversationId, chatResponse.missing_fields as any);
          actions.setHideProfileBanner(
            hideProfileBanner || (chatResponse.missing_fields || []).length === 0
          );
        }
        if (chatResponse.triage) {
          actions.setTriageFor(targetConversationId, chatResponse.triage as any);
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
      setAnalysisState("idle");
    } catch (err: any) {
      addToast({ type: "error", message: err?.message || "Chat request failed. Please try again." });
      setAnalysisState("error");
      setAnalysisError(err?.message || "Chat request failed");
    } finally {
      actions.setBusy(false);
    }
  }

  async function handleSend() {
    const text = input.trim();
    await runChatRequest(text, (conversationId) => apiChat(text, conversationId));
  }

  async function onFilePicked(file: File) {
    const allowedTypes = new Set(["application/pdf", "image/jpeg", "image/png"]);
    const ext = (file.name || "").toLowerCase().split(".").pop() || "";
    const allowedExts = new Set(["pdf", "jpg", "jpeg", "png"]);
    const hasAllowedType = allowedTypes.has(file.type);
    const hasAllowedExt = allowedExts.has(ext);
    if (!hasAllowedType && !hasAllowedExt) {
      addToast({ type: "error", message: "Unsupported file type. Please upload a PDF or JPG/PNG image." });
      return;
    }
    actions.setFileBusy(true);
    setUploadStatus("loading");
    setUploadError(null);
    try {
      const isPdf = file.type === "application/pdf" || ext === "pdf";
      if (isPdf) {
        const lab = await uploadLabAndSave(file);
        const rawText = lab.raw_text || "";
        if (!rawText.trim()) {
          throw new Error("No text could be extracted from this PDF.");
        }
        addToast({ type: "info", message: "PDF uploaded. Sending for analysis..." });
        await runChatRequest(rawText, (conversationId) => apiChat(rawText, conversationId), { clearInput: false });
      } else {
        addToast({ type: "info", message: "Image uploaded. Analyzing..." });
        const imageUrl = URL.createObjectURL(file);
        await runChatRequest("Uploaded lab image.", (conversationId) => apiChatImage(file, conversationId), { clearInput: false, imageUrl });
      }
      setUploadStatus("success");
      setTimeout(() => setUploadStatus("idle"), 1200);
    } catch (e: any) {
      addToast({ type: "error", message: e?.message || "File upload or parsing failed. Please try again." });
      setUploadStatus("error");
      setUploadError(e?.message || "Upload failed");
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
        setAnalysisState("loading");
        setAnalysisError(null);
        const resp: ChatResponseCombined = await apiChat(lastUser.content, activeId || undefined);
        if (activeId) {
          actions.setStructuredFor(activeId, resp.pipeline || null);
          actions.setExplanationFor(activeId, resp.ai_explanation || "");
          actions.setExplanationSourceFor(activeId, resp.ai_explanation_source as any);
          if (Array.isArray(resp.missing_fields)) {
            actions.setMissingFieldsFor(activeId, resp.missing_fields as any);
          }
          if (resp.triage) actions.setTriageFor(activeId, resp.triage as any);
        }
        setAnalysisState("idle");
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

  const gridCols = analysisOpen
    ? "lg:grid-cols-[300px_minmax(0,1fr)_360px]"
    : "lg:grid-cols-[300px_minmax(0,1fr)]";

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

      <div className={`h-[calc(100vh-0px)] grid grid-cols-1 ${gridCols}`}>
        <div
          className={`min-h-0 h-full ${sidebarOpen ? "block" : "hidden"} lg:block`}
        >
          <ConversationList
            conversations={conversations}
            activeId={activeId}
            onSelect={(id) => selectConversation(id)}
            onNew={newConversation}
            onDelete={deleteConversation}
          onToggleAnalysis={() => setAnalysisOpen((v) => !v)}
          analysisOpen={analysisOpen}
          dark={dark}
          onToggleDark={() => actions.setDark(!dark)}
          user={user}
          onProfile={() => actions.setProfileOpen(true)}
          onLogout={() => logout()}
          firstRun={firstRun}
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
          triage={triage}
          urgentAck={urgentAckById[activeId || ""]}
          onAckUrgent={() => actions.setUrgentAckFor(activeId || "", true)}
          missingFields={missingFieldsById[activeId || ""] || []}
          hideProfileBanner={hideProfileBanner}
          onOpenProfile={() => actions.setProfileOpen(true)}
          onHideProfileBanner={() => actions.setHideProfileBanner(true)}
          uploadInputRef={uploadInputRef}
          onUploadClick={() => uploadInputRef.current?.click()}
          onFilePicked={onFilePicked}
          uploadStatus={uploadStatus}
          uploadError={uploadError}
          onClearUploadError={() => setUploadError(null)}
          analysisState={analysisState}
          analysisError={analysisError}
      messages={active?.messages || []}
      onSend={handleSend}
      onKeyDown={onKeyDown}
      input={input}
      setInput={actions.setInput}
      busy={busy}
      fileBusy={fileBusy}
      needsConsent={!user.profile?.consent_given}
      consentBusy={consentBusy}
      onConsent={async () => {
        if (consentBusy) return;
        try {
          setConsentBusy(true);
          const res = await setConsent(true);
          setUser({ ...user, profile: { ...(user.profile || {}), consent_given: res.consent_given, consent_at: res.consent_at } });
          addToast({ type: "info", message: "Consent recorded. You can withdraw anytime from Profile." });
        } catch (e: any) {
          addToast({ type: "error", message: e?.message || "Could not record consent." });
        } finally {
          setConsentBusy(false);
        }
      }}
    />

        {analysisOpen && (
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
          recommendations={recommendations as any}
          analysisState={analysisState}
          analysisError={analysisError}
        />
      )}
      </div>
    </div>
  );
}

export default function ClassicChatbotUI() {
  return <ChatView />;
}
