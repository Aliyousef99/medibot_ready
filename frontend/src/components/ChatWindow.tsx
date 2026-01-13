import React, { useEffect, useRef } from "react";
import { ShieldAlert, MessageSquare, AlertTriangle, CheckCircle2 } from "lucide-react";
import ChatMessage from "./ChatMessage";
import Composer from "./Composer";
import type { Conversation, Message } from "../types";

type ChatWindowProps = {
  active: Conversation | null;
  devMode: boolean;
  triage?: { level: string; reasons?: string[]; suggested_window?: string };
  urgentAck?: boolean;
  onAckUrgent: () => void;
  needsConsent?: boolean;
  consentBusy?: boolean;
  onConsent?: () => void;
  missingFields?: string[];
  hideProfileBanner: boolean;
  onOpenProfile: () => void;
  onHideProfileBanner: () => void;
  uploadInputRef: React.RefObject<HTMLInputElement>;
  onUploadClick: () => void;
  onFilePicked: (file: File) => void;
  uploadStatus: "idle" | "loading" | "success" | "error";
  uploadError?: string | null;
  onClearUploadError?: () => void;
  analysisState?: "idle" | "loading" | "error";
  analysisError?: string | null;
  messages: Message[];
  onSend: () => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  input: string;
  setInput: (v: string) => void;
  busy: boolean;
  fileBusy: boolean;
};

export default function ChatWindow({
  active,
  devMode,
  triage,
  urgentAck,
  onAckUrgent,
  needsConsent = false,
  consentBusy = false,
  onConsent,
  missingFields = [],
  hideProfileBanner,
  onOpenProfile,
  onHideProfileBanner,
  uploadInputRef,
  onUploadClick,
  onFilePicked,
  uploadStatus,
  uploadError,
  onClearUploadError,
  analysisState = "idle",
  analysisError,
  messages,
  onSend,
  onKeyDown,
  input,
  setInput,
  busy,
  fileBusy,
}: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);
  const lastActiveIdRef = useRef<string | null>(null);
  const skipNextScrollRef = useRef(false);

  useEffect(() => {
    const activeId = active?.id || null;
    if (lastActiveIdRef.current !== activeId) {
      lastActiveIdRef.current = activeId;
      skipNextScrollRef.current = true;
    }
    const el = scrollRef.current as any;
    if (!el) return;
    if (skipNextScrollRef.current) {
      skipNextScrollRef.current = false;
      return;
    }
    if (typeof el.scrollTo === "function") {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    } else {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages.length, active?.id]);

  function triageBadge(level?: string) {
    const lv = (level || "").toLowerCase();
    if (lv === "urgent" || lv === "high") {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-200">
          <ShieldAlert className="w-3.5 h-3.5" /> Urgent
        </span>
      );
    }
    if (lv === "moderate" || lv === "watch") {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-200">
          <AlertTriangle className="w-3.5 h-3.5" /> Watch
        </span>
      );
    }
    if (lv) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-1 rounded-full text-xs bg-emerald-100 text-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-200">
          <CheckCircle2 className="w-3.5 h-3.5" /> {lv}
        </span>
      );
    }
    return null;
  }

  return (
    <main className="relative flex flex-col min-h-0">
      <div className="hidden lg:flex items-center justify-between px-5 py-3 border-b border-zinc-200 dark:border-zinc-800">
        <div className="font-semibold tracking-tight">{active?.title || "Chat"}</div>
        {triageBadge(triage?.level)}
      </div>

      {triage?.level === "urgent" && !urgentAck && (
        <div className="mx-4 mt-3 rounded-xl border border-red-300 bg-red-50 text-red-900 dark:border-red-700 dark:bg-red-900/20 dark:text-red-200 p-3 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5" />
          <div className="flex-1 text-sm">
            <div className="font-semibold mb-0.5">Seek care now</div>
            <div className="text-[13px] opacity-90">
              {(triage?.reasons || []).length ? `Red flags detected: ${(triage?.reasons || []).join(", ")}` : "Red flags detected."}
            </div>
            {triage?.suggested_window && (
              <div className="text-[12px] opacity-75">Recommended timing: {triage.suggested_window}</div>
            )}
          </div>
          <div className="flex items-center gap-2">
            <button
              className="text-xs px-2 py-1 rounded-lg border border-red-300 dark:border-red-700 hover:bg-red-100 dark:hover:bg-red-900/30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-red-400"
              onClick={onAckUrgent}
              aria-label="Acknowledge urgent triage"
            >
              Okay, understood
            </button>
          </div>
        </div>
      )}
      {triage?.level === "urgent" && urgentAck && (
        <div className="mx-4 mt-3">
          <span
            className="inline-flex items-center gap-1 text-[10px] px-2 py-0.5 rounded-full bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200"
            role="status"
            aria-label="Urgent triage detected"
            title={(triage?.reasons || []).join(", ")}
          >
            <ShieldAlert className="w-3 h-3" /> Urgent triage
          </span>
        </div>
      )}
      {needsConsent && (
        <div className="mx-4 mt-3 rounded-xl border border-emerald-300 bg-emerald-50 text-emerald-900 dark:border-emerald-700 dark:bg-emerald-900/20 dark:text-emerald-200 p-3 flex items-center gap-3">
          <ShieldAlert className="w-5 h-5" />
          <div className="flex-1 text-sm">
            <div className="font-medium mb-0.5">Consent needed</div>
            <div className="text-[12px] opacity-90">
              Please consent to storing and processing your health data. You can still view the app, but chat results are best when consented.
            </div>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="text-xs px-3 py-1.5 rounded-lg border border-emerald-300 dark:border-emerald-700 hover:bg-emerald-100 dark:hover:bg-emerald-900/30 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-400 disabled:opacity-60"
              onClick={onConsent}
              disabled={consentBusy}
            >
              {consentBusy ? "Saving..." : "I consent"}
            </button>
            <button
              className="text-xs px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-emerald-400"
              onClick={onOpenProfile}
            >
              Review settings
            </button>
          </div>
        </div>
      )}
      {uploadStatus === "loading" && (
        <div className="mx-4 mt-3 rounded-xl border border-blue-200 bg-blue-50 text-blue-900 dark:border-blue-700 dark:bg-blue-900/30 dark:text-blue-100 p-3 flex items-center gap-2 text-sm">
          <ShieldAlert className="w-4 h-4" />
          Uploading and parsing your lab... you can keep typing while we process it.
        </div>
      )}
      {uploadStatus === "error" && (
        <div className="mx-4 mt-3 rounded-xl border border-red-200 bg-red-50 text-red-900 dark:border-red-700 dark:bg-red-900/30 dark:text-red-100 p-3 flex items-center gap-3 text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span className="flex-1">Upload failed: {uploadError || "Please try again."}</span>
          <div className="flex items-center gap-2">
            <button
              className="text-xs px-2 py-1 rounded-lg border border-red-200 dark:border-red-600 hover:bg-red-100 dark:hover:bg-red-900/50"
              onClick={onUploadClick}
            >
              Retry upload
            </button>
            {onClearUploadError && (
              <button
                className="text-xs px-2 py-1 rounded-lg border border-zinc-200 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                onClick={onClearUploadError}
              >
                Dismiss
              </button>
            )}
          </div>
        </div>
      )}
      {uploadStatus === "success" && (
        <div className="mx-4 mt-3 rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-100 p-3 flex items-center gap-2 text-sm">
          <CheckCircle2 className="w-4 h-4" />
          Lab uploaded. We sent it for analysis.
        </div>
      )}
      {analysisState === "loading" && (
        <div className="mx-4 mt-3 rounded-xl border border-emerald-200 bg-emerald-50 text-emerald-900 dark:border-emerald-700 dark:bg-emerald-900/30 dark:text-emerald-100 p-3 flex items-center gap-2 text-sm">
          <CheckCircle2 className="w-4 h-4" />
          Generating recommendations and triage...
        </div>
      )}
      {analysisState === "error" && (
        <div className="mx-4 mt-3 rounded-xl border border-amber-200 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-900/30 dark:text-amber-100 p-3 flex items-center gap-3 text-sm">
          <AlertTriangle className="w-4 h-4" />
          <span className="flex-1">Could not generate recommendations. {analysisError || "Please try again."}</span>
          <button
            className="text-xs px-2 py-1 rounded-lg border border-amber-200 dark:border-amber-700 hover:bg-amber-100 dark:hover:bg-amber-900/50"
            onClick={onSend}
          >
            Retry
          </button>
        </div>
      )}
      {!hideProfileBanner && Array.isArray(missingFields) && (missingFields?.length || 0) > 0 && (
        <div className="mx-4 mt-3 rounded-xl border border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-200 p-3 flex items-center gap-3">
          <ShieldAlert className="w-5 h-5" />
          <div className="flex-1 text-sm">
            <div className="font-medium mb-0.5">Profile optional. For personalization, add: {missingFields.join(", ")}.</div>
            <div className="text-[12px] opacity-80">You can keep chatting even without completing the profile.</div>
          </div>
          <div className="flex items-center gap-2">
            <button
              className="text-xs px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-400"
              onClick={onOpenProfile}
              aria-label="Open profile settings"
            >
              Open Profile
            </button>
            <button
              className="text-xs px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-400"
              onClick={onUploadClick}
              aria-label="Upload latest lab report"
            >
              Upload Lab
            </button>
            <button
              className="text-xs px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-400"
              onClick={onHideProfileBanner}
              aria-label="Continue chatting without profile"
            >
              Continue chatting
            </button>
            <button
              className="text-xs px-2 py-1 rounded-lg text-zinc-500 hover:bg-zinc-100 dark:hover:bg-zinc-800 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-amber-400"
              onClick={onHideProfileBanner}
              aria-label="Hide profile completion banner"
            >
              Don't show again
            </button>
          </div>
        </div>
      )}
      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto p-4 lg:p-6 space-y-4">
        {messages.length ? messages.map((m) => <ChatMessage key={m.id} msg={m} devMode={devMode} />) : <EmptyState />}
      </div>

      <Composer
        input={input}
        onInputChange={setInput}
        onKeyDown={onKeyDown}
        onSend={onSend}
        uploadInputRef={uploadInputRef}
        onFilePicked={onFilePicked}
        busy={busy}
        fileBusy={fileBusy}
      />
    </main>
  );
}

function EmptyState() {
  return (
    <div className="h-full flex items-center justify-center px-4">
      <div className="text-center max-w-md space-y-3">
        <div className="mx-auto mb-4 w-12 h-12 rounded-2xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center">
          <MessageSquare className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-semibold">Ask anything</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Upload a file (PDF/Image) or paste lab text, then hit Send. We'll structure entities with heuristics/BioBERT and explain with Gemini.
        </p>
        <div className="text-left text-xs text-zinc-600 dark:text-zinc-300 bg-zinc-100 dark:bg-zinc-800 rounded-xl p-3 space-y-1">
          <div className="font-semibold text-zinc-700 dark:text-zinc-100 text-center mb-1">First-time flow</div>
          <div>1) Open profile and add age / sex / conditions / meds (optional but improves recs).</div>
          <div>2) Upload a lab report to prefill the composer with extracted text.</div>
          <div>3) Ask your question while we parse and generate recommendations.</div>
        </div>
      </div>
    </div>
  );
}
