import React, { useEffect, useRef } from "react";
import { ShieldAlert, MessageSquare } from "lucide-react";
import ChatMessage from "./ChatMessage";
import Composer from "./Composer";
import type { Conversation, Message } from "../types";

type ChatWindowProps = {
  active: Conversation | null;
  devMode: boolean;
  triage?: { level: string; reasons?: string[] };
  urgentAck?: boolean;
  onAckUrgent: () => void;
  missingFields?: string[];
  hideProfileBanner: boolean;
  onOpenProfile: () => void;
  onHideProfileBanner: () => void;
  uploadInputRef: React.RefObject<HTMLInputElement>;
  onUploadClick: () => void;
  onFilePicked: (file: File) => void;
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
  missingFields = [],
  hideProfileBanner,
  onOpenProfile,
  onHideProfileBanner,
  uploadInputRef,
  onUploadClick,
  onFilePicked,
  messages,
  onSend,
  onKeyDown,
  input,
  setInput,
  busy,
  fileBusy,
}: ChatWindowProps) {
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = scrollRef.current as any;
    if (!el) return;
    if (typeof el.scrollTo === "function") {
      el.scrollTo({ top: el.scrollHeight, behavior: "smooth" });
    } else {
      el.scrollTop = el.scrollHeight;
    }
  }, [messages.length]);

  return (
    <main className="relative flex flex-col min-h-0">
      <div className="hidden lg:flex items-center justify-between px-5 py-3 border-b border-zinc-200 dark:border-zinc-800">
        <div className="font-semibold tracking-tight">{active?.title || "Chat"}</div>
      </div>

      {triage?.level === "urgent" && !urgentAck && (
        <div className="mx-4 mt-3 rounded-xl border border-red-300 bg-red-50 text-red-900 dark:border-red-700 dark:bg-red-900/20 dark:text-red-200 p-3 flex items-start gap-3">
          <ShieldAlert className="w-5 h-5" />
          <div className="flex-1 text-sm">
            <div className="font-semibold mb-0.5">Seek care now</div>
            <div className="text-[13px] opacity-90">
              {(triage?.reasons || []).length ? `Red flags detected: ${(triage?.reasons || []).join(", ")}` : "Red flags detected."}
            </div>
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
      {!hideProfileBanner && Array.isArray(missingFields) && (missingFields?.length || 0) > 0 && (
        <div className="mx-4 mt-3 rounded-xl border border-amber-300 bg-amber-50 text-amber-900 dark:border-amber-700 dark:bg-amber-900/20 dark:text-amber-200 p-3 flex items-center gap-3">
          <ShieldAlert className="w-5 h-5" />
          <div className="flex-1 text-sm">
            <div className="font-medium mb-0.5">To personalize guidance, add: {missingFields.join(", ")}.</div>
            <div className="text-[12px] opacity-80">This banner appears once per session.</div>
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
    <div className="h-full flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="mx-auto mb-4 w-12 h-12 rounded-2xl bg-zinc-100 dark:bg-zinc-800 flex items-center justify-center">
          <MessageSquare className="w-6 h-6" />
        </div>
        <h2 className="text-lg font-semibold">Ask anything</h2>
        <p className="text-sm text-zinc-500 mt-1">
          Upload a file (PDF/Image) or paste lab text, then hit Send. We'll structure entities with heuristics/BioBERT and explain
          with Gemini.
        </p>
      </div>
    </div>
  );
}
