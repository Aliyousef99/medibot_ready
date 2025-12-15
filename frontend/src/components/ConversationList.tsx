import React from "react";
import { MessageSquare, Trash2, Sun, Moon, PanelRightOpen, PanelLeftOpen } from "lucide-react";
import UserMenu from "./UserMenu";
import type { Conversation, User } from "../types";

type ConversationListProps = {
  conversations: Conversation[];
  activeId: string | null;
  onSelect: (id: string) => void;
  onNew: () => void;
  onDelete: (id: string) => void;
  onToggleAnalysis?: () => void;
  analysisOpen?: boolean;
  dark: boolean;
  onToggleDark: () => void;
  user: User | null;
  onProfile: () => void;
  onLogout: () => void;
  firstRun?: boolean;
};

export default function ConversationList({
  conversations,
  activeId,
  onSelect,
  onNew,
  onDelete,
  onToggleAnalysis,
  analysisOpen,
  dark,
  onToggleDark,
  user,
  onProfile,
  onLogout,
  firstRun = false,
}: ConversationListProps) {
  return (
    <aside className="min-h-0 h-full border-r border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950">
      <div className="h-full flex flex-col min-h-0">
        <div
          className="shrink-0 px-3 py-2 border-b border-zinc-200 dark:border-zinc-800
                              bg-white/80 dark:bg-zinc-950/80 backdrop-blur"
        >
          <div className="flex items-center gap-2">
            <button
              onClick={onNew}
              aria-label="Start a new chat"
              className="inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium
                              bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90"
            >
              New chat
            </button>
            <div className="ml-auto" />
            {onToggleAnalysis && (
              <button
                onClick={onToggleAnalysis}
                className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800"
                title={analysisOpen ? "Hide analysis" : "Show analysis"}
                aria-label={analysisOpen ? "Hide analysis" : "Show analysis"}
              >
                {analysisOpen ? <PanelRightOpen className="w-5 h-5" /> : <PanelLeftOpen className="w-5 h-5" />}
              </button>
            )}
            <button
              onClick={onToggleDark}
              className="p-2 rounded-xl hover:bg-zinc-100 dark:hover:bg-zinc-800"
              aria-label="Toggle theme"
            >
              {dark ? <Sun className="w-5 h-5" /> : <Moon className="w-5 h-5" />}
            </button>
          </div>
        </div>

        <div className="flex-1 min-h-0 overflow-y-auto overscroll-contain p-2 space-y-2">
          {conversations.length ? (
            conversations.map((c) => (
              <button
                key={c.id}
                onClick={() => onSelect(c.id)}
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
                    onDelete(c.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 p-1 rounded-lg hover:bg-zinc-200 dark:hover:bg-zinc-700"
                >
                  <Trash2 className="w-4 h-4" />
                </span>
              </button>
            ))
          ) : (
            <div className="rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700 p-4 text-xs text-zinc-500 text-center">
              No conversations yet.
              <br />
              <button
                onClick={onNew}
                className="mt-2 inline-flex items-center justify-center rounded-md bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 px-3 py-1.5 text-xs font-semibold hover:opacity-90"
              >
                Start a chat
              </button>
              {firstRun && (
                <div className="mt-3 space-y-1 text-xs text-emerald-700 dark:text-emerald-300">
                  <div>1) Complete your profile (age/sex/conditions/meds) for personalization.</div>
                  <div>2) Upload a lab report (PDF/Image/Text) to structure your labs.</div>
                  <div>3) Ask your question to get recommendations.</div>
                </div>
              )}
            </div>
          )}
        </div>

        <div className="shrink-0 border-t border-zinc-200 dark:border-zinc-800 p-3">
          <UserMenu user={{ email: user?.email || "" }} onProfile={onProfile} onLogout={onLogout} />
        </div>
      </div>
    </aside>
  );
}
