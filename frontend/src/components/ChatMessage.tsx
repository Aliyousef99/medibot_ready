import React, { useMemo } from "react";
import DOMPurify from "dompurify";
import { Bot, User } from "lucide-react";
import LabSummaryCard from "./LabSummaryCard";
import type { Message } from "../types";

function formatTime(d: Date) {
  try {
    return new Intl.DateTimeFormat(undefined, { hour: "2-digit", minute: "2-digit" }).format(d);
  } catch {
    return d.toLocaleTimeString();
  }
}

type ChatMessageProps = {
  msg: Message;
  devMode?: boolean;
};

export default function ChatMessage({ msg, devMode }: ChatMessageProps) {
  const isUser = msg.role === "user";
  const safe = useMemo(
    () => DOMPurify.sanitize(msg.content, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] }),
    [msg.content]
  );
  const uv = (msg.rawResponse as any)?.user_view as any | undefined;
  const hasUserView = !!uv && (Array.isArray(uv.abnormal) || Array.isArray(uv.normal));

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
        {isUser ? (
          <div>{safe}</div>
        ) : hasUserView ? (
          <LabSummaryCard response={msg.rawResponse} aiText={msg.content} devMode={!!devMode} />
        ) : (
          <div>{safe}</div>
        )}
        {!isUser && devMode && (
          <div className="mt-2 text-[10px] text-zinc-400 flex items-center justify-between">
            <span>{formatTime(msg.ts)}</span>
            <span className="ml-2">
              {msg.requestId ? `req: ${msg.requestId}` : ""}
              {msg.requestId && msg.aiSource ? " | " : ""}
              {msg.aiSource ? `source: ${msg.aiSource}` : ""}
            </span>
          </div>
        )}
      </div>
    </div>
  );
}
