import React, { useEffect, useMemo, useRef, useState } from "react";
import DOMPurify from "dompurify";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import rehypeSanitize from "rehype-sanitize";
import { Bot, User, Copy, Check, Volume2, VolumeX } from "lucide-react";
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

function stripMarkdown(value: string): string {
  return value
    .replace(/```[\s\S]*?```/g, "")
    .replace(/`([^`]+)`/g, "$1")
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, "$1")
    .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
    .replace(/^\s{0,3}#{1,6}\s+/gm, "")
    .replace(/^\s*>\s?/gm, "")
    .replace(/^\s*[-*+]\s+/gm, "")
    .replace(/^\s*\d+\.\s+/gm, "")
    .replace(/^\s*[-*_]{3,}\s*$/gm, "")
    .replace(/(\*\*|__)(.*?)\1/g, "$2")
    .replace(/(\*|_)(.*?)\1/g, "$2")
    .replace(/~~(.*?)~~/g, "$1")
    .replace(/\r\n/g, "\n")
    .trim();
}

export default function ChatMessage({ msg, devMode }: ChatMessageProps) {
  const isUser = msg.role === "user";
  const [copied, setCopied] = useState(false);
  const [speaking, setSpeaking] = useState(false);
  const utteranceRef = useRef<SpeechSynthesisUtterance | null>(null);
  const safe = useMemo(
    () => DOMPurify.sanitize(msg.content, { ALLOWED_TAGS: [], ALLOWED_ATTR: [] }),
    [msg.content]
  );
  const plainText = useMemo(() => stripMarkdown(msg.content || ""), [msg.content]);
  const uv = (msg.rawResponse as any)?.user_view as any | undefined;
  const hasUserView = !!uv && (Array.isArray(uv.abnormal) || Array.isArray(uv.normal));
  const canSpeak =
    !isUser &&
    typeof window !== "undefined" &&
    "speechSynthesis" in window &&
    plainText.trim().length > 0;

  useEffect(() => {
    return () => {
      if (typeof window !== "undefined" && "speechSynthesis" in window) {
        window.speechSynthesis.cancel();
      }
    };
  }, []);

  async function handleCopy() {
    try {
      if (navigator.clipboard && navigator.clipboard.writeText) {
        await navigator.clipboard.writeText(plainText);
      } else {
        const area = document.createElement("textarea");
        area.value = plainText;
        area.style.position = "fixed";
        area.style.opacity = "0";
        document.body.appendChild(area);
        area.select();
        document.execCommand("copy");
        document.body.removeChild(area);
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {
      // ignore clipboard errors
    }
  }

  function stopSpeech() {
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    window.speechSynthesis.cancel();
    utteranceRef.current = null;
    setSpeaking(false);
  }

  function handleSpeakToggle() {
    if (!canSpeak) return;
    if (speaking) {
      stopSpeech();
      return;
    }
    if (typeof window === "undefined" || !("speechSynthesis" in window)) return;
    const utterance = new SpeechSynthesisUtterance(plainText);
    utterance.onend = () => setSpeaking(false);
    utterance.onerror = () => setSpeaking(false);
    utteranceRef.current = utterance;
    setSpeaking(true);
    window.speechSynthesis.speak(utterance);
  }

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
          <div className="prose prose-sm dark:prose-invert max-w-none">
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              rehypePlugins={[rehypeSanitize]}
              components={{
                p: ({ children }) => <p className="mb-2 last:mb-0">{children}</p>,
                ul: ({ children }) => <ul className="list-disc pl-5 mb-2 last:mb-0">{children}</ul>,
                ol: ({ children }) => <ol className="list-decimal pl-5 mb-2 last:mb-0">{children}</ol>,
                li: ({ children }) => <li className="mb-1 last:mb-0">{children}</li>,
                a: ({ children, href }) => (
                  <a className="text-blue-600 dark:text-blue-400 underline" href={href} target="_blank" rel="noreferrer">
                    {children}
                  </a>
                ),
                code: ({ children }) => (
                  <code className="px-1 py-0.5 rounded bg-zinc-100 dark:bg-zinc-800 text-[12px]">{children}</code>
                ),
                pre: ({ children }) => (
                  <pre className="p-2 rounded bg-zinc-100 dark:bg-zinc-800 overflow-x-auto text-[12px]">{children}</pre>
                ),
                blockquote: ({ children }) => (
                  <blockquote className="border-l-2 border-zinc-300 dark:border-zinc-700 pl-3 italic text-zinc-600 dark:text-zinc-300">
                    {children}
                  </blockquote>
                ),
              }}
            >
              {msg.content || ""}
            </ReactMarkdown>
          </div>
        )}
        {!isUser && (
          <div className="mt-2 flex items-center justify-between gap-2">
            <div className="text-[10px] text-zinc-400">{formatTime(msg.ts)}</div>
            <div className="flex items-center gap-1">
              <button
                type="button"
                className="p-1 rounded-md text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                onClick={handleCopy}
                aria-label="Copy assistant response"
                title="Copy"
              >
                {copied ? <Check className="w-3.5 h-3.5" /> : <Copy className="w-3.5 h-3.5" />}
              </button>
              <button
                type="button"
                className="p-1 rounded-md text-zinc-500 hover:text-zinc-800 hover:bg-zinc-100 dark:hover:bg-zinc-800 disabled:opacity-50"
                onClick={handleSpeakToggle}
                disabled={!canSpeak}
                aria-label={speaking ? "Stop voice playback" : "Play response as audio"}
                title={speaking ? "Stop voice" : "Play voice"}
              >
                {speaking ? <VolumeX className="w-3.5 h-3.5" /> : <Volume2 className="w-3.5 h-3.5" />}
              </button>
              {devMode && (
                <span className="ml-2 text-[10px] text-zinc-400">
                  {msg.requestId ? `req: ${msg.requestId}` : ""}
                  {msg.requestId && msg.aiSource ? " | " : ""}
                  {msg.aiSource ? `source: ${msg.aiSource}` : ""}
                </span>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
