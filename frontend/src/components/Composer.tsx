import React, { RefObject, useEffect, useRef, useState } from "react";
import { Loader2, Mic, MicOff, Plus, Send } from "lucide-react";

// Minimal SpeechRecognition typings to keep Web Speech API usage type-safe without external defs
type SpeechRecognition = {
  start: () => void;
  stop: () => void;
  abort: () => void;
  continuous: boolean;
  interimResults: boolean;
  lang: string;
  onresult: ((event: SpeechRecognitionEvent) => void) | null;
  onstart: (() => void) | null;
  onend: (() => void) | null;
  onerror: ((event: SpeechRecognitionErrorEvent) => void) | null;
};

type SpeechRecognitionEvent = {
  resultIndex: number;
  results: SpeechRecognitionResultList;
};

type SpeechRecognitionResultList = {
  length: number;
  [index: number]: SpeechRecognitionResult;
};

type SpeechRecognitionResult = {
  isFinal: boolean;
  [index: number]: SpeechRecognitionAlternative;
};

type SpeechRecognitionAlternative = {
  transcript: string;
};

type SpeechRecognitionErrorEvent = {
  error: string;
};

type ComposerProps = {
  input: string;
  onInputChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  uploadInputRef: RefObject<HTMLInputElement>;
  onFilePicked: (files: FileList) => void;
  attachments?: { id: string; name: string; kind: "image" | "pdf"; previewUrl?: string }[];
  onRemoveAttachment?: (id: string) => void;
  busy: boolean;
  fileBusy: boolean;
};

export default function Composer({
  input,
  onInputChange,
  onKeyDown,
  onSend,
  uploadInputRef,
  onFilePicked,
  attachments = [],
  onRemoveAttachment,
  busy,
  fileBusy,
}: ComposerProps) {
  const recognitionRef = useRef<SpeechRecognition | null>(null);
  const listeningBaseRef = useRef<string>("");
  const latestInputRef = useRef<string>(input);
  const textareaRef = useRef<HTMLTextAreaElement | null>(null);
  const [listening, setListening] = useState(false);
  const [voiceSupported, setVoiceSupported] = useState(true);
  const [voiceError, setVoiceError] = useState<string | null>(null);

  const appendWithSpace = (base: string, addition: string) => {
    if (!addition) return base;
    const spacer = base.trim().length === 0 || base.endsWith(" ") ? "" : " ";
    return `${base}${spacer}${addition}`.replace(/\s+/g, " ");
  };

  useEffect(() => {
    latestInputRef.current = input;
  }, [input]);

  useEffect(() => {
    const el = textareaRef.current;
    if (!el || typeof window === "undefined") return;
    el.style.height = "auto";
    const styles = window.getComputedStyle(el);
    const fontSize = Number.parseFloat(styles.fontSize || "0");
    const rawLineHeight = Number.parseFloat(styles.lineHeight || "");
    const lineHeight = Number.isFinite(rawLineHeight) && rawLineHeight > 0 ? rawLineHeight : fontSize * 1.4;
    const paddingTop = Number.parseFloat(styles.paddingTop || "0");
    const paddingBottom = Number.parseFloat(styles.paddingBottom || "0");
    const maxLines = 10;
    const maxHeight = Math.ceil(lineHeight * maxLines + paddingTop + paddingBottom);
    const nextHeight = Math.min(el.scrollHeight, maxHeight);
    el.style.height = `${nextHeight}px`;
    el.style.overflowY = el.scrollHeight > maxHeight ? "auto" : "hidden";
  }, [input]);

  useEffect(() => {
    if (typeof window === "undefined") return;
    const SpeechRecognitionClass: any =
      (window as any).SpeechRecognition || (window as any).webkitSpeechRecognition;
    if (!SpeechRecognitionClass) {
      setVoiceSupported(false);
      return;
    }
    const rec: SpeechRecognition = new SpeechRecognitionClass();
    rec.continuous = true;
    rec.interimResults = true;
    rec.lang = "en-US";

    rec.onresult = (event: SpeechRecognitionEvent) => {
      let interim = "";
      let finalChunk = "";
      for (let i = event.resultIndex; i < event.results.length; i++) {
        const res = event.results[i];
        const transcript = res[0]?.transcript || "";
        if (res.isFinal) finalChunk += transcript;
        else interim += transcript;
      }
      if (finalChunk.trim()) {
        listeningBaseRef.current = appendWithSpace(listeningBaseRef.current, finalChunk.trim());
      }
      const combined = appendWithSpace(listeningBaseRef.current || latestInputRef.current, interim.trim());
      onInputChange(combined);
    };

    rec.onstart = () => {
      setListening(true);
      setVoiceError(null);
    };
    rec.onend = () => {
      setListening(false);
      listeningBaseRef.current = "";
    };
    rec.onerror = (event: SpeechRecognitionErrorEvent) => {
      setListening(false);
      const code = event.error;
      if (code === "not-allowed" || code === "service-not-allowed") {
        setVoiceError("Microphone permission denied. Please allow mic access.");
      } else {
        setVoiceError("Voice input unavailable right now. Try again or type instead.");
      }
    };

    recognitionRef.current = rec;
    return () => {
      try {
        rec.stop();
      } catch (e) {
        // ignore cleanup errors
      }
    };
  }, [onInputChange]);

  const toggleListening = () => {
    const rec = recognitionRef.current;
    if (!rec || !voiceSupported) {
      setVoiceError("Voice input is not supported in this browser.");
      return;
    }
    if (listening) {
      rec.stop();
      return;
    }
    listeningBaseRef.current = latestInputRef.current;
    setVoiceError(null);
    try {
      rec.start();
    } catch (e: any) {
      setListening(false);
      const msg =
        (e?.message as string) ||
        "Could not start microphone. Check permissions and try again.";
      setVoiceError(msg);
    }
  };

  return (
    <div className="px-4 lg:px-6 py-3 bg-gradient-to-t from-zinc-50 via-zinc-50/90 to-transparent dark:from-zinc-900 dark:via-zinc-900/90">
      <div className="mx-auto max-w-3xl">
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-sm p-2">
          {attachments.length > 0 && (
            <div className="mb-2 flex flex-wrap items-center gap-3 rounded-xl border border-zinc-200 dark:border-zinc-800 bg-zinc-50/70 dark:bg-zinc-900/60 px-3 py-2">
              {attachments.map((attachment) => (
                <div key={attachment.id} className="flex items-center gap-2">
                  {attachment.kind === "image" && attachment.previewUrl ? (
                    <img
                      src={attachment.previewUrl}
                      alt={attachment.name}
                      className="h-10 w-10 rounded-lg object-cover border border-zinc-200 dark:border-zinc-800"
                    />
                  ) : (
                    <div className="h-10 w-10 rounded-lg bg-zinc-200 dark:bg-zinc-800 flex items-center justify-center text-[10px] font-semibold text-zinc-600 dark:text-zinc-300">
                      PDF
                    </div>
                  )}
                  <div className="max-w-[140px] text-xs text-zinc-700 dark:text-zinc-200 truncate">
                    {attachment.name}
                  </div>
                  {onRemoveAttachment && (
                    <button
                      type="button"
                      onClick={() => onRemoveAttachment(attachment.id)}
                      className="text-[11px] px-2 py-1 rounded-lg border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                      aria-label={`Remove ${attachment.name}`}
                    >
                      Remove
                    </button>
                  )}
                </div>
              ))}
            </div>
          )}
          <div className="flex items-center gap-2">
            <label
              className={`shrink-0 inline-flex items-center justify-center rounded-xl p-2 hover:bg-zinc-100 dark:hover:bg-zinc-800 cursor-pointer ${
                fileBusy ? "animate-pulse" : ""
              } focus-within:outline-none focus-within:ring-2 focus-within:ring-offset-2 focus-within:ring-zinc-400`}
              title="Upload (PDF/Image/Text)"
              aria-label="Upload file"
            >
              <input
                type="file"
                className="hidden"
                accept=".pdf,.jpg,.jpeg,.png"
                multiple
                ref={uploadInputRef}
                onChange={(e) => e.target.files && onFilePicked(e.target.files)}
                disabled={fileBusy}
              />
              {fileBusy ? <Loader2 className="w-5 h-5 animate-spin" /> : <Plus className="w-5 h-5" />}
            </label>

            <textarea
              ref={textareaRef}
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={fileBusy ? "Extracting text from file..." : "Paste lab text or type your question..."}
              className="flex-1 min-h-[3rem] resize-none bg-transparent px-2 py-2 outline-none placeholder:text-zinc-400 text-sm"
            />

            <button
              type="button"
              onClick={toggleListening}
              className={`shrink-0 inline-flex items-center justify-center rounded-xl px-2 py-2 transition ${
                listening
                  ? "bg-red-100 text-red-700 ring-2 ring-red-400 animate-pulse"
                  : "hover:bg-zinc-100 dark:hover:bg-zinc-800"
              } ${!voiceSupported ? "opacity-60 cursor-not-allowed" : ""}`}
              title={listening ? "Stop voice input" : "Start voice input"}
              aria-label={listening ? "Stop voice input" : "Start voice input"}
              aria-pressed={listening}
              disabled={!voiceSupported || fileBusy}
            >
              {listening ? <MicOff className="w-5 h-5" /> : <Mic className="w-5 h-5" />}
            </button>

            <button
              onClick={onSend}
              className="shrink-0 inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90 disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-zinc-400"
              disabled={(!input.trim() && attachments.length === 0) || busy || fileBusy}
              aria-label="Send"
            >
              {busy ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" /> Working...
                </>
              ) : (
                <>
                  <Send className="w-4 h-4" /> Send
                </>
              )}
            </button>
          </div>
          {voiceError && (
            <div className="mt-2 text-xs text-red-600 dark:text-red-400" role="status" aria-live="polite">
              {voiceError}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
