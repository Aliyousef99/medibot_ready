import React, { RefObject } from "react";
import { Loader2, Plus, Send } from "lucide-react";

type ComposerProps = {
  input: string;
  onInputChange: (value: string) => void;
  onKeyDown: (e: React.KeyboardEvent<HTMLTextAreaElement>) => void;
  onSend: () => void;
  uploadInputRef: RefObject<HTMLInputElement>;
  onFilePicked: (file: File) => void;
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
  busy,
  fileBusy,
}: ComposerProps) {
  return (
    <div className="px-4 lg:px-6 py-3 bg-gradient-to-t from-zinc-50 via-zinc-50/90 to-transparent dark:from-zinc-900 dark:via-zinc-900/90">
      <div className="mx-auto max-w-3xl">
        <div className="rounded-2xl border border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950 shadow-sm p-2">
          <div className="flex items-end gap-2">
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
                ref={uploadInputRef}
                onChange={(e) => e.target.files && onFilePicked(e.target.files[0])}
                disabled={fileBusy}
              />
              {fileBusy ? <Loader2 className="w-5 h-5 animate-spin" /> : <Plus className="w-5 h-5" />}
            </label>

            <textarea
              value={input}
              onChange={(e) => onInputChange(e.target.value)}
              onKeyDown={onKeyDown}
              rows={1}
              placeholder={fileBusy ? "Extracting text from file..." : "Paste lab text or type your question..."}
              className="flex-1 max-h-40 h-12 resize-none bg-transparent px-2 py-2 outline-none placeholder:text-zinc-400 text-sm"
              disabled={fileBusy}
            />

            <button
              onClick={onSend}
              className="shrink-0 inline-flex items-center gap-2 rounded-xl px-3 py-2 text-sm font-medium bg-zinc-900 text-white dark:bg-zinc-100 dark:text-zinc-900 hover:opacity-90 disabled:opacity-40 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-zinc-400"
              disabled={!input.trim() || busy || fileBusy}
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
        </div>
      </div>
    </div>
  );
}
