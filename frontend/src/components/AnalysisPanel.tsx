import React from "react";
import { ShieldAlert, Copy } from "lucide-react";
import type { SymptomAnalysisResult, User } from "../types";

type AnalysisPanelProps = {
  devMode: boolean;
  onToggleDevMode: (value: boolean) => void;
  structured: any;
  explanation: string;
  explanationSource?: string;
  user: User | null;
  sanitizedExplanation: string;
  prettyRef: (ref?: any) => string;
  activeId: string | null;
  symptomAnalysisResult: SymptomAnalysisResult | null;
};

export default function AnalysisPanel({
  devMode,
  onToggleDevMode,
  structured,
  explanation,
  explanationSource,
  user,
  sanitizedExplanation,
  prettyRef,
  activeId,
  symptomAnalysisResult,
}: AnalysisPanelProps) {
  return (
    <section
      className={`hidden lg:flex flex-col overflow-hidden border-l border-zinc-200 dark:border-zinc-800 bg-white dark:bg-zinc-950`}
      aria-label="Analysis panel"
    >
      <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 text-sm font-semibold flex items-center gap-3">
        <span>Analysis</span>
        <label className="ml-auto inline-flex items-center gap-2 text-xs text-zinc-600 dark:text-zinc-400">
          <input type="checkbox" checked={devMode} onChange={(e) => onToggleDevMode(e.target.checked)} />
          Dev Mode (Ctrl+D)
        </label>
      </div>
      {devMode && (
        <div className="flex-1 min-h-0 overflow-y-auto p-3 space-y-3">
          <div className="text-xs uppercase tracking-wide text-zinc-400">Structured (BioBERT-style)</div>
          {structured ? (
            <pre className="text-xs whitespace-pre-wrap break-words rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900">
              {JSON.stringify(structured, null, 2)}
            </pre>
          ) : (
            <div className="rounded-xl border border-dashed border-zinc-300 dark:border-zinc-700 p-3 text-xs text-zinc-500">
              No structured data yet. Send a message or upload labs to see analysis here.
            </div>
          )}

          <div className="text-xs uppercase tracking-wide text-zinc-400">Parsed values</div>
          <div className="text-[11px] text-zinc-400 mb-1">Language: {structured?.language || "unknown"}</div>
          <ul className="text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 space-y-2">
            {(structured?.tests || []).length ? (
              (structured?.tests || []).map((t: any, i: number) => (
                <li key={i} className="flex items-center justify-between gap-2">
                  <div>
                    <span className="font-medium">{t.name}</span>: {t.value} {t.unit}
                    {t.reference ? ` (${prettyRef(t.reference)})` : ""}
                  </div>
                  <span
                    className={`text-[11px] px-2 py-0.5 rounded-full border ${
                      t.status === "high"
                        ? "border-red-400 text-red-600"
                        : t.status === "low"
                        ? "border-blue-400 text-blue-600"
                        : t.status === "normal"
                        ? "border-emerald-400 text-emerald-600"
                        : "border-zinc-300 text-zinc-500"
                    }`}
                  >
                    {t.status}
                  </span>
                </li>
              ))
            ) : (
              <li className="text-xs text-zinc-500 italic">No lab values parsed.</li>
            )}
          </ul>

          <div className="text-xs uppercase tracking-wide text-zinc-400">Detected terms</div>
          <ul className="text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 space-y-2">
            {structured?.entities?.length ? (
              structured.entities.map((ent: any, i: number) => (
                <li key={i}>
                  <div className="flex flex-col gap-1">
                    <div className="flex items-center justify-between">
                      <span className="font-medium">{ent.term || ent.text}</span>
                      <span className="text-[11px] text-zinc-400">
                        {ent.label || ent.entity}
                        {ent.source ? ` | ${ent.source === "gemini" ? "Gemini" : "Glossary"}` : ""}
                      </span>
                    </div>
                    {ent.explanation ? (
                      <p className="text-[13px] text-zinc-500 dark:text-zinc-400">{ent.explanation}</p>
                    ) : (
                      <p className="text-[13px] text-zinc-500 dark:text-zinc-400 italic">No glossary description available.</p>
                    )}
                  </div>
                </li>
              ))
            ) : (
              <li className="text-[13px] text-zinc-500 dark:text-zinc-400 italic">No medical terms detected.</li>
            )}
          </ul>

          <div className="flex items-center justify-between">
            <div className="text-xs uppercase tracking-wide text-zinc-400">AI Explanation</div>
            <div className="flex items-center gap-2">
              {process.env.NODE_ENV !== "production" && (
                <span className="text-[10px] text-zinc-400">source: {explanationSource || "unknown"}</span>
              )}
              <button
                className="text-[11px] px-2 py-0.5 rounded border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800 flex items-center gap-1 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-zinc-400"
                onClick={() => {
                  try {
                    navigator.clipboard.writeText(explanation || "");
                  } catch {
                    // ignore
                  }
                }}
                title="Copy AI explanation"
                aria-label="Copy AI explanation"
              >
                <Copy className="w-3 h-3" /> Copy
              </button>
            </div>
          </div>
          <div className="text-xs uppercase tracking-wide text-zinc-400">User Profile</div>
          <pre className="text-xs whitespace-pre-wrap break-words rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900">
            {user?.profile ? JSON.stringify(user.profile, null, 2) : "N/A"}
          </pre>
          <div className="text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 min-h-[120px]">
            {sanitizedExplanation || "No explanation generated yet."}
          </div>
        </div>
      )}
      {devMode && symptomAnalysisResult && (
        <div className="p-3 space-y-3">
          <div className="text-xs uppercase tracking-wide text-zinc-400">Symptom Analysis</div>
          <div
            className={`text-sm rounded-xl border p-3 bg-zinc-50 dark:bg-zinc-900 ${
              symptomAnalysisResult.urgency === "urgent" ? "border-red-500" : "border-zinc-200 dark:border-zinc-800"
            }`}
          >
            <div className="flex items-center gap-2 mb-2">
              <ShieldAlert
                className={`w-5 h-5 ${
                  symptomAnalysisResult.urgency === "urgent" ? "text-red-500" : "text-amber-500"
                }`}
              />
              <span className="font-semibold">Urgency: {symptomAnalysisResult.urgency}</span>
            </div>
            <p className="text-xs text-zinc-500 dark:text-zinc-400 mb-3">{symptomAnalysisResult.summary}</p>
            <ul className="space-y-1 text-xs">
              {symptomAnalysisResult.symptoms.map((s, i) => (
                <li key={i} className={`flex justify-between ${s.negated ? "line-through text-zinc-400" : ""}`}>
                  <span>{s.text}</span>
                  <span className="font-mono text-zinc-500">
                    {s.label} {(s.score * 100).toFixed(0)}%
                  </span>
                </li>
              ))}
              {symptomAnalysisResult.symptoms.length === 0 && (
                <li className="italic text-zinc-400">No specific symptoms detected.</li>
              )}
            </ul>
          </div>
        </div>
      )}
    </section>
  );
}
