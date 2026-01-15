import React from "react";
import { ShieldAlert, Copy, AlertTriangle, CheckCircle2, Loader2 } from "lucide-react";
import type { SymptomAnalysisResult, User } from "../types";
import type { RecommendationSet } from "../services/api";

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
  recommendations?: RecommendationSet | null;
  analysisState?: "idle" | "loading" | "error";
  analysisError?: string | null;
  systemPrompt?: string;
  systemPromptDraft?: string;
  onSystemPromptChange?: (value: string) => void;
  onSaveSystemPrompt?: () => void;
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
  recommendations,
  analysisState = "idle",
  analysisError,
  systemPrompt,
  systemPromptDraft,
  onSystemPromptChange,
  onSaveSystemPrompt,
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
      {!devMode && analysisState === "loading" && (
        <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 text-xs flex items-center gap-2 text-emerald-700 dark:text-emerald-200">
          <Loader2 className="w-4 h-4 animate-spin" />
          Generating recommendations...
        </div>
      )}
      {!devMode && analysisState === "error" && (
        <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 text-xs flex items-center gap-2 text-amber-700 dark:text-amber-200">
          <AlertTriangle className="w-4 h-4" />
          {analysisError || "Unable to generate recommendations."}
        </div>
      )}
      {!devMode && recommendations && (
        <div className="px-4 py-3 border-b border-zinc-200 dark:border-zinc-800 text-xs flex items-center gap-2">
          {recommendations.risk_tier === "high" ? (
            <ShieldAlert className="w-4 h-4 text-red-500" title="High risk" />
          ) : recommendations.risk_tier === "moderate" ? (
            <AlertTriangle className="w-4 h-4 text-amber-500" title="Moderate risk" />
          ) : (
            <CheckCircle2 className="w-4 h-4 text-emerald-500" title="Low risk" />
          )}
          <span className="font-semibold capitalize">{recommendations.risk_tier} risk</span>
          <span className="ml-auto text-[11px] text-zinc-400">
            AI source: {explanationSource === "fallback" ? "fallback" : explanationSource || "model"}
          </span>
        </div>
      )}
      {!devMode && !recommendations && analysisState === "idle" && (
        <div className="px-4 py-3 text-xs text-zinc-500 dark:text-zinc-400 border-b border-zinc-200 dark:border-zinc-800">
          Send a message or upload labs to see recommendations here.
        </div>
      )}
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
          <div className="text-xs uppercase tracking-wide text-zinc-400">Conversation System Prompt</div>
          <div className="rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900 space-y-2">
            <textarea
              className="w-full min-h-[120px] text-xs bg-transparent outline-none resize-y"
              placeholder="Add per-conversation system instructions here..."
              value={systemPromptDraft ?? systemPrompt ?? ""}
              onChange={(e) => onSystemPromptChange?.(e.target.value)}
            />
            <div className="flex items-center justify-end">
              <button
                type="button"
                className="text-[11px] px-2 py-1 rounded border border-zinc-300 dark:border-zinc-700 hover:bg-zinc-100 dark:hover:bg-zinc-800"
                onClick={onSaveSystemPrompt}
                disabled={!onSaveSystemPrompt}
              >
                Save prompt
              </button>
            </div>
          </div>
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
          {recommendations && (
            <div className="space-y-2 text-sm rounded-xl border border-zinc-200 dark:border-zinc-800 p-3 bg-zinc-50 dark:bg-zinc-900">
              <div className="flex items-center gap-2 text-xs uppercase tracking-wide text-zinc-400">
                <span>Recommendations</span>
                <span className="ml-auto inline-flex items-center gap-2 px-2 py-0.5 rounded-full bg-zinc-100 dark:bg-zinc-800 text-[11px]">
                  {recommendations.risk_tier === "high" ? (
                    <ShieldAlert className="w-3.5 h-3.5 text-red-500" />
                  ) : recommendations.risk_tier === "moderate" ? (
                    <AlertTriangle className="w-3.5 h-3.5 text-amber-500" />
                  ) : (
                    <CheckCircle2 className="w-3.5 h-3.5 text-emerald-500" />
                  )}
                  Tier: {recommendations.risk_tier}
                </span>
              </div>
              <ul className="space-y-1">
                {(recommendations.actions || []).map((a, i) => (
                  <li key={i} className="flex items-start gap-2">
                    <Copy className="w-3 h-3 mt-1 text-emerald-500" />
                    <span>{a}</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>
      )}
    </section>
  );
}
