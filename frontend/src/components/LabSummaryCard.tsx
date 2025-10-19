import React, { useMemo, useState } from 'react';
import type { ChatResponseCombined } from '../services/api';

type Props = {
  response?: ChatResponseCombined;
  aiText?: string; // fallback source for heuristics
  devMode?: boolean;
};

type Analyte = { name: string; value?: string; unit?: string; status?: string; ref?: string };

function parseAnalytesFromText(text?: string): Analyte[] {
  if (!text) return [];
  const lines = (text.split(/\n|\r/).map(l => l.trim()).filter(Boolean));
  const out: Analyte[] = [];
  const rx = /^([A-Za-z][A-Za-z\s/]+?)\s[—-]\s*([0-9]+(?:\.[0-9]+)?)\s*([^()\s]+)?\s*\(([^)]*)\)\s*$/;
  for (const ln of lines) {
    const m = ln.match(rx);
    if (!m) continue;
    const [, nameRaw, val, unit, inside] = m;
    const name = nameRaw.trim();
    const statusMatch = inside.match(/(High|Low|Normal|Desirable)/i);
    const status = statusMatch ? statusMatch[1][0].toUpperCase() + statusMatch[1].slice(1).toLowerCase() : undefined;
    // grab any reference text after a ';' or starting with 'ref'
    let ref = '';
    const semi = inside.indexOf(';');
    if (semi >= 0) ref = inside.slice(semi + 1).trim();
    else if (/ref/i.test(inside)) ref = inside.trim();
    out.push({ name, value: val, unit, status, ref });
  }
  return out;
}

function pickRecommendation(resp?: ChatResponseCombined, aiText?: string): string {
  const acts = resp?.local_recommendations?.actions;
  if (acts && acts.length) {
    return acts[0].length > 200 ? acts[0].slice(0, 197) + '…' : acts[0];
  }
  const follow = resp?.local_recommendations?.follow_up;
  if (follow) return follow.length > 200 ? follow.slice(0, 197) + '…' : follow;
  // fallback: first sentence from ai text
  const txt = (aiText || '').split(/\.(\s|$)/)[0]?.trim();
  return txt ? (txt.length > 200 ? txt.slice(0, 197) + '…' : txt) : 'Consider discussing these results with your clinician.';
}

function extractDate(text?: string): string | null {
  if (!text) return null;
  // match dd/mm/yyyy or dd-mm-yyyy
  const m = text.match(/\b(\d{1,2})[\/\-](\d{1,2})[\/\-](\d{4})\b/);
  if (!m) return null;
  const [ , d, mth, y] = m;
  const day = d.padStart(2, '0');
  const mon = mth.padStart(2, '0');
  return `${day}/${mon}/${y}`;
}

function briefExplanation(aiText?: string): string {
  const txt = (aiText || '').trim();
  if (!txt) return '';
  // Take the first 1–2 sentences up to ~280 chars
  const parts = txt.split(/(?<=[\.\!\?])\s+/).filter(Boolean);
  let out = parts.slice(0, 2).join(' ').trim();
  if (out.length > 280) out = out.slice(0, 277).trimEnd() + '…';
  return out;
}

export default function LabSummaryCard({ response, aiText, devMode }: Props) {
  const uv = (response as any)?.user_view as ChatResponseCombined['user_view'] | undefined;
  const usingUV = !!uv && (Array.isArray(uv.abnormal) || Array.isArray(uv.normal));

  const analytes = useMemo(() => parseAnalytesFromText(aiText), [aiText]);
  const abFromText = analytes.filter(a => /high|low/i.test(a.status || ''));
  const nlFromText = analytes.filter(a => !/high|low/i.test(a.status || ''));

  const isMedicalItem = (a: any): boolean => {
    const nm = ((a?.name ?? '') + '').trim().toLowerCase();
    if (!nm) return false;
    // filter out legend/threshold headings
    if (/^(desirable|normal|borderline\s*high?|very\s*high|high:?|low:?)(\b|:|\s)/i.test(nm)) return false;
    const slug = nm.replace(/[^a-z0-9]+/g, '');
    if (["desirable","normal","borderline","borderlinehigh","veryhigh","high","low","reference","ref"].includes(slug)) return false;
    return true;
  };

  const abnormal = (usingUV ? (uv?.abnormal || []) : abFromText).filter(isMedicalItem);
  const normal = (usingUV ? (uv?.normal || []) : nlFromText).filter(isMedicalItem);

  const rec = usingUV ? (uv?.recommendation || pickRecommendation(response, aiText)) : pickRecommendation(response, aiText);
  const conf = usingUV ? uv?.confidence : (response as any)?.confidence;
  const saConf = response?.symptom_analysis?.confidence as number | undefined;
  const confidence = typeof conf === 'number' ? conf : (typeof saConf === 'number' ? saConf : undefined);
  const date = usingUV ? null : extractDate(aiText);
  const brief = usingUV ? ((uv as any)?.explanation || briefExplanation(aiText)) : '';
  const fullExplanation = usingUV ? (aiText || '') : '';

  function formatItem(a: any): string {
    const nm = a?.name ?? '';
    const v = a?.value ?? '';
    const u = a?.unit ? ` ${a.unit}` : '';
    const st = (a?.status || '').toString();
    let ref = '';
    if (typeof a?.ref_min === 'number' || typeof a?.ref_max === 'number') {
      const lo = typeof a.ref_min === 'number' ? a.ref_min : undefined;
      const hi = typeof a.ref_max === 'number' ? a.ref_max : undefined;
      if (typeof lo === 'number' && typeof hi === 'number') ref = `; ref ${lo}–${hi}`;
      else if (typeof hi === 'number') ref = `; ref ≤${hi}`;
      else if (typeof lo === 'number') ref = `; ref ≥${lo}`;
    } else if (a?.reference && typeof a.reference === 'object') {
      // Fallback for alternative reference shapes
      const k = a.reference.kind;
      if (k === 'between') ref = `; ref ${a.reference.lo}–${a.reference.hi}${a.reference.unit ? ' ' + a.reference.unit : ''}`;
      else if (k === 'lte') ref = `; ref ≤${a.reference.v}${a.reference.unit ? ' ' + a.reference.unit : ''}`;
      else if (k === 'lt') ref = `; ref <${a.reference.v}${a.reference.unit ? ' ' + a.reference.unit : ''}`;
      else if (k === 'gte') ref = `; ref ≥${a.reference.v}${a.reference.unit ? ' ' + a.reference.unit : ''}`;
      else if (k === 'gt') ref = `; ref >${a.reference.v}${a.reference.unit ? ' ' + a.reference.unit : ''}`;
    }
    const statusLabel = st ? (st[0].toUpperCase() + st.slice(1).toLowerCase()) : '';
    return `${nm} — ${v}${u} (${statusLabel}${ref})`;
  }

  return (
    <div className="text-sm">
      <div className="mb-2">
        <div className="text-base font-semibold">{usingUV ? (uv?.summary || 'Your lab summary') : `Your lab summary${date ? ` (${date})` : ''}`}</div>
      </div>
      {abnormal.length > 0 && (
        <div className="mb-1">
          <div className="text-xs uppercase tracking-wide text-red-600 dark:text-red-400 mb-1 flex items-center gap-2">
            <span>Abnormal</span>
            <span className="px-1.5 py-0.5 text-[10px] rounded-full bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200">{abnormal.length}</span>
          </div>
          <ul className="pl-0 space-y-1">
            {abnormal.map((a, i) => (
              <li key={`ab-${i}`} className="flex items-start gap-2 text-red-700 dark:text-red-300 font-medium">
                <span className="mt-1 inline-block w-1.5 h-1.5 rounded-full bg-red-500 dark:bg-red-400"/>
                <span>{formatItem(a)}</span>
              </li>
            ))}
          </ul>
        </div>
      )}
      {abnormal.length > 0 && normal.length > 0 && (
        <div className="my-2 h-px bg-zinc-200 dark:bg-zinc-800" />
      )}
      {normal.length > 0 && (
        <div className="mt-1">
          <div className="text-xs uppercase tracking-wide text-emerald-600 dark:text-emerald-400 mb-1 flex items-center gap-2">
            <span>Normal</span>
            <span className="px-1.5 py-0.5 text-[10px] rounded-full bg-emerald-100 text-emerald-700 dark:bg-emerald-900/40 dark:text-emerald-200">{normal.length}</span>
          </div>
          <ul className="pl-5 list-disc space-y-1">
            {normal.map((a, i) => (
              <li key={`nl-${i}`}>{formatItem(a)}</li>
            ))}
          </ul>
        </div>
      )}
      {!abnormal.length && !normal.length && (
        <ul className="list-disc pl-5 space-y-1">
          <li className="italic text-zinc-500">No detected results</li>
        </ul>
      )}
      <div className="mt-3">
        <div className="font-medium">Recommendations</div>
        <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-snug">{rec}</p>
      </div>
      <div className="mt-2 text-xs text-zinc-500">
        {typeof confidence === 'number' ? `Confidence: ${Math.round(confidence * 100)}%` : 'Confidence: -'}
      </div>

      {usingUV && (fullExplanation || brief) && (
        <div className="mt-3">
          <div className="font-medium">Explanation</div>
          <p className="text-sm text-zinc-700 dark:text-zinc-300 leading-snug whitespace-pre-wrap">
            {fullExplanation || brief}
          </p>
        </div>
      )}

      {devMode && response ? (
        <details className="mt-3">
          <summary className="cursor-pointer text-xs text-zinc-500">Developer Panel</summary>
          <div className="mt-2 space-y-2 text-xs">
            <div><span className="font-semibold">request_id:</span> {(response as any).request_id}</div>
            <div><span className="font-semibold">event_id:</span> {response.symptom_analysis?.event_id ?? ''}</div>
            <div><span className="font-semibold">ai_explanation_source:</span> {(response as any).ai_explanation_source ?? ''}</div>
            <div><span className="font-semibold">timed_out:</span> {String((response as any).timed_out ?? '')}</div>
            <div><span className="font-semibold">missing_fields:</span> {JSON.stringify((response as any).missing_fields ?? [])}</div>
            <div><span className="font-semibold">triage:</span> {JSON.stringify((response as any).triage ?? {})}</div>
            <div><span className="font-semibold">pipeline:</span>
              <pre className="whitespace-pre-wrap break-words bg-zinc-100 dark:bg-zinc-900 p-2 rounded-lg">{JSON.stringify((response as any).pipeline, null, 2)}</pre>
            </div>
            <div><span className="font-semibold">raw response:</span>
              <pre className="whitespace-pre-wrap break-words bg-zinc-100 dark:bg-zinc-900 p-2 rounded-lg">{JSON.stringify(response, null, 2)}</pre>
            </div>
          </div>
        </details>
      ) : null}
    </div>
  );
}
