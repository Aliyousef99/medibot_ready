import React, { useEffect, useRef, useState } from "react";
import { Loader2 } from "lucide-react";
import { updateProfile as apiUpdateProfile } from "../services/api";
import { useToastStore } from "../state/toastStore";
import type { User } from "../types";

type ProfileModalProps = {
  open: boolean;
  onClose: () => void;
  user: User;
  onUpdate: (profile: any) => void;
};

export default function ProfileModal({ open, onClose, user, onUpdate }: ProfileModalProps) {
  const [age, setAge] = useState<number | "">("");
  const [sex, setSex] = useState("");
  const [conditions, setConditions] = useState("");
  const [medications, setMedications] = useState("");
  const [saving, setSaving] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const dialogRef = useRef<HTMLDivElement>(null);
  const { add: addToast } = useToastStore();

  useEffect(() => {
    if (!open) return;
    const p = user.profile || {};
    setAge(p.age ?? "");
    setSex(p.sex ?? "");
    setConditions(Array.isArray(p.conditions) ? p.conditions.join(", ") : "");
    setMedications(Array.isArray(p.medications) ? p.medications.join(", ") : "");
    setMsg(null);

    // focus first input when opening
    const firstInput = dialogRef.current?.querySelector("input, select, textarea") as HTMLElement | null;
    firstInput?.focus();
  }, [open, user.profile]);

   // Escape to close and basic focus trap
  useEffect(() => {
    if (!open) return;
    const handler = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        onClose();
      }
      if (e.key === "Tab") {
        const focusable = dialogRef.current?.querySelectorAll<HTMLElement>(
          'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
        );
        if (!focusable || focusable.length === 0) return;
        const first = focusable[0];
        const last = focusable[focusable.length - 1];
        if (e.shiftKey) {
          if (document.activeElement === first) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (document.activeElement === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    };
    document.addEventListener("keydown", handler);
    return () => document.removeEventListener("keydown", handler);
  }, [open, onClose]);

  async function save() {
    // basic client-side validation
    if (age !== "" && (Number.isNaN(Number(age)) || Number(age) <= 0 || Number(age) > 120)) {
      addToast({ type: "error", message: "Please enter a valid age between 1 and 120." });
      return;
    }
    const normConditions = conditions
      ? conditions.split(",").map((s) => s.trim()).filter(Boolean)
      : [];
    const normMedications = medications
      ? medications.split(",").map((s) => s.trim()).filter(Boolean)
      : [];

    setSaving(true);
    setMsg(null);
    try {
      const updated = await apiUpdateProfile({
        age: age === "" ? null : Number(age),
        sex: sex || null,
        conditions: normConditions,
        medications: normMedications,
      });
      onUpdate(updated);
      setMsg("Profile saved successfully!");
      setTimeout(() => onClose(), 1000);
    } catch (e: any) {
      const message = e?.message || "Unknown error";
      setMsg(`Save failed: ${message}`);
      addToast({ type: "error", message: `Profile save failed: ${message}` });
    } finally {
      setSaving(false);
    }
  }

  if (!open) return null;

  return (
    <div
      className="fixed inset-0 z-50 bg-black/60 backdrop-blur-sm flex items-center justify-center p-4"
      role="dialog"
      aria-modal="true"
      aria-label="Profile settings"
      onMouseDown={(e) => {
        if (e.currentTarget === e.target) onClose();
      }}
    >
      <div
        ref={dialogRef}
        className="w-full max-w-xl rounded-3xl bg-gradient-to-br from-emerald-50 via-white to-white dark:from-zinc-900/90 dark:via-zinc-950 dark:to-zinc-950 border border-emerald-100 dark:border-zinc-800 shadow-2xl p-6"
      >
        <div className="flex items-center justify-between mb-4">
          <div>
            <p className="text-xs uppercase tracking-wide text-emerald-600 dark:text-emerald-300">Profile</p>
            <h2 className="text-xl font-semibold text-zinc-900 dark:text-zinc-50">Profile Settings</h2>
            <p className="text-sm text-zinc-500 dark:text-zinc-400">Tune these fields so recommendations stay personalized.</p>
          </div>
          <button
            onClick={onClose}
            className="rounded-full px-2 py-1 hover:bg-zinc-100 dark:hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            aria-label="Close"
          >
            &times;
          </button>
        </div>

        <div className="space-y-4">
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
            <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
              <span className="block font-medium">Age</span>
              <input
                type="number"
                value={age}
                onChange={(e) => setAge(e.target.value ? parseInt(e.target.value, 10) : "")}
                placeholder="e.g., 42"
                className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
                min={1}
                max={120}
              />
              <span className="text-[12px] text-zinc-400">Enter an age between 1 and 120.</span>
            </label>
            <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
              <span className="block font-medium">Sex</span>
              <select
                value={sex}
                onChange={(e) => setSex(e.target.value)}
                className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
              >
                <option value="">Select</option>
                <option value="male">male</option>
                <option value="female">female</option>
                <option value="other">other</option>
              </select>
            </label>
          </div>

          <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
            <span className="block font-medium">Conditions</span>
            <input
              value={conditions}
              onChange={(e) => setConditions(e.target.value)}
              placeholder="e.g., diabetes, hypertension"
              className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            />
            <span className="text-[12px] text-zinc-400">Comma-separated list.</span>
          </label>

          <label className="text-sm text-zinc-600 dark:text-zinc-300 space-y-1">
            <span className="block font-medium">Medications</span>
            <input
              value={medications}
              onChange={(e) => setMedications(e.target.value)}
              placeholder="e.g., metformin, lisinopril"
              className="w-full rounded-lg border px-3 py-2 bg-white dark:bg-zinc-950 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            />
            <span className="text-[12px] text-zinc-400">Comma-separated list.</span>
          </label>

          {msg && (
            <div className={`text-sm ${msg.startsWith("Save failed") ? "text-red-500" : "text-green-500"}`}>
              {msg}
            </div>
          )}

          <div className="pt-2 flex justify-end gap-2">
            <button
              type="button"
              onClick={onClose}
              className="rounded-xl px-3 py-2 border hover:bg-zinc-100 dark:hover:bg-zinc-900 focus:outline-none focus:ring-2 focus:ring-emerald-400"
            >
              Cancel
            </button>
            <button
              onClick={save}
              disabled={saving}
              className="rounded-xl px-4 py-2 bg-emerald-600 text-white hover:bg-emerald-500 disabled:opacity-60 flex items-center gap-2 shadow-sm focus:outline-none focus:ring-2 focus:ring-emerald-400"
            >
              {saving && <Loader2 className="w-4 h-4 animate-spin" />}
              {saving ? "Saving..." : "Save"}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
